# ============================
# File: train_agent_aimaway.py
# ============================
#!/usr/bin/env python3
"""
Train a PPO policy on PettingZoo Atari Pong (multi-agent) where our model
controls one paddle and a heuristic controls the other.

This version adds **aim-away** bias in two ways:
  1) Potential-based reward shaping that rewards trajectories where, when the
     ball is heading to the opponent, its predicted landing point is far from
     the opponent's paddle center.
  2) An optional near-contact action override that tries to send the ball away
     from the opponent's current position by forcing an extreme (UP/DOWN) just
     before impact.

Deps (venv recommended):
    python -m pip install "pettingzoo[atari]" SuperSuit "gymnasium[accept-rom-license]" ale-py autorom
    python -m pip install stable-baselines3 opencv-python
    # plus PyTorch from https://pytorch.org
    autorom --accept-license

Usage examples:
    # pure shaping
    python train_agent_aimaway.py --timesteps 2000000 --learn_as first_0 --aimaway --aimaway_coef 0.01

    # shaping + surgical override
    python train_agent_aimaway.py --timesteps 2000000 --learn_as second_0 --aimaway --aimaway_override --override_prob 0.5

Saves model to: ppo_pong_ma.zip (default)
"""
from __future__ import annotations
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.atari import pong_v3
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.utils import set_random_seed

# ---------- Env Builders ----------

def make_pz_env(render_mode=None, frame_skip: int = 4):
    env = pong_v3.env(render_mode=render_mode)
    env = ss.frame_skip_v0(env, max(1, int(frame_skip)))
    env = ss.color_reduction_v0(env, "full")  # grayscale
    env = ss.resize_v1(env, 84, 84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.dtype_v0(env, dtype="uint8")
    return env

def get_action_meanings(env, agent_name="first_0"):
    try:
        meanings = env.unwrapped.get_action_meanings(agent_name)
        return [m.upper() for m in meanings]
    except Exception:
        return ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

def calibrate_up_down(env, agent_name="first_0"):
    meanings = get_action_meanings(env, agent_name)
    up_candidates, down_candidates = ["UP", "RIGHT"], ["DOWN", "LEFT"]
    up_idx = next((i for i,m in enumerate(meanings) if m in up_candidates), 2)
    dn_idx = next((i for i,m in enumerate(meanings) if m in down_candidates), 3)
    return up_idx, dn_idx

# ---------- Opponent & Helper geometry ----------

class HeuristicOpponent:
    def __init__(self, up_idx: int, down_idx: int, noop_idx: int = 0):
        self.up_idx, self.down_idx, self.noop_idx = up_idx, down_idx, noop_idx
    def act(self, last_obs: np.ndarray, is_left_side: bool) -> int:
        # last_obs: (84,84,4) uint8; use newest frame
        f = last_obs[..., -1]
        ys, xs = np.where(f > 200)
        if len(ys) == 0:
            return self.noop_idx
        ball_y = int(np.median(ys))
        h, w = f.shape
        col = 6 if is_left_side else w - 7
        col = int(np.clip(col, 0, w - 1))
        paddle_pixels = np.where(f[:, col] > 180)[0]
        if len(paddle_pixels) == 0:
            return self.noop_idx
        pad_y = int(np.mean(paddle_pixels))
        if ball_y < pad_y - 2: return self.up_idx
        if ball_y > pad_y + 2: return self.down_idx
        return self.noop_idx

# --- simple pixel geometry helpers ---

def _extract_ball_yx_from_frame(frame: np.ndarray) -> tuple[int | None, int | None]:
    ys, xs = np.where(frame > 200)
    if len(ys) == 0:
        return None, None
    return int(np.median(ys)), int(np.median(xs))

def _paddle_y(frame: np.ndarray, is_left: bool) -> int | None:
    h, w = frame.shape
    col = 6 if is_left else w - 7
    col = int(np.clip(col, 0, w - 1))
    pixels = np.where(frame[:, col] > 180)[0]
    if len(pixels) == 0:
        return None
    return int(np.mean(pixels))

def _ball_velocity_from_obs(obs: np.ndarray) -> tuple[float | None, float | None, int | None, int | None]:
    # obs: (84,84,4) stacked old->new along channel; newest is -1
    f2 = obs[..., -1]
    f1 = obs[..., -2]
    y2, x2 = _extract_ball_yx_from_frame(f2)
    y1, x1 = _extract_ball_yx_from_frame(f1)
    if y2 is None or y1 is None or x2 is None or x1 is None:
        return None, None, y2, x2
    return float(x2 - x1), float(y2 - y1), y2, x2

def _reflect_y(y: float, h: int) -> float:
    # Triangular wave reflection into [0, h-1]
    if h <= 1:
        return 0.0
    period = 2 * (h - 1)
    y_mod = y % period
    if y_mod <= (h - 1):
        return y_mod
    return period - y_mod

def _predict_landing_y_on_column(y: float, x: float, vy: float, vx: float, x_target: float, h: int) -> float | None:
    if vx is None or vx == 0:
        return None
    dt = (x_target - x) / vx
    # if dt is negative, we're moving away; treat as None
    if dt <= 0:
        return None
    y_final = y + vy * dt
    return _reflect_y(y_final, h)

# ---------- Single-agent wrapper with aim-away shaping/override ----------

class SingleAgentVsPolicy(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, seed=0, learning_agent="first_0", opponent_policy_fn=None,
                 render_mode=None, frame_skip: int = 4,
                 aimaway_coef: float = 0.0, shaping_gamma: float = 0.99,
                 aimaway_override: bool = False, override_prob: float = 1.0,
                 override_zone_px: int = 14):
        super().__init__()
        self.learning_agent = learning_agent
        self.opponent_name = "second_0" if learning_agent == "first_0" else "first_0"
        self.env = make_pz_env(render_mode=render_mode, frame_skip=frame_skip)
        self.env.reset(seed=seed)
        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)
        up_idx, dn_idx = calibrate_up_down(self.env, self.learning_agent)
        self.ACTION = {"NOOP": 0, "FIRE": 1, "UP": up_idx, "DOWN": dn_idx}
        self.opponent_policy_fn = opponent_policy_fn
        self._last_obs = {a: None for a in self.env.agents}

        # Aim-away knobs
        self.aimaway_coef = float(aimaway_coef)
        self.shaping_gamma = float(shaping_gamma)
        self.aimaway_override = bool(aimaway_override)
        self.override_prob = float(override_prob)
        self.override_zone_px = int(override_zone_px)

    # ---- shaping potential Φ(s): bigger when predicted landing is far from opponent paddle ----
    def _phi(self, obs: np.ndarray) -> float:
        try:
            f = obs[..., -1]
            h, w = f.shape
            vx, vy, by, bx = _ball_velocity_from_obs(obs)
            if by is None or bx is None or vx is None or vy is None:
                return 0.0
            # Determine sides
            i_am_left = (self.learning_agent == "first_0")
            ball_toward_opponent = (vx > 0) if i_am_left else (vx < 0)
            if not ball_toward_opponent:
                return 0.0
            # target column is opponent gutter
            x_target = w - 2 if i_am_left else 1
            y_pred = _predict_landing_y_on_column(by, bx, vy, vx, x_target, h)
            if y_pred is None:
                return 0.0
            opp_y = _paddle_y(f, is_left=(not i_am_left))
            if opp_y is None:
                return 0.0
            d = abs(y_pred - opp_y) / max(1, h - 1)
            return float(np.clip(d, 0.0, 1.0))
        except Exception:
            return 0.0

    # ---- near-contact action override to send ball away from opponent ----
    def _override_action_if_needed(self, obs: np.ndarray, proposed_action: int) -> int:
        if not self.aimaway_override or np.random.rand() > self.override_prob:
            return proposed_action
        f = obs[..., -1]
        h, w = f.shape
        vx, vy, by, bx = _ball_velocity_from_obs(obs)
        if by is None or bx is None or vx is None or vy is None:
            return proposed_action
        i_am_left = (self.learning_agent == "first_0")
        # Is ball incoming to me and close to my x-side?
        incoming = (vx < 0) if i_am_left else (vx > 0)
        near = (bx <= self.override_zone_px) if i_am_left else (bx >= w - 1 - self.override_zone_px)
        if not (incoming and near):
            return proposed_action
        opp_y = _paddle_y(f, is_left=(not i_am_left))
        if opp_y is None:
            return proposed_action
        # Coarse strategy: if opponent is high, aim DOWN; if low, aim UP
        if opp_y < h // 2:
            return self.ACTION["DOWN"]
        else:
            return self.ACTION["UP"]

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        obs_learning = None
        for agent in self.env.agent_iter():
            obs, r, t, tr, info = self.env.last()
            self._last_obs[agent] = obs
            if agent == self.learning_agent:
                obs_learning = obs
                self.env.step(self.ACTION["NOOP"]) if not (t or tr) else self.env.step(None)
                break
            else:
                self.env.step(self.ACTION["NOOP"]) if not (t or tr) else self.env.step(None)
        if obs_learning is None:
            import numpy as _np
            obs_learning = _np.zeros(self.observation_space.shape, dtype=_np.uint8)
        return obs_learning, {}

    def step(self, action: int):
        total_r, termd, truncd = 0.0, False, False
        info_acc = {}
        phi_before = None
        # --- first sweep: each agent acts once ---
        for agent in self.env.agent_iter():
            obs, r, t, tr, info = self.env.last()
            self._last_obs[agent] = obs
            if agent == self.learning_agent:
                total_r += float(r)
                if t or tr:
                    self.env.step(None)
                    termd |= t; truncd |= tr
                    info_acc.update(info)
                    break
                # potential before
                if self.aimaway_coef > 0.0:
                    phi_before = self._phi(obs)
                # optional action override near contact
                action_to_take = self._override_action_if_needed(obs, int(action))
                self.env.step(int(action_to_take))
            else:
                if t or tr:
                    self.env.step(None)
                    termd |= t; truncd |= tr
                    info_acc.update(info)
                    continue
                is_left = (agent == "first_0")
                last = self._last_obs.get(agent)
                opp_action = self.ACTION["NOOP"] if self.opponent_policy_fn is None else int(self.opponent_policy_fn(last, is_left))
                self.env.step(opp_action)
            if agent != self.learning_agent:
                break

        # --- second sweep: advance to get next_obs for learner ---
        next_obs = None
        for agent in self.env.agent_iter():
            obs, r, t, tr, info = self.env.last()
            self._last_obs[agent] = obs
            if agent == self.learning_agent:
                next_obs = obs
                total_r += float(r)
                termd |= t; truncd |= tr
                info_acc.update(info)
                # satisfy AEC contract: always step once before break
                if t or tr: self.env.step(None)
                else:       self.env.step(self.ACTION["NOOP"])
                break
            if not (t or tr): self.env.step(self.ACTION["NOOP"]) 
            else:             self.env.step(None)

        if next_obs is None:
            import numpy as _np
            next_obs = _np.zeros(self.observation_space.shape, dtype=_np.uint8)

        # --- add potential-based shaping r += α (γ Φ(s') - Φ(s)) ---
        if self.aimaway_coef > 0.0:
            try:
                phi_after = self._phi(next_obs)
                phi_before = 0.0 if phi_before is None else float(phi_before)
                shaping = self.aimaway_coef * (self.shaping_gamma * float(phi_after) - phi_before)
                total_r += shaping
                info_acc["aimaway_shaping"] = shaping
            except Exception:
                pass

        return next_obs, total_r, termd, truncd, info_acc

    def close(self):
        self.env.close()

# ---------- Training Harness ----------

def make_training_env(seed: int, learn_as: str, frame_skip: int = 4, n_envs: int = 8,
                      aimaway_coef: float = 0.0, shaping_gamma: float = 0.99,
                      aimaway_override: bool = False, override_prob: float = 1.0,
                      override_zone_px: int = 14):
    probe = make_pz_env(frame_skip=frame_skip)
    up_idx, dn_idx = calibrate_up_down(probe, learn_as)
    probe.close()
    heuristic = HeuristicOpponent(up_idx=up_idx, down_idx=dn_idx, noop_idx=0)
    def thunk(i):
        def _t():
            return SingleAgentVsPolicy(
                seed=seed + i,
                learning_agent=learn_as,
                opponent_policy_fn=heuristic.act,
                render_mode=None,
                frame_skip=frame_skip,
                aimaway_coef=aimaway_coef,
                shaping_gamma=shaping_gamma,
                aimaway_override=aimaway_override,
                override_prob=override_prob,
                override_zone_px=override_zone_px,
            )
        return _t
    envs = [thunk(i) for i in range(n_envs)]
    venv = DummyVecEnv(envs)
    venv = VecMonitor(venv)
    venv = VecTransposeImage(venv)
    return venv


def train_model(timesteps: int, save_path: str, seed: int, learn_as: str, use_cuda: bool,
                n_envs: int, n_steps: int, batch_size: int | None, use_compile: bool, compile_mode: str,
                use_adamw: bool, aimaway: bool, aimaway_coef: float, aimaway_override: bool,
                override_prob: float, override_zone_px: int):
    set_random_seed(seed)
    device = "cuda"
    if use_cuda:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Using CUDA" if device == "cuda" else "CUDA requested but not available; using CPU")
        except Exception:
            device = "cpu"

    # Keep PPO gamma and shaping gamma aligned by default
    ppo_gamma = 0.99

    env = make_training_env(
        seed=seed,
        learn_as=learn_as,
        frame_skip=4,
        n_envs=n_envs,
        aimaway_coef=(aimaway_coef if aimaway else 0.0),
        shaping_gamma=ppo_gamma,
        aimaway_override=aimaway_override,
        override_prob=override_prob,
        override_zone_px=override_zone_px,
    )

    rollout = n_steps * n_envs
    if batch_size is None:
        # pick a divisor of rollout close to rollout/4
        cand = max(128, rollout // 4)
        while rollout % cand != 0 and cand > 32:
            cand //= 2
        batch_size = cand

    import torch
    opt_class = torch.optim.AdamW if use_adamw else torch.optim.Adam

    model = PPO(
        "CnnPolicy", env=env, device=device, verbose=1, learning_rate=2.5e-4,
        n_steps=n_steps, batch_size=batch_size, n_epochs=4, gamma=ppo_gamma, gae_lambda=0.95,
        ent_coef=0.01, vf_coef=0.5, clip_range=0.1, tensorboard_log="./tb_pong_ma/",
        policy_kwargs=dict(
            optimizer_class=opt_class,
            optimizer_kwargs=dict(weight_decay=1e-4, eps=1e-5) if use_adamw else dict(eps=1e-5)
        ),
    )

    if use_compile:
        try:
            if hasattr(torch, "compile"):
                try: torch.set_float32_matmul_precision("high")
                except Exception: pass
                model.policy.features_extractor = torch.compile(model.policy.features_extractor, mode=compile_mode, fullgraph=False, dynamic=True)
                model.policy.mlp_extractor = torch.compile(model.policy.mlp_extractor, mode=compile_mode, fullgraph=False, dynamic=True)
                model.policy.action_net = torch.compile(model.policy.action_net, mode=compile_mode, fullgraph=False, dynamic=True)
                model.policy.value_net = torch.compile(model.policy.value_net, mode=compile_mode, fullgraph=False, dynamic=True)
                print(f"✅ torch.compile enabled (mode={compile_mode})")
            else:
                print("⚠️ PyTorch < 2.0; torch.compile unavailable")
        except Exception as e:
            print(f"⚠️ torch.compile failed gracefully: {e}")

    print(f"Rollout={rollout}, batch_size={batch_size}")
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    env.close()
    return save_path


def main():
    ap = argparse.ArgumentParser(description="Train PPO on Pong (multi-agent wrapper) with aim-away shaping/override")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--save_path", type=str, default="ppo_pong_ma.zip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learn_as", type=str, default="first_0", choices=["first_0", "second_0"]) 
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--n_steps", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile_mode", type=str, default="reduce-overhead", choices=["reduce-overhead", "max-autotune", "default"]) 
    ap.add_argument("--adamw", action="store_true", help="Use AdamW instead of Adam")

    # Aim-away options
    ap.add_argument("--aimaway", action="store_true", help="Enable potential-based aim-away reward shaping")
    ap.add_argument("--aimaway_coef", type=float, default=0.01, help="Coefficient α for shaping term α(γΦ(s')-Φ(s))")
    ap.add_argument("--aimaway_override", action="store_true", help="Enable near-contact action override to bias away-from-opponent shots")
    ap.add_argument("--override_prob", type=float, default=1.0, help="Probability to apply the override when the trigger condition fires")
    ap.add_argument("--override_zone_px", type=int, default=14, help="Horizontal proximity (in px) to consider 'near-contact' for override")

    args = ap.parse_args()

    path = train_model(
        args.timesteps, args.save_path, args.seed, args.learn_as, args.cuda,
        args.n_envs, args.n_steps, args.batch_size, args.compile, args.compile_mode, args.adamw,
        aimaway=args.aimaway, aimaway_coef=args.aimaway_coef,
        aimaway_override=args.aimaway_override, override_prob=args.override_prob,
        override_zone_px=args.override_zone_px,
    )
    print(f"Saved to {path}")

if __name__ == "__main__":
    main()
