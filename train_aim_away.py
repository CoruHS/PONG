# ============================
# File: train_agent.py
# ============================
#!/usr/bin/env python3
"""
Train a PPO policy on PettingZoo Atari Pong (multi-agent) where our model
controls one paddle and a heuristic controls the other.

Deps (venv recommended):
    python -m pip install "pettingzoo[atari]" SuperSuit "gymnasium[accept-rom-license]" ale-py autorom
    python -m pip install stable-baselines3 opencv-python
    # plus PyTorch from https://pytorch.org
    autorom --accept-license

Usage:
    python train_agent.py --timesteps 2000000 --learn_as first_0 --cuda

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

# ---------- Opponent & Wrapper ----------

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

class SingleAgentVsPolicy(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, seed=0, learning_agent="first_0", opponent_policy_fn=None, render_mode=None, frame_skip: int = 4):
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
                self.env.step(int(action))
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
        return next_obs, total_r, termd, truncd, info_acc
    def close(self):
        self.env.close()

# ---------- Training Harness ----------

def make_training_env(seed: int, learn_as: str, frame_skip: int = 4, n_envs: int = 8):
    probe = make_pz_env(frame_skip=frame_skip)
    up_idx, dn_idx = calibrate_up_down(probe, learn_as)
    probe.close()
    heuristic = HeuristicOpponent(up_idx=up_idx, down_idx=dn_idx, noop_idx=0)
    def thunk(i):
        def _t():
            return SingleAgentVsPolicy(seed=seed + i, learning_agent=learn_as, opponent_policy_fn=heuristic.act, render_mode=None, frame_skip=frame_skip)
        return _t
    envs = [thunk(i) for i in range(n_envs)]
    venv = DummyVecEnv(envs)
    venv = VecMonitor(venv)
    venv = VecTransposeImage(venv)
    return venv

def train_model(timesteps: int, save_path: str, seed: int, learn_as: str, use_cuda: bool,
                n_envs: int, n_steps: int, batch_size: int | None, use_compile: bool, compile_mode: str,
                use_adamw: bool):
    set_random_seed(seed)
    device = "cpu"
    if use_cuda:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Using CUDA" if device == "cuda" else "CUDA requested but not available; using CPU")
        except Exception:
            device = "cpu"
    env = make_training_env(seed=seed, learn_as=learn_as, frame_skip=4, n_envs=n_envs)
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
        n_steps=n_steps, batch_size=batch_size, n_epochs=4, gamma=0.99, gae_lambda=0.95,
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
    ap = argparse.ArgumentParser(description="Train PPO on Pong (multi-agent wrapper)")
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
    args = ap.parse_args()
    path = train_model(args.timesteps, args.save_path, args.seed, args.learn_as, args.cuda,
                       args.n_envs, args.n_steps, args.batch_size, args.compile, args.compile_mode, args.adamw)
    print(f"Saved to {path}")

if __name__ == "__main__":
    main()
