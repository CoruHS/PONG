#!/usr/bin/env python3
"""
Play human (keyboard) vs the trained PPO model (PettingZoo Pong).

Features:
- Hold-to-glide movement (hold ↑/↓ or W/S to move smoothly)
- Debounced Space: serve once per press, mapped to the human side's FIRE index
- Auto-serve after each point to avoid hanging
- Robust against None observations on serve/terminal turns
- Frameskip=1 for precise control
- Uses pynput for keyboard (no root)

Install (once):
    python -m pip install pynput "pettingzoo[atari]" SuperSuit "gymnasium[accept-rom-license]" ale-py stable-baselines3
    autorom --accept-license

Run:
    python play_vs_model.py --model_path ppo_pong_ma.zip --model_side first_0
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import gymnasium as gym
from pettingzoo.atari import pong_v3
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# ---------------- Keyboard (pynput) ----------------

try:
    from pynput import keyboard as _kb
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False

class KeyState:
    """Tracks pressed keys via a background listener (no root)."""
    def __init__(self):
        self.down = set()
        if not PYNPUT_AVAILABLE:
            self.listener = None
            return
        self.listener = _kb.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.daemon = True
        self.listener.start()

    def _on_press(self, key):
        try:
            ch = key.char.lower() if getattr(key, "char", None) else None
        except Exception:
            ch = None
        if ch: self.down.add(ch)
        if key == _kb.Key.up: self.down.add("up")
        if key == _kb.Key.down: self.down.add("down")
        if key == _kb.Key.space: self.down.add("space")

    def _on_release(self, key):
        try:
            ch = key.char.lower() if getattr(key, "char", None) else None
            if ch: self.down.discard(ch)
        except Exception:
            pass
        if key == _kb.Key.up: self.down.discard("up")
        if key == _kb.Key.down: self.down.discard("down")
        if key == _kb.Key.space: self.down.discard("space")

    def pressed(self, name: str) -> bool:
        return name in self.down


# ---------------- Env helpers ----------------

def make_pz_env(render_mode=None, frame_skip: int = 1):
    """PettingZoo Pong preprocessed for SB3; frameskip=1 for human play."""
    env = pong_v3.env(render_mode=render_mode)
    env = ss.frame_skip_v0(env, max(1, int(frame_skip)))  # 1 = precise control
    env = ss.color_reduction_v0(env, "full")              # grayscale
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

def calibrate_indices(env, agent_name="first_0"):
    """Return (UP, DOWN, FIRE) indices for a given PettingZoo agent."""
    meanings = get_action_meanings(env, agent_name)
    up_idx   = next((i for i,m in enumerate(meanings) if m in ("UP","RIGHT")), 2)
    down_idx = next((i for i,m in enumerate(meanings) if m in ("DOWN","LEFT")), 3)
    try:
        fire_idx = next(i for i,m in enumerate(meanings) if m == "FIRE")
    except StopIteration:
        fire_idx = 1  # ALE fallback
    return up_idx, down_idx, fire_idx

def newest_frame(obs):
    """Safely get newest 84x84 frame from stacked obs or None."""
    if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[-1] >= 1:
        return obs[..., -1]
    return None

def is_ball_visible_obs(obs) -> bool:
    f = newest_frame(obs)
    if f is None:
        return False
    ys, xs = np.where(f > 200)
    return ys.size > 0


# ---------------- AEC -> Gym wrapper (model acts; human is opponent) ----------------

class SingleAgentVsPolicy(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=0, learning_agent="first_0", opponent_policy_fn=None,
                 render_mode=None, frame_skip: int = 1, opponent_fire_idx: int = 1):
        super().__init__()
        self.learning_agent = learning_agent
        self.opponent_name = "second_0" if learning_agent == "first_0" else "first_0"

        self.env = make_pz_env(render_mode=render_mode, frame_skip=frame_skip)
        self.env.reset(seed=seed)

        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)

        # Model side action indices
        meanings = get_action_meanings(self.env, self.learning_agent)
        up_idx = next((i for i,m in enumerate(meanings) if m in ("UP","RIGHT")), 2)
        dn_idx = next((i for i,m in enumerate(meanings) if m in ("DOWN","LEFT")), 3)
        try:
            fire_idx_model = next(i for i,m in enumerate(meanings) if m == "FIRE")
        except StopIteration:
            fire_idx_model = 1

        self.ACTION = {"NOOP": 0, "FIRE": fire_idx_model, "UP": up_idx, "DOWN": dn_idx}

        self.opponent_policy_fn = opponent_policy_fn
        self.opponent_fire_idx = int(opponent_fire_idx)  # FIRE index for human side
        self._last_obs = {a: None for a in self.env.agents}

        # Auto-serve flags – prime both; first active side will send FIRE
        self._need_fire_model = True
        self._need_fire_opp = True

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        self._need_fire_model = True
        self._need_fire_opp = True

        obs_learning = None
        for agent in self.env.agent_iter():
            obs, r, t, tr, info = self.env.last()
            self._last_obs[agent] = obs
            if agent == self.learning_agent:
                obs_learning = obs
                self.env.step(0) if not (t or tr) else self.env.step(None)  # 0=NOOP is valid
                break
            else:
                self.env.step(0) if not (t or tr) else self.env.step(None)

        if obs_learning is None:
            obs_learning = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs_learning, {}

    def step(self, action: int):
        total_r, termd, truncd = 0.0, False, False
        info_acc = {}

        # Phase 1: model acts, then opponent acts
        for agent in self.env.agent_iter():
            obs, r, t, tr, info = self.env.last()
            self._last_obs[agent] = obs
            act_n = self.env.action_space(agent).n  # clip by CURRENT agent's action space

            if agent == self.learning_agent:
                total_r += float(r)
                if t or tr:
                    self.env.step(None)
                    termd |= t; truncd |= tr; info_acc.update(info)
                    break

                # Model action; auto-serve if needed or ball not visible
                act = int(action)
                if self._need_fire_model or not is_ball_visible_obs(obs):
                    act = self.ACTION["FIRE"]
                    self._need_fire_model = False
                self.env.step(int(np.clip(act, 0, act_n - 1)))

            else:
                if t or tr:
                    self.env.step(None)
                    termd |= t; truncd |= tr; info_acc.update(info)
                    continue

                is_left = (agent == "first_0")
                last = self._last_obs.get(agent)

                # Human policy (guarded against None)
                try:
                    opp_action = 0 if self.opponent_policy_fn is None else int(self.opponent_policy_fn(last, is_left))
                except Exception:
                    opp_action = 0

                # Auto-serve for opponent if needed or ball not visible
                if self._need_fire_opp or not is_ball_visible_obs(last):
                    opp_action = self.opponent_fire_idx
                    self._need_fire_opp = False

                self.env.step(int(np.clip(opp_action, 0, act_n - 1)))

            # If any reward happened this turn, prime next rally to auto-serve
            if r != 0:
                self._need_fire_model = True
                self._need_fire_opp = True

            if agent != self.learning_agent:
                break

        # Phase 2: fetch next obs for model (respect AEC: always step once)
        next_obs = None
        for agent in self.env.agent_iter():
            obs, r, t, tr, info = self.env.last()
            self._last_obs[agent] = obs
            if agent == self.learning_agent:
                next_obs = obs if isinstance(obs, np.ndarray) else np.zeros(self.observation_space.shape, dtype=np.uint8)
                total_r += float(r)
                termd |= t; truncd |= tr; info_acc.update(info)
                self.env.step(None if (t or tr) else 0)  # 0=NOOP
                break
            self.env.step(None if (t or tr) else 0)

        if next_obs is None:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.uint8)

        return next_obs, total_r, termd, truncd, info_acc

    def close(self):
        self.env.close()


# ---------------- Human policy: hold-to-glide + debounced FIRE ----------------

def make_human_policy(
    up_idx: int,
    down_idx: int,
    noop_idx: int = 0,
    fire_idx: int = 1,
    *,
    key_state: KeyState | None,
    initial_delay_ms: int = 120,  # delay before repeats start when holding
    hold_repeat_ms: int = 30,     # cadence while holding
    fire_cooldown_ms: int = 250   # debounce FIRE presses
):
    """
    Hold-to-glide movement: on hold, emit immediately, then after `initial_delay_ms`,
    emit again every `hold_repeat_ms`. Space (FIRE) is debounced.
    """
    t0 = {"up": 0.0, "down": 0.0}       # first press time
    last_rep = {"up": 0.0, "down": 0.0} # last repeat time
    last_fire = 0.0
    prev_space = False

    def should_emit(now: float, key: str) -> bool:
        if t0[key] == 0.0:
            t0[key] = now
            last_rep[key] = now
            return True
        if (now - t0[key]) * 1000.0 < initial_delay_ms:
            return False
        if (now - last_rep[key]) * 1000.0 >= hold_repeat_ms:
            last_rep[key] = now
            return True
        return False

    def reset_key(key: str):
        t0[key] = 0.0
        last_rep[key] = 0.0

    def _fn(_last_obs, _is_left: bool) -> int:
        nonlocal last_fire, prev_space
        ks = key_state
        if ks is None:
            return noop_idx

        now = time.monotonic()
        k_up = ks.pressed("up") or ks.pressed("w")
        k_dn = ks.pressed("down") or ks.pressed("s")
        k_sp = ks.pressed("space")

        # FIRE (debounced)
        edge_sp = k_sp and not prev_space
        prev_space = k_sp
        if edge_sp and (now - last_fire) * 1000.0 >= fire_cooldown_ms:
            last_fire = now
            return fire_idx

        # Movement (hold-to-glide)
        if k_up and not k_dn:
            if should_emit(now, "up"):
                return up_idx
        else:
            reset_key("up")

        if k_dn and not k_up:
            if should_emit(now, "down"):
                return down_idx
        else:
            reset_key("down")

        return noop_idx

    return _fn


# ---------------- Play loop ----------------

def play_human_vs_model(model_path: str, seed: int, model_side: str, render_mode: str = "human"):
    opponent_name = "second_0" if model_side == "first_0" else "first_0"

    # Calibrate the HUMAN (opponent) side so Space uses the correct FIRE
    probe = make_pz_env(render_mode=render_mode, frame_skip=1)
    up_h, dn_h, fire_h = calibrate_indices(probe, opponent_name)
    probe.close()

    key_state = KeyState() if PYNPUT_AVAILABLE else None
    if not PYNPUT_AVAILABLE:
        print("Note: install 'pynput' for keyboard input:  python -m pip install pynput")

    human_policy = make_human_policy(
        up_idx=up_h,
        down_idx=dn_h,
        noop_idx=0,
        fire_idx=fire_h,
        key_state=key_state,
        initial_delay_ms=120,  # tweak 100–160 for feel
        hold_repeat_ms=30,     # tweak 25–50 for faster/slower glide
        fire_cooldown_ms=250
    )

    # Model plays as model_side; human controls the other
    env_model = SingleAgentVsPolicy(
        seed=seed,
        learning_agent=model_side,
        opponent_policy_fn=human_policy,
        render_mode=render_mode,
        frame_skip=1,
        opponent_fire_idx=fire_h,
    )

    venv = DummyVecEnv([lambda: env_model])
    venv = VecTransposeImage(venv)
    model = PPO.load(model_path, device="cpu")

    print("Controls: HOLD ↑/↓ or W/S to glide; Space to serve (debounced). Ctrl+C to quit.")
    obs = venv.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        if done[0]:
            obs = venv.reset()
        # Pace loop; increase to 0.008–0.010 for slower feel
        time.sleep(0.006)


def main():
    ap = argparse.ArgumentParser(description="Play human vs model (Pong)")
    ap.add_argument("--model_path", type=str, default="ppo_pong_ma.zip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_side", type=str, default="first_0", choices=["first_0", "second_0"])
    args = ap.parse_args()
    play_human_vs_model(args.model_path, args.seed, args.model_side, render_mode="human")


if __name__ == "__main__":
    main()
