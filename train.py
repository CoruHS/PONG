# pong_right_vs_cpu.py
# Left paddle = built-in Atari CPU (from the ROM). Right paddle = your controller (model or placeholder).

import argparse
import time
import numpy as np
import shimmy
import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from dataclasses import dataclass
from typing import Optional, Tuple

# --- FrameStack compat across Gymnasium versions ---
try:
    from gymnasium.wrappers import FrameStackObservation as FrameStackLike
    def frame_stack(env, n): return FrameStackLike(env, n)            # new API
except Exception:
    from gymnasium.wrappers import FrameStack as FrameStackLike
    def frame_stack(env, n): return FrameStackLike(env, num_stack=n)   # old API

def make_env(render: bool = True, sticky: float = 0.0):
    #Builds ALE/Pong-v5 so that:
    #  - Left paddle = built-in CPU (part of the ROM)
    #  - Right paddle = your actions (what this script supplies)
    #We disable base frameskip and let AtariPreprocessing do DQN-style frame skipping and max-pool.
    env = gym.make(
        "ALE/Pong-v5",
        frameskip=1,                         # disable base skip; wrapper will handle it
        repeat_action_probability=sticky,    # 0.0 = v4-like deterministic; 0.25 ~ v5 default "sticky"
        render_mode="human" if render else "rgb_array",
    )
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,                        # skip=4 + max-pool(last2)
        terminal_on_life_loss=False,         # Pong has no lives; just being explicit
        scale_obs=False,                     # keep uint8; we’ll scale for NN if needed
    )
    env = frame_stack(env, 4)                # stack last 4 frames
    return env

# ---- Optional: use an SB3 model (.zip) to control the right paddle ----
def load_sb3_model(path: str):
    from stable_baselines3 import DQN, PPO, A2C
    # Try common algos; succeed with the right one
    for Algo in (DQN, PPO, A2C):
        try:
            return Algo.load(path, device="cpu")
        except Exception:
            continue
    raise RuntimeError(f"Could not load SB3 model from: {path}")

def obs_to_sb3(obs):
    """Convert stacked LazyFrames -> (1,C,H,W) float32 in [0,1] for SB3 CNN policies."""
    x = np.asarray(obs)
    assert x.ndim == 3, f"Expected HWC, got {x.shape}"
    if x.shape[-1] in (1, 4):
        x = np.transpose(x, (2, 0, 1))
    return (x.astype(np.float32)/255.0)[None]


def self_check(render=False, sticky=0.0, steps=3000, seed=0):
    env = make_env(render=render, sticky=sticky)
    obs, info = env.reset(seed = seed)
    x = np.asarray(obs)
    print("[Step0] obs.shape:", x.shape, "dtype:", x.dtype, "min/max:", x.min(), x.max())
    print("[Step0] action_space:", env.action_space)
    print("[Step0] action_meanings:", env.unwrapped.get_action_meanings())
    uniq_rewards, ep = set(), 0
    for _ in range(steps):
        a = env.action_space.sample()
        obs, r, terminated, truncated, _ = env.step(a)
        uniq_rewards.add(float(r))
        if terminated or truncated:
            ep += 1
            obs, _ = env.reset()
    print("[Step0] unique rewards seen:", sorted(uniq_rewards), "| episodes:", ep)
    env.close()


@dataclass
class TransitionBatch:
    obs: np.ndarray         # (B, C, H, W) uint8
    actions: np.ndarray     # (B,) int32
    rewards: np.ndarray     # (B,) float32
    next_obs: np.ndarray    # (B, C, H, W) uint8
    dones: np.ndarray       # (B,) bool

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, int, int] = (4, 84, 84),
        rng: Optional[np.random.Generator] = None,
    ):
        self.capacity = int(capacity)
        C, H, W = obs_shape
        self.obs = np.empty((self.capacity, C, H, W), dtype=np.uint8)
        self.next_obs = np.empty((self.capacity, C, H, W), dtype=np.uint8)
        self.actions = np.empty((self.capacity,), dtype=np.int32)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.dones = np.empty((self.capacity,), dtype=np.bool_)
        self.idx = 0
        self.full = False
        self.rng = rng or np.random.default_rng()

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int) -> TransitionBatch:
        size = len(self)
        assert size >= batch_size, "Replay underflow: not enough samples yet"
        idxs = self.rng.integers(0, size, size=batch_size)
        return TransitionBatch(
            obs=self.obs[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_obs=self.next_obs[idxs],
            dones=self.dones[idxs],
        )
    

def step2_rollout_fill(buffer_capacity=100_000, steps=50_000, sticky=0.0, seed=0, render=False):
    """
    Populate a ReplayBuffer with random-policy transitions using your env
    (already does frame skip + stack). Transitions respect episode boundaries.
    """
    env = make_env(render=render, sticky=sticky)
    rb = ReplayBuffer(capacity=buffer_capacity)
    obs, _ = env.reset(seed=seed)

    uniq_rewards = set()
    for t in range(steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        uniq_rewards.add(float(reward))

        # Convert (H,W,C) -> (C,H,W) and keep as uint8
        o = np.transpose(np.asarray(obs), (2, 0, 1))
        no = np.transpose(np.asarray(next_obs), (2, 0, 1))
        rb.add(o, int(action), float(reward), no, bool(terminated or truncated))

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    print(f"[Step2] replay size={len(rb)} | unique rewards seen={sorted(uniq_rewards)}")
    return rb

class EpsilonScheduler:
    """Linear anneal from eps_start -> eps_final over decay_steps."""
    def __init__(self, eps_start=1.0, eps_final=0.01, decay_steps=1_000_000):
        self.eps_start = float(eps_start)
        self.eps_final = float(eps_final)
        self.decay_steps = int(decay_steps)

    def value(self, t: int) -> float:
        if t >= self.decay_steps:
            return self.eps_final
        frac = t / self.decay_steps
        return self.eps_start + frac * (self.eps_final - self.eps_start)

def select_action_eps_greedy(obs_uint8_chw: np.ndarray,
                            n_actions: int,
                            epsilon: float,
                            rng: np.random.Generator,
                            q_values_fn=None) -> int:
    """
    - obs_uint8_chw: (4,84,84) uint8 stacked state
    - q_values_fn(x): maps float32 scaled (1,4,84,84) -> (1,n_actions) Q-values
    If None (pre-network), the greedy branch returns action 0 as a placeholder.
    """
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    if q_values_fn is None:
        return 0
    x = obs_uint8_chw.astype(np.float32, copy=False)[None] / 255.0
    q = q_values_fn(x)  # expected (1, n_actions)
    return int(np.argmax(q, axis=1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_zip", type=str, default="",
                    help="Path to an SB3 .zip to control the RIGHT paddle. If empty, uses a placeholder.")
    ap.add_argument("--policy", choices=["random", "noop"], default="random",
                    help="Placeholder policy when no model_zip is provided.")
    ap.add_argument("--sticky", type=float, default=0.0,
                    help="Sticky action probability (0.0 = deterministic, 0.25 ~ v5 default).")
    ap.add_argument("--steps", type=int, default=20000)
    args = ap.parse_args()

    env = make_env(render=True, sticky=args.sticky)
    obs, info = env.reset(seed=0)

    # Right-side controller (your agent)
    model = load_sb3_model(args.model_zip) if args.model_zip else None

    def select_action(o):
        if model is not None:
            a, _ = model.predict(obs_to_sb3(o), deterministic=True)
            return int(np.asarray(a).item())
        return 0 if args.policy == "noop" else env.action_space.sample()

    print("LEFT = Atari ROM CPU | RIGHT = your controller (model or placeholder).")
    print("Obs shape:", np.asarray(obs).shape, "| Action space:", env.action_space)

    steps, ep_return = 0, 0.0
    try:
        while steps < args.steps:
            action = select_action(obs)              # this controls the RIGHT paddle
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            steps += 1
            time.sleep(0.001)                        # small throttle so it’s watchable
            if terminated or truncated:
                print(f"Episode done. return={ep_return:.1f}")
                ep_return = 0.0
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()
    
