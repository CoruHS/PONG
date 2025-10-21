#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pong_dqn_v4_autoserve.py

Multi-agent Pong (PettingZoo + SuperSuit) with a simple, memory-safe DQN agent
("me") versus a fixed opponent powered by a pre-trained Stable-Baselines3 PPO
policy.

v4 highlights (fixes the "stutter / never-bounces" failure mode):
- **Auto-serve**: the environment requires `FIRE` to launch the ball after resets
  and after every scored point. The DQN action head stays on a small set
  [NOOP, UP, DOWN]; we trigger `FIRE` outside learning for one step when needed.
- **Eval/demo do not advance training frame counters**: `act(..., explore=False)`
  avoids bumping `frame_num` and lets us set an evaluation epsilon directly.
- Everything else stays close to Nature-DQN (Double DQN + target net, Huber loss).

Python 3.11 compatible.
"""

from __future__ import annotations
import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# PettingZoo Atari Pong (two-player)
from pettingzoo.atari import pong_v3
import supersuit as ss

# For loading the SB3 PPO opponent
from stable_baselines3 import PPO


# -----------------------------
# Utility: helpers & seeding
# -----------------------------

def pz_reset(env, seed=None):
    res = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(res, tuple) and len(res) == 2:
        return res[0]
    return res


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Environment factory (Parallel API)
# -----------------------------

def make_wrapped_pong_parallel(
    render: bool = False,
    seed: int = 0,
    frameskip: int = 4,
    sticky: float = 0.0,
):
    """Create two-player Pong with Atari-style preprocessing via SuperSuit.

    Order (DeepMind-style):
      1) frame_skip_v0
      2) max_observation_v0 over last 2 frames
      3) resize to 84x84
      4) dtype uint8
      5) frame_stack_v1 to (H,W,4) channels-last
      6) reward clipping to [-1, 1]
      7) optional sticky_actions_v0

    Observation per agent: (84, 84, 4) uint8
    """
    env = pong_v3.parallel_env(
        obs_type="grayscale_image",
        full_action_space=False,  # minimal action set: 0:NOOP,1:FIRE,2:UP,3:DOWN,4:UPFIRE,5:DOWNFIRE
        render_mode="rgb_array" if render else None,
    )
    env.reset(seed=seed)

    # Wrapper order
    env = ss.frame_skip_v0(env, frameskip)
    env = ss.max_observation_v0(env, 2)
    env = ss.resize_v1(env, 84, 84)
    env = ss.dtype_v0(env, np.uint8)
    env = ss.frame_stack_v1(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    if sticky and sticky > 0.0:
        env = ss.sticky_actions_v0(env, repeat_action_probability=float(sticky))

    return env


# -----------------------------
# DQN network (Nature-CNN style)
# -----------------------------

class NatureCNN(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        # 84x84 -> 7x7 feature map with 64 channels
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0  # uint8 -> float in [0,1]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        q = self.head(x)
        return q


# -----------------------------
# Memory-safe replay buffer storing single frames (uint8)
# -----------------------------

class ReplayBufferFrames:
    def __init__(
        self,
        capacity_frames: int = 300_000,
        h: int = 84,
        w: int = 84,
        stack: int = 4,
        use_memmap: bool = False,
        directory: str = "./replay",
    ):
        self.capacity = int(capacity_frames)
        self.h = h
        self.w = w
        self.stack = stack
        self.use_memmap = use_memmap
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

        def arr_uint8(shape):
            if use_memmap:
                return np.memmap(os.path.join(directory, "frames.uint8"), mode="w+", dtype=np.uint8, shape=shape)
            return np.empty(shape, dtype=np.uint8)

        self.frames = arr_uint8((self.capacity, h, w))
        self.actions = np.empty(self.capacity, dtype=np.uint8)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=np.bool_)
        self.idx = 0
        self.full = False

    def size(self) -> int:
        return self.capacity if self.full else self.idx

    def add(self, frame_uint8: np.ndarray, action: int, reward: float, done: bool):
        self.frames[self.idx, ...] = frame_uint8  # (84,84) uint8
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or (self.idx == 0)

    def _valid_indices(self, batch_size: int) -> np.ndarray:
        max_idx = self.capacity if self.full else self.idx
        assert max_idx > self.stack + 1, "Not enough frames collected to sample."
        out = []
        while len(out) < batch_size:
            i = np.random.randint(self.stack, max_idx - 1)
            # no done inside the interior of s (exclude the last frame i-1)
            if self.dones[i - self.stack: i - 1].any():
                continue
            # if current transition is non-terminal, avoid stitching s2 across a terminal
            if not self.dones[i - 1] and self.dones[i - (self.stack - 1): i + 1].any():
                continue
            out.append(i)
        return np.array(out, dtype=np.int64)

    def sample(self, batch_size: int, device: torch.device):
        idx_next = self._valid_indices(batch_size)
        idx_cur = idx_next - 1

        def stack_for(end_idx):
            idxs = np.stack([end_idx - j for j in range(self.stack - 1, -1, -1)], axis=1)
            stacked = self.frames[idxs, ...]  # (B,4,H,W)
            return stacked

        s = stack_for(idx_cur)
        s2 = stack_for(idx_next)
        a = self.actions[idx_cur]
        r = self.rewards[idx_cur]
        d = self.dones[idx_cur]

        s = torch.as_tensor(s, device=device, dtype=torch.uint8)
        s2 = torch.as_tensor(s2, device=device, dtype=torch.uint8)
        a = torch.as_tensor(a, device=device, dtype=torch.long)
        r = torch.as_tensor(r, device=device, dtype=torch.float32)
        d = torch.as_tensor(d, device=device, dtype=torch.float32)
        return s, a, r, s2, d


# -----------------------------
# Opponent policy using SB3 PPO model
# -----------------------------

class SB3PPOOpponent:
    """Wrap a loaded SB3 PPO policy to act inside our PettingZoo env.

    Assumes observations are (84,84,4) uint8, channels-last from our wrappers.
    Auto-detects policy's expected layout and transposes to match.
    Optionally adds epsilon-greedy randomness on top of PPO's choice.
    """
    def __init__(self, sb3_path: str, device: str = "cpu", stochastic: bool = False, rand_eps: float = 0.0):
        import warnings as _warnings
        assert os.path.exists(sb3_path), f"SB3 model not found at {sb3_path}"
        _warnings.filterwarnings("ignore", message="Could not deserialize object learning_rate")
        _warnings.filterwarnings("ignore", message="Could not deserialize object lr_schedule")
        _warnings.filterwarnings("ignore", message="Could not deserialize object clip_range")
        _warnings.filterwarnings("ignore", message="You loaded a model that was trained using OpenAI Gym")
        _warnings.filterwarnings("ignore", message="You are probably loading a A2C/PPO model saved with SB3 < 1.7.0")
        custom = {
            "learning_rate": 2.5e-4,
            "lr_schedule": lambda _: 2.5e-4,
            "clip_range": 0.1,
        }
        try:
            self.model: PPO = PPO.load(sb3_path, device=device, custom_objects=custom, print_system_info=False)
        except TypeError:
            self.model: PPO = PPO.load(sb3_path, device=device, custom_objects=custom)
        self.device = device
        self.n_actions = int(self.model.policy.action_space.n)
        self.stochastic = bool(stochastic)
        self.rand_eps = float(rand_eps)
        # Detect obs layout expected by the policy
        self.obs_shape = tuple(getattr(self.model, "observation_space", getattr(self.model.policy, "observation_space", None)).shape)
        self.expect_chw = (len(self.obs_shape) == 3 and self.obs_shape[0] in (1, 3, 4))

    def act(self, obs_hw4_uint8: np.ndarray) -> int:
        if self.expect_chw:
            x = obs_hw4_uint8.transpose(2, 0, 1)[None, ...].astype(np.float32)  # (1,4,84,84)
        else:
            x = obs_hw4_uint8[None, ...].astype(np.float32)                      # (1,84,84,4)
        action, _ = self.model.predict(x, deterministic=not self.stochastic)
        a = int(action if np.isscalar(action) else action.item())
        # Optional epsilon-greedy noise toward UP/DOWN to ease early play
        if self.rand_eps > 0.0 and np.random.rand() < self.rand_eps:
            a = int(np.random.choice([2, 3]))  # 2:UP, 3:DOWN in minimal set
        return a


# -----------------------------
# DQN Agent
# -----------------------------

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 32
    target_update_freq: int = 5000
    learn_freq: int = 4
    warmup_frames: int = 20_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_frames: int = 300_000
    replay_capacity_frames: int = 300_000
    stack: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_memmap: bool = False
    replay_dir: str = "./replay"


class DQNAgent:
    def __init__(self, n_actions: int, cfg: DQNConfig):
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.q = NatureCNN(cfg.stack, n_actions).to(self.device)
        self.q_target = NatureCNN(cfg.stack, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = ReplayBufferFrames(capacity_frames=cfg.replay_capacity_frames, stack=cfg.stack, use_memmap=cfg.use_memmap, directory=cfg.replay_dir)
        self.frame_num = 0

    def epsilon(self) -> float:
        # linear schedule with warmup
        d = max(0, min(1, (self.cfg.epsilon_decay_frames - max(0, self.frame_num - self.cfg.warmup_frames)) / self.cfg.epsilon_decay_frames))
        return self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * d

    def act(self, obs_hw4_uint8: np.ndarray, explore: bool = True, epsilon_override: Optional[float] = None) -> int:
        if explore:
            self.frame_num += 1
        x = torch.as_tensor(obs_hw4_uint8.transpose(2, 0, 1), device=self.device, dtype=torch.uint8).unsqueeze(0)
        eps = self.epsilon() if (epsilon_override is None) else float(epsilon_override)
        if explore and np.random.rand() < eps:
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            q = self.q(x)
            a = int(q.argmax(dim=1).item())
        return a

    def remember(self, obs_hw4_uint8: np.ndarray, action: int, reward: float, done: bool):
        # Store only the last frame of the stack (LazyFrames-like)
        last_frame = obs_hw4_uint8[:, :, -1]
        self.replay.add(last_frame, action, reward, done)

    def learn_step(self) -> Optional[Dict[str, float]]:
        if self.frame_num < self.cfg.warmup_frames:
            return None
        if self.frame_num % self.cfg.learn_freq != 0:
            return None
        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size, self.device)
        q = self.q(s)
        q_sa = q.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            # Double DQN target
            a_max = self.q(s2).argmax(dim=1)
            q_tgt_next = self.q_target(s2).gather(1, a_max[:, None]).squeeze(1)
            y = r + (1.0 - d) * self.cfg.gamma * q_tgt_next
        td = y - q_sa
        loss = F.smooth_l1_loss(q_sa, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(self.q.parameters(), 10.0))
        self.optimizer.step()
        if self.frame_num % self.cfg.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return {
            "loss": float(loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "q_std": float(q_sa.std().item()),
            "tgt_mean": float(y.mean().item()),
            "td_abs_mean": float(td.abs().mean().item()),
            "grad_norm": grad_norm,
        }

    def save(self, path: str):
        torch.save({
            'q': self.q.state_dict(),
            'cfg': self.cfg.__dict__,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt['q'])
        self.q_target.load_state_dict(self.q.state_dict())
        if 'cfg' in ckpt:
            for k, v in ckpt['cfg'].items():
                setattr(self.cfg, k, v)


# -----------------------------
# Pretty logging helpers
# -----------------------------

def _fmt_hist(counts: np.ndarray) -> str:
    total = counts.sum()
    if total <= 0:
        return "{}"
    probs = counts / total
    return "{" + ", ".join(f"{i}:{counts[i]}/{probs[i]:.2f}" for i in range(len(counts))) + "}"


# -----------------------------
# Training loop (with auto-serve)
# -----------------------------

def train(
    sb3_path: str,
    save_path: str,
    train_steps: int = 300_000,
    seed: int = 0,
    log_interval: int = 10_000,
    frameskip: int = 4,
    sticky: float = 0.0,
    max_points: int = 21,
    max_ep_steps: int = 30_000,
    replay_capacity: Optional[int] = None,
    memmap: bool = False,
    replay_dir: str = "./replay",
    opp_stochastic: bool = False,
    opp_rand_eps: float = 0.0,
):
    set_seed(seed)
    env = make_wrapped_pong_parallel(render=False, seed=seed, frameskip=frameskip, sticky=sticky)

    agents = env.possible_agents
    assert len(agents) == 2, f"Expected 2 agents, got {len(agents)}: {agents}"
    me, opp = agents[0], agents[1]

    # Minimal action mapping for *our* agent: 0=NOOP, 2=UP, 3=DOWN
    ME_ACTIONS = np.array([0, 2, 3], dtype=np.int64)

    n_actions_env = env.action_space(me).n
    cfg = DQNConfig()
    if replay_capacity is not None:
        cfg.replay_capacity_frames = int(replay_capacity)
    cfg.use_memmap = bool(memmap)
    cfg.replay_dir = replay_dir

    agent = DQNAgent(n_actions=len(ME_ACTIONS), cfg=cfg)
    opponent = SB3PPOOpponent(sb3_path, device=agent.cfg.device, stochastic=opp_stochastic, rand_eps=opp_rand_eps)

    obs = pz_reset(env, seed=seed)
    assert obs[me].shape == (84, 84, 4), f"Unexpected obs shape: {obs[me].shape}"

    # Episode bookkeeping
    episode_reward = {me: 0.0, opp: 0.0}
    points_me = 0
    points_opp = 0
    ep = 0
    last_log = 0
    ep_steps = 0

    # Action histograms (index in ME_ACTIONS for us; raw env actions for opp)
    hist_me = np.zeros(len(ME_ACTIONS), dtype=np.int64)
    hist_opp = np.zeros(n_actions_env, dtype=np.int64)

    print(f"[INIT] device={agent.cfg.device} frameskip={frameskip} sticky={sticky} replay_capacity={agent.cfg.replay_capacity_frames} memmap={agent.cfg.use_memmap}")

    reward_events_window = 0
    steps_in_window = 0

    # Auto-serve flag: press FIRE for one step after resets or points
    needs_fire = True

    while agent.frame_num < train_steps:
        # Serve assist: fire the ball before the agent acts
        if needs_fire:
            # 1 == FIRE in Atari minimal action set; apply to both to be safe
            obs, rewards, terms, truncs, infos = env.step({me: 1, opp: 1})
            needs_fire = False
            # Do not store/learn from this forced step
            continue

        o0 = obs[me]
        o1 = obs[opp]

        a0_idx = agent.act(o0, explore=True)   # 0..2
        a0 = int(ME_ACTIONS[a0_idx])           # map to env action {0,2,3}
        a1 = opponent.act(o1)

        hist_me[a0_idx] += 1
        hist_opp[a1] += 1
        actions = {me: a0, opp: a1}

        next_obs, rewards, terms, truncs, infos = env.step(actions)

        # Raw reward accumulation
        r_me = float(rewards[me])
        episode_reward[me] += r_me
        episode_reward[opp] += float(rewards[opp])

        # Points per side
        if r_me > 0:
            points_me += 1
        elif r_me < 0:
            points_opp += 1

        # Learning termination: terminal-on-score
        done_for_learn = (abs(r_me) > 0.0)
        if done_for_learn:
            reward_events_window += 1
            # Next iteration will auto-serve to re-launch the ball
            needs_fire = True

        # Environment termination conditions
        ep_steps += 1
        manual_done = (points_me >= max_points) or (points_opp >= max_points)
        env_done = any(terms.values()) or any(truncs.values()) or manual_done or (ep_steps >= max_ep_steps)

        # Store transition (with learning terminal)
        agent.remember(o0, a0_idx, r_me, done_for_learn or env_done)
        stats = agent.learn_step()

        if env_done:
            ep += 1
            print(
                f"Episode {ep} | frames {agent.frame_num} | rawR(me)={episode_reward[me]:.1f} rawR(opp)={episode_reward[opp]:.1f} | "
                f"points(me)={points_me} points(opp)={points_opp} | steps={ep_steps} | eps={agent.epsilon():.3f}"
            )
            episode_reward = {me: 0.0, opp: 0.0}
            points_me = 0
            points_opp = 0
            ep_steps = 0
            obs = pz_reset(env)
            needs_fire = True
        else:
            obs = next_obs

        # Periodic logs
        steps_in_window += 1
        if agent.frame_num - last_log >= log_interval:
            last_log = agent.frame_num
            rep_size = agent.replay.size()
            qlog = "N/A" if stats is None else (
                f"loss={stats['loss']:.5f} qμ={stats['q_mean']:.3f} qσ={stats['q_std']:.3f} tgtμ={stats['tgt_mean']:.3f} | |δ|μ={stats['td_abs_mean']:.3f} grad={stats['grad_norm']:.2f}"
            )
            reward_rate = reward_events_window / max(1, steps_in_window)
            print(
                f"[LOG] step={agent.frame_num} eps={agent.epsilon():.3f} replay={rep_size}/{agent.cfg.replay_capacity_frames} | "
                f"actions(me idx)={_fmt_hist(hist_me)} actions(opp)={_fmt_hist(hist_opp)} | reward_event_rate={reward_rate:.4f} | {qlog}"
            )
            # reset histograms and window counters
            hist_me[:] = 0
            hist_opp[:] = 0
            reward_events_window = 0
            steps_in_window = 0

    agent.save(save_path)
    print(f"Saved DQN to {save_path}")


# -----------------------------
# Evaluation (no render) with auto-serve and no frame bump
# -----------------------------

def evaluate(save_path: str, sb3_path: str, episodes: int = 10, seed: int = 123, frameskip: int = 4, sticky: float = 0.0, max_points: int = 21, max_ep_steps: int = 30_000, opp_stochastic: bool = False, opp_rand_eps: float = 0.0):
    set_seed(seed)
    env = make_wrapped_pong_parallel(render=False, seed=seed, frameskip=frameskip, sticky=sticky)

    agents = env.possible_agents
    assert len(agents) == 2
    me, opp = agents[0], agents[1]

    # Minimal action mapping for *our* agent
    ME_ACTIONS = np.array([0, 2, 3], dtype=np.int64)

    agent = DQNAgent(n_actions=len(ME_ACTIONS), cfg=DQNConfig())
    agent.load(save_path)

    opponent = SB3PPOOpponent(sb3_path, device=agent.cfg.device, stochastic=opp_stochastic, rand_eps=opp_rand_eps)

    scores = []
    for ep in range(episodes):
        obs = pz_reset(env)
        needs_fire = True
        ep_rew = {me: 0.0, opp: 0.0}
        points_me = 0
        points_opp = 0
        steps = 0
        while True:
            if needs_fire:
                obs, rewards, terms, truncs, infos = env.step({me: 1, opp: 1})
                needs_fire = False
                continue

            o0 = obs[me]
            o1 = obs[opp]
            a0_idx = agent.act(o0, explore=False, epsilon_override=0.01)
            a0 = int(ME_ACTIONS[a0_idx])
            a1 = opponent.act(o1)
            obs, rewards, terms, truncs, infos = env.step({me: a0, opp: a1})
            ep_rew[me] += rewards[me]
            ep_rew[opp] += rewards[opp]
            if rewards[me] > 0:
                points_me += 1
                needs_fire = True
            elif rewards[me] < 0:
                points_opp += 1
                needs_fire = True
            steps += 1
            manual_done = (points_me >= max_points) or (points_opp >= max_points)
            if any(terms.values()) or any(truncs.values()) or manual_done or steps >= max_ep_steps:
                break
        print(f"Eval ep {ep+1}: rawR(me)={ep_rew[me]:.1f}, rawR(opp)={ep_rew[opp]:.1f} | points(me)={points_me} points(opp)={points_opp} steps={steps}")
        scores.append({"raw_me": ep_rew[me], "raw_opp": ep_rew[opp], "points_me": points_me, "points_opp": points_opp, "steps": steps})

    return scores


# -----------------------------
# Post-training exhibition: DQN vs PPO opponent (with auto-serve)
# -----------------------------

def play_vs_opponent(
    save_path: str,
    sb3_path: str,
    episodes: int = 5,
    render: bool = False,
    fps: int = 30,
    seed: int = 321,
    frameskip: int = 4,
    sticky: float = 0.0,
    max_points: int = 21,
    max_ep_steps: int = 30_000,
    opp_stochastic: bool = False,
    opp_rand_eps: float = 0.0,
):
    set_seed(seed)
    env = make_wrapped_pong_parallel(render=render, seed=seed, frameskip=frameskip, sticky=sticky)

    agents = env.possible_agents
    assert len(agents) == 2
    me, opp = agents[0], agents[1]

    # Minimal action mapping for *our* agent
    ME_ACTIONS = np.array([0, 2, 3], dtype=np.int64)

    agent = DQNAgent(n_actions=len(ME_ACTIONS), cfg=DQNConfig())
    agent.load(save_path)

    opponent = SB3PPOOpponent(sb3_path, device=agent.cfg.device, stochastic=opp_stochastic, rand_eps=opp_rand_eps)

    for ep in range(episodes):
        obs = pz_reset(env)
        needs_fire = True
        ep_rew = {me: 0.0, opp: 0.0}
        points_me = 0
        points_opp = 0
        t0 = time.time()
        steps = 0
        while True:
            if needs_fire:
                obs, rewards, terms, truncs, infos = env.step({me: 1, opp: 1})
                needs_fire = False
                continue

            o0 = obs[me]
            o1 = obs[opp]
            a0_idx = agent.act(o0, explore=False, epsilon_override=0.0)  # greedy demo
            a0 = int(ME_ACTIONS[a0_idx])
            a1 = opponent.act(o1)
            obs, rewards, terms, truncs, infos = env.step({me: a0, opp: a1})
            ep_rew[me] += rewards[me]
            ep_rew[opp] += rewards[opp]
            if rewards[me] > 0:
                points_me += 1
                needs_fire = True
            elif rewards[me] < 0:
                points_opp += 1
                needs_fire = True
            steps += 1

            if render:
                _ = env.render()  # returns rgb array at wrapper's resolution
                if fps > 0:
                    time.sleep(1.0 / fps)

            manual_done = (points_me >= max_points) or (points_opp >= max_points)
            if any(terms.values()) or any(truncs.values()) or manual_done or steps >= max_ep_steps:
                dur = time.time() - t0
                print(
                    f"Demo ep {ep+1}: steps={steps} duration={dur:.1f}s | rawR(me)={ep_rew[me]:.1f}, rawR(opp)={ep_rew[opp]:.1f} | points(me)={points_me} points(opp)={points_opp}"
                )
                break


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-agent Pong: DQN vs SB3 PPO opponent (v4 with auto-serve)")
    parser.add_argument("--sb3_path", type=str, required=True, help="Path to SB3 PPO zip (opponent)")
    parser.add_argument("--save_path", type=str, default="./dqn_pong.pt", help="Path to save/load DQN weights")
    parser.add_argument("--train_steps", type=int, default=300_000, help="Training frames for DQN (me)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true", help="Only evaluate the saved model")
    parser.add_argument("--post_episodes", type=int, default=5, help="Episodes to run after training vs PPO")
    parser.add_argument("--render_demo", action="store_true", help="Render the post-training exhibition")
    parser.add_argument("--frameskip", type=int, default=4, help="Action repeat (frameskip). 4 is standard for Atari")
    parser.add_argument("--sticky", type=float, default=0.0, help="Sticky action probability (0.0 to 1.0)")
    parser.add_argument("--max_points", type=int, default=21, help="End an episode when either player reaches this score")
    parser.add_argument("--max_ep_steps", type=int, default=30000, help="Safety cutoff for endless episodes")
    parser.add_argument("--replay_capacity", type=int, default=None, help="Replay capacity in frames (overrides default)")
    parser.add_argument("--memmap", action="store_true", help="Use memory-mapped replay on disk")
    parser.add_argument("--replay_dir", type=str, default="./replay", help="Directory for memmap replay")
    parser.add_argument("--opp_stochastic", action="store_true", help="Make PPO opponent non-deterministic (sampled actions)")
    parser.add_argument("--opp_rand_eps", type=float, default=0.0, help="With this probability, override PPO with random UP/DOWN")
    args = parser.parse_args()

    if args.eval:
        evaluate(
            args.save_path,
            args.sb3_path,
            frameskip=args.frameskip,
            sticky=args.sticky,
            max_points=args.max_points,
            max_ep_steps=args.max_ep_steps,
            opp_stochastic=args.opp_stochastic,
            opp_rand_eps=args.opp_rand_eps,
        )
    else:
        train(
            args.sb3_path,
            args.save_path,
            train_steps=args.train_steps,
            seed=args.seed,
            frameskip=args.frameskip,
            sticky=args.sticky,
            max_points=args.max_points,
            max_ep_steps=args.max_ep_steps,
            replay_capacity=args.replay_capacity,
            memmap=args.memmap,
            replay_dir=args.replay_dir,
            opp_stochastic=args.opp_stochastic,
            opp_rand_eps=args.opp_rand_eps,
        )
        play_vs_opponent(
            args.save_path,
            args.sb3_path,
            episodes=args.post_episodes,
            render=args.render_demo,
            frameskip=args.frameskip,
            sticky=args.sticky,
            max_points=args.max_points,
            max_ep_steps=args.max_ep_steps,
            opp_stochastic=args.opp_stochastic,
            opp_rand_eps=args.opp_rand_eps,
        )


if __name__ == "__main__":
    main()
