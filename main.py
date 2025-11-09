"""
Self-play Double DQN on PettingZoo Pong (two learners trained simultaneously).

What this script does
- Builds pong_v3 with SuperSuit preprocessing (grayscale→84x84→frame-stack=4→uint8)
- Creates two QNetworks (left/right) + two Learners (Double DQN) + two ReplayBuffers
- Runs self-play; at each step, stores transitions for BOTH players and trains them
- Prints a per-episode summary (who reached 21 first, epsilon, scores)
- Prints a one-line training summary every 10k env steps
- **Saves checkpoints** for both agents periodically and at the end

Usage
    pip install "pettingzoo[atari]" "autorom[accept-rom-license]" supersuit torch imageio
    AutoROM --accept-license
    python main.py --total_steps 500000 --save_every 10000 --save_dir checkpoints

Notes
- Observations from env are HWC (84,84,4) uint8; we convert to CHW (4,84,84) for networks/replay.
- Replay buffers are independent per agent; learners update on their own samples.
- Target nets sync every --target_update steps.
"""
from __future__ import annotations
import argparse
import os
import time
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch

from envs import make_pong_env, hwc_to_chw_uint8
from network import QNetwork
from learner import Learner
from policy import EpsilonScheduler, select_action_eps_greedy
from replay import ReplayBuffer


def set_seed_everywhere(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # make CuDNN deterministic(ish)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_agents(n_actions: int, device: torch.device, lr: float, gamma: float):
    # Left agent
    left_online = QNetwork(n_actions=n_actions, in_channels=4).to(device)
    left_target = QNetwork(n_actions=n_actions, in_channels=4).to(device)
    left_learner = Learner(left_online, left_target, n_actions=n_actions, lr=lr, gamma=gamma, device=str(device))

    # Right agent
    right_online = QNetwork(n_actions=n_actions, in_channels=4).to(device)
    right_target = QNetwork(n_actions=n_actions, in_channels=4).to(device)
    right_learner = Learner(right_online, right_target, n_actions=n_actions, lr=lr, gamma=gamma, device=str(device))

    return left_learner, right_learner


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(save_dir: str, side: str, learner: Learner, global_step: int, episodes: int, epsilon: float):
    ensure_dir(save_dir)
    ckpt = {
        "step": int(global_step),
        "episodes": int(episodes),
        "epsilon": float(epsilon),
        "online_state_dict": learner.q_online.state_dict(),
        "target_state_dict": learner.q_target.state_dict(),
        "optimizer_state_dict": learner.opt.state_dict(),
        "gamma": getattr(learner, "gamma", None),
    }
    path = os.path.join(save_dir, f"{side}_step{global_step}.pt")
    torch.save(ckpt, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Replay & learning
    parser.add_argument("--replay_capacity", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learn_start", type=int, default=50_000)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--target_update", type=int, default=10_000)

    # Optimization + DDQN
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Exploration schedule (shared for both agents)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_final", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=int, default=1_000_000)

    # Rendering
    parser.add_argument("--render_mode", type=str, default="rgb_array", choices=["none", "rgb_array", "human"],
                        help="Rendering choice (rgb_array may be slower; 'none' skips render calls)")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=10_000, help="Save every N env steps; 0 disables periodic saves")
    parser.add_argument("--save_on_episode", type=int, default=1, help="Also save at the end of each episode (0/1)")

    args = parser.parse_args()

    set_seed_everywhere(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    # --- Environment ---
    env = make_pong_env(render_mode=None if args.render_mode == "none" else args.render_mode)
    obs, infos = env.reset(seed=args.seed)
    agents = list(env.agents)
    assert len(agents) == 2, f"Expected 2 agents, got {agents}"
    left_name, right_name = agents[0], agents[1]
    n_actions = int(env.action_space(left_name).n)

    # --- Agents & Replay ---
    left_learner, right_learner = build_agents(n_actions, device, args.lr, args.gamma)
    left_buffer = ReplayBuffer(capacity=args.replay_capacity, obs_shape=(4, 84, 84), rng=rng)
    right_buffer = ReplayBuffer(capacity=args.replay_capacity, obs_shape=(4, 84, 84), rng=rng)

    # Epsilon schedule (shared)
    eps_sched = EpsilonScheduler(args.eps_start, args.eps_final, args.eps_decay)

    # Tracking
    global_step = 0
    episodes = 0
    last_target_sync = 0
    t_start = time.time()

    # Per-episode scoreboards (points to 21)
    left_points = 0
    right_points = 0
    left_return = 0.0
    right_return = 0.0

    # EMA losses for summaries
    ema_decay = 0.99
    left_loss_ema = None
    right_loss_ema = None

    # Episode returns buffer for quick moving stats
    ep_hist_len = 50
    left_ep_returns = deque(maxlen=ep_hist_len)
    right_ep_returns = deque(maxlen=ep_hist_len)

    # Prepare initial CHW views
    obs_chw = {a: hwc_to_chw_uint8(obs[a]) for a in agents}

    while global_step < args.total_steps:
        epsilon = eps_sched.value(global_step)

        # --- Select actions for both players (ε-greedy over online nets) ---
        @torch.no_grad()
        def q_from(net: torch.nn.Module, x_np: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(x_np).to(device)
            q = net(x).detach().cpu().numpy()
            return q

        a_left = select_action_eps_greedy(
            obs_chw[left_name], n_actions, epsilon, rng,
            q_values_fn=lambda x: q_from(left_learner.q_online, x)
        )
        a_right = select_action_eps_greedy(
            obs_chw[right_name], n_actions, epsilon, rng,
            q_values_fn=lambda x: q_from(right_learner.q_online, x)
        )

        actions = {left_name: a_left, right_name: a_right}

        # Keep a copy of current obs for replay before stepping
        prev_obs_chw = {left_name: obs_chw[left_name].copy(), right_name: obs_chw[right_name].copy()}

        # --- Step env ---
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # Update scoreboards and returns (Pong gives +1 to scorer, -1 to the other)
        left_return += float(rewards.get(left_name, 0.0))
        right_return += float(rewards.get(right_name, 0.0))
        if rewards.get(left_name, 0.0) > 0:
            left_points += 1
        if rewards.get(right_name, 0.0) > 0:
            right_points += 1

        # --- Build next states (handle agent removal at episode end) ---
        def get_next_chw(name: str) -> np.ndarray:
            if name in next_obs:
                return hwc_to_chw_uint8(next_obs[name])
            return np.zeros((4, 84, 84), dtype=np.uint8)

        next_obs_chw_left = get_next_chw(left_name)
        next_obs_chw_right = get_next_chw(right_name)
        done_left = bool(terminations.get(left_name, False) or truncations.get(left_name, False))
        done_right = bool(terminations.get(right_name, False) or truncations.get(right_name, False))

        # --- Store transitions ---
        left_buffer.add(prev_obs_chw[left_name], a_left, float(rewards.get(left_name, 0.0)), next_obs_chw_left, done_left)
        right_buffer.add(prev_obs_chw[right_name], a_right, float(rewards.get(right_name, 0.0)), next_obs_chw_right, done_right)

        # --- Learn (both agents) ---
        if global_step >= args.learn_start and global_step % args.train_freq == 0:
            if len(left_buffer) >= args.batch_size:
                batch = left_buffer.sample(args.batch_size)
                l = left_learner.train_step(batch)
                left_loss_ema = l if left_loss_ema is None else (ema_decay * left_loss_ema + (1 - ema_decay) * l)
            if len(right_buffer) >= args.batch_size:
                batch = right_buffer.sample(args.batch_size)
                l = right_learner.train_step(batch)
                right_loss_ema = l if right_loss_ema is None else (ema_decay * right_loss_ema + (1 - ema_decay) * l)

        # --- Periodic target sync ---
        if (global_step - last_target_sync) >= args.target_update and global_step > 0:
            left_learner.target_update()
            right_learner.target_update()
            last_target_sync = global_step

        global_step += 1

        # One-line summary + periodic checkpoint every 10k steps
        if global_step % 10_000 == 0:
            elapsed = time.time() - t_start
            sps = global_step / max(1e-6, elapsed)  # steps per second
            print(
                f"step {global_step:,} | eps {epsilon:.3f} | left_loss {left_loss_ema if left_loss_ema is not None else float('nan'):.4f} "
                f"| right_loss {right_loss_ema if right_loss_ema is not None else float('nan'):.4f} | SPS {sps:.1f} "
                f"| replay L/R {len(left_buffer):,}/{len(right_buffer):,} | last {ep_hist_len} avg return L/R "
                f"{(np.mean(left_ep_returns) if left_ep_returns else float('nan')):.2f}/"
                f"{(np.mean(right_ep_returns) if right_ep_returns else float('nan')):.2f}"
            )
            if args.save_every and (global_step % args.save_every == 0):
                lp = save_checkpoint(args.save_dir, "left", left_learner, global_step, episodes, epsilon)
                rp = save_checkpoint(args.save_dir, "right", right_learner, global_step, episodes, epsilon)
                print(f"[ckpt] saved @ step {global_step:,} → {lp} | {rp}")

        # Episode end? In parallel API, when both are done env.agents becomes empty.
        done_episode = len(env.agents) == 0
        if done_episode:
            episodes += 1
            # Winner by points (should be 21)
            if left_points > right_points:
                winner = left_name
            elif right_points > left_points:
                winner = right_name
            else:
                winner = "tie"

            # Log episode summary
            print(
                f"EP {episodes} | winner: {winner} | score L/R: {left_points}-{right_points} | "
                f"eps {epsilon:.3f} | returns L/R: {left_return:.1f}/{right_return:.1f} | steps {global_step:,}"
            )

            if args.save_on_episode:
                lp = save_checkpoint(args.save_dir, "left", left_learner, global_step, episodes, epsilon)
                rp = save_checkpoint(args.save_dir, "right", right_learner, global_step, episodes, epsilon)
                print(f"[ckpt] episode {episodes} saved → {lp} | {rp}")

            # Track moving averages
            left_ep_returns.append(left_return)
            right_ep_returns.append(right_return)

            # Reset episode counters
            left_points = right_points = 0
            left_return = right_return = 0.0

            # Reset env
            obs, infos = env.reset()
            obs_chw = {a: hwc_to_chw_uint8(obs[a]) for a in env.agents}
            continue

        # Not done: carry forward next obs
        obs = next_obs
        # Env may drop finished agents at the very last step before episode end, but we handle done_episode above.
        obs_chw = {a: hwc_to_chw_uint8(obs[a]) for a in env.agents}

    # Final save
    lp = save_checkpoint(args.save_dir, "left", left_learner, global_step, episodes, eps_sched.value(global_step))
    rp = save_checkpoint(args.save_dir, "right", right_learner, global_step, episodes, eps_sched.value(global_step))
    env.close()
    total_time = time.time() - t_start
    print(f"Finished {global_step:,} steps across {episodes} episodes in {total_time/3600:.2f} h. Final ckpts → {lp} | {rp}")


if __name__ == "__main__":
    main()
