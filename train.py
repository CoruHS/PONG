# train.py
import os, time, math, random
import numpy as np
import torch
from tqdm import tqdm

# >>> Make this consistent with your wrapper file name:
from train_env import PongLeftGym  # was: from envs import PongLeftGym

from config import DQNConfig
from utils import FrameStacker, set_seed, linear_epsilon
from replay_buffer import ReplayBuffer
from agent import DQNAgent


# >>> Pretty epsilon schedule preview
def print_epsilon_schedule(cfg):
    milestones = [
        ("start", 0),
        ("25%", int(0.25 * cfg.eps_decay_steps)),
        ("50%", int(0.50 * cfg.eps_decay_steps)),
        ("75%", int(0.75 * cfg.eps_decay_steps)),
        ("eps_decay_steps", cfg.eps_decay_steps),
        ("total_steps", cfg.total_env_steps),
    ]
    print("\nε-greedy schedule:")
    print("step".ljust(14), "epsilon")
    for label, s in milestones:
        e = linear_epsilon(s, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)
        print(f"{label:<14} {e:.4f}")
    print("")  # spacer


def main():
    cfg = DQNConfig()
    set_seed(cfg.seed)

    # >>> Preview the epsilon decay once at startup
    print_epsilon_schedule(cfg)

    env = PongLeftGym(render_mode=None, right_noise=cfg.right_noise)
    obs, _ = env.reset(seed=cfg.seed)
    H, W, _ = obs.shape
    stacker = FrameStacker(cfg.stack_size, cfg.frame_size, cfg.frame_size)
    state = stacker.reset(obs)

    n_actions = env.action_space.n
    agent = DQNAgent(obs_shape=state.shape, n_actions=n_actions, cfg=cfg)

    rb = ReplayBuffer(cfg.replay_capacity, obs_shape=state.shape, dtype=np.uint8)

    # bootstrap replay with random policy
    ep_return, ep_len = 0.0, 0
    left_score, right_score = 0, 0
    print("Filling replay...")
    pbar = tqdm(total=cfg.min_replay_size)
    while len(rb) < cfg.min_replay_size:
        action = np.random.randint(n_actions)
        next_obs, rew, done, _, _ = env.step(action)
        next_state = stacker.step(next_obs)
        rb.add(state, action, rew, next_state, done)
        state = next_state
        ep_return += rew
        ep_len += 1

        if rew > 0:
            left_score += int(rew)
        elif rew < 0:
            right_score += int(-rew)

        if done:
            print(f"[warmup] Episode finished: score {left_score}-{right_score}, return={ep_return:.1f}, len={ep_len}")
            obs, _ = env.reset()
            state = stacker.reset(obs)
            ep_return, ep_len = 0.0, 0
            left_score, right_score = 0, 0
        pbar.update(1)
    pbar.close()

    # Training
    log_t0 = time.time()
    last_eval = 0
    last_ckpt = 0
    episode = 0
    obs, _ = env.reset()
    state = stacker.reset(obs)
    ep_return, ep_len = 0.0, 0
    left_score, right_score = 0, 0

    for step in tqdm(range(cfg.total_env_steps), desc="Training"):
        
        epsilon = linear_epsilon(step, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)
        a = agent.act(state, epsilon)
        if a is None:
            a = np.random.randint(n_actions)

        next_obs, rew, done, _, _ = env.step(a)
        next_state = stacker.step(next_obs)

        rb.add(state, a, rew, next_state, done)
        state = next_state
        ep_return += rew
        ep_len += 1

        if rew > 0:
            left_score += int(rew)
        elif rew < 0:
            right_score += int(-rew)

        # optimization
        if step % cfg.train_freq == 0:
            batch = rb.sample(cfg.batch_size)
            loss = agent.optimize(batch)

        # end of episode
        if done:
            episode += 1
            # >>> Include the epsilon at episode end
            print(f"Episode {episode}: score {left_score}-{right_score} "
                  f"(agent-left vs heuristic-right), return={ep_return:.1f}, len={ep_len}, eps={epsilon:.3f}")
            obs, _ = env.reset()
            state = stacker.reset(obs)
            ep_return, ep_len = 0.0, 0
            left_score, right_score = 0, 0

        # periodic logging
        if (step+1) % cfg.log_interval == 0:
            dt = time.time() - log_t0
            sps = (step+1)/dt
            print(f"[{step+1:,}] eps={epsilon:.3f} replay={len(rb):,} SPS={sps:.1f}")

        # >>> Optional: milestone epsilon breadcrumbs (extra clarity)
        for mark in (int(0.25*cfg.eps_decay_steps), int(0.5*cfg.eps_decay_steps),
                     int(0.75*cfg.eps_decay_steps), cfg.eps_decay_steps):
            if step+1 == mark:
                e_mark = linear_epsilon(step+1, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)
                print(f"[eps milestone] step={step+1:,} -> eps={e_mark:.4f}")

        # periodic eval
        if (step+1) - last_eval >= cfg.eval_interval:
            avg_r = evaluate_policy(agent, episodes=5, render=False, right_noise=cfg.right_noise,
                                    frame_size=cfg.frame_size, stack_size=cfg.stack_size, seed=cfg.seed+7)
            print(f"Eval@{step+1:,}: avg_return={avg_r:.2f}")
            last_eval = step+1

        # checkpoint
        if (step+1) - last_ckpt >= cfg.checkpoint_interval:
            os.makedirs("checkpoints", exist_ok=True)
            agent.save(f"checkpoints/dqn_{step+1:07d}.pt")
            last_ckpt = step+1

    env.close()


def evaluate_policy(agent, episodes: int, render: bool, right_noise: float, frame_size: int, stack_size: int, seed: int):
    import numpy as np
    from train_env import PongLeftGym  # keep consistent
    from utils import FrameStacker
    import torch

    env = PongLeftGym(render_mode=("rgb_array" if render else None), right_noise=right_noise)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed+ep)
        stacker = FrameStacker(stack_size, frame_size, frame_size)
        state = stacker.reset(obs)
        done = False
        ret = 0.0
        left_score, right_score = 0, 0
        while not done:
            with torch.no_grad():
                q = agent.policy(torch.as_tensor(state, device=agent.device).unsqueeze(0))
                a = int(torch.argmax(q, dim=1).item())
            obs, r, done, _, _ = env.step(a)
            state = stacker.step(obs)
            ret += r
            if r > 0:
                left_score += int(r)
            elif r < 0:
                right_score += int(-r)
        print(f"[eval] Episode {ep+1}: score {left_score}-{right_score}, return={ret:.1f}")
        returns.append(ret)
    env.close()
    return float(np.mean(returns))


if __name__ == "__main__":
    main()
