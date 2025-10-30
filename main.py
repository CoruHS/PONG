import os
import numpy as np
import gymnasium as gym
import torch

from train import make_env, ReplayBuffer, EpsilonScheduler, select_action_eps_greedy
from network import QNetwork
from learner import Learner


def train(
    total_steps: int = 1_000_000,
    buffer_size: int = 100_000,
    batch_size: int = 32,
    learning_starts: int = 50_000,
    train_freq: int = 4,
    target_interval: int = 10_000,
    sticky: float = 0.0,
    save_dir: str = "models",
    seed: int = 0,
    device: str = "cuda",
):
    os.makedirs(save_dir, exist_ok=True)

    # --- Env & RNG ---
    env = make_env(render=False, sticky=sticky)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    # --- Replay & epsilon ---
    rb = ReplayBuffer(buffer_size)
    eps = EpsilonScheduler(1.0, 0.01, decay_steps=int(0.1 * total_steps))

    # --- Networks & learner ---
    n_actions = env.action_space.n
    q_online = QNetwork(n_actions=n_actions)
    q_target = QNetwork(n_actions=n_actions)
    L = Learner(q_online, q_target, n_actions, device=device)

    # Optional but tidy: explicit modes (no BN/Dropout here, just hygiene)
    q_online.train()
    q_target.eval()

    # Helper: compute Q-values → NumPy without building a graph  [FIX: no_grad]
    def q_values_fn_numpy(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x_np).to(device)
            return q_online(x_t).detach().cpu().numpy()

    steps = 0
    last_target_sync = 0
    ep_return = 0.0

    while steps < total_steps:
        # HWC (84,84,4) -> CHW (4,84,84) uint8
        o_chw = np.transpose(np.asarray(obs), (2, 0, 1))

        # ε-greedy action (uses no-grad Q call)  [FIX: pass safe fn]
        a = select_action_eps_greedy(
            o_chw, n_actions, eps.value(steps), rng, q_values_fn=q_values_fn_numpy
        )

        # Step environment
        nobs, r, term, trunc, _ = env.step(int(a))

        # Store transition (again HWC->CHW)
        no_chw = np.transpose(np.asarray(nobs), (2, 0, 1))
        rb.add(o_chw, int(a), float(r), no_chw, bool(term or trunc))

        # Bookkeeping
        ep_return += float(r)
        obs = nobs
        steps += 1

        # Episode end
        if term or trunc:
            # (optional) print episodic return or log it here
            obs, _ = env.reset()
            ep_return = 0.0

        # Learning step
        if steps > learning_starts and (steps % train_freq == 0):
            batch = rb.sample(batch_size)
            _ = L.train_step(batch)  # (optionally log the loss)

        # Target syncdone
        if steps - last_target_sync >= target_interval:
            L.target_update()
            last_target_sync = steps

    env.close()
    torch.save(q_online.state_dict(), os.path.join(save_dir, "dqn_pong_nature.pt"))
