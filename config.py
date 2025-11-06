# main.py
import os
import numpy as np
import torch
import cv2
from collections import deque

from agent import make_env
from replay import ReplayBuffer
from policy import EpsilonScheduler, select_action_eps_greedy
from network import QNetwork
from learner import Learner


# ---- Score tracking ----------------------------------------------------------
class ScoreTracker:
    def __init__(self, ma_episodes: int = 50):
        self.ep_scores = deque(maxlen=ma_episodes)   # episode returns (right paddle)
        self.ep_wins = deque(maxlen=ma_episodes)     # 1 if win (return>0)
        self.you = 0     # right paddle (training agent)
        self.cpu = 0     # left paddle (handcoded opponent)
        self.current_return = 0.0

    def step(self, reward: float):
        if reward > 0:
            self.you += 1
        elif reward < 0:
            self.cpu += 1
        self.current_return += reward

    def end_episode(self):
        self.ep_scores.append(self.current_return)
        self.ep_wins.append(1 if self.current_return > 0 else 0)
        you, cpu, ret = self.you, self.cpu, self.current_return
        self.you = 0
        self.cpu = 0
        self.current_return = 0.0
        return you, cpu, ret

    def moving_avg_return(self) -> float:
        if not self.ep_scores:
            return 0.0
        return float(sum(self.ep_scores) / len(self.ep_scores))

    def win_rate(self) -> float:
        if not self.ep_wins:
            return float("nan")
        return 100.0 * (sum(self.ep_wins) / len(self.ep_wins))

    def stats_str(self) -> str:
        if not self.ep_scores:
            total_points = self.you + self.cpu
            win_rate = (100.0 * self.you / total_points) if total_points > 0 else float("nan")
            return f"avg_return@0={self.current_return:+.2f} | win_rate={win_rate:.1f}%"
        avg_ret = self.moving_avg_return()
        win_rate = self.win_rate()
        return f"avg_return@{len(self.ep_scores)}={avg_ret:+.2f} | win_rate={win_rate:.1f}%"


# ---- Scoreboard helper -------------------------------------------------------
def format_scoreboard(ep: int, steps: int, epsilon: float,
                      you: int, cpu: int, tracker: ScoreTracker) -> str:
    """
    Produce a compact, single-line scoreboard for the just-finished match.

    you: right paddle (training agent)
    cpu: left paddle (handcoded opponent policy controlling pong_v3's left side)
    """
    avg_ret = tracker.moving_avg_return()
    return (
        f"[scoreboard] ep={ep} | steps={steps:,} | eps={epsilon:.3f} | "
        f"score(training)={you} | score(handcoded)={cpu} | avg_return={avg_ret:+.2f}"
    )


# ---- Obs normalization -------------------------------------------------------
def to_chw(x: np.ndarray) -> np.ndarray:
    """
    Convert raw Atari frames to (C,H,W) uint8 with C∈{1,4}.
    - If input is (210,160,3) RGB: -> grayscale -> resize to (84,84) -> (1,84,84).
    - If already CHW (C,H,W) or HWC (H,W,C) with C in {1,4} and square spatial dims, keep/transpose.
    - If (H,W) grayscale, expand to (1,H,W).
    """
    x = np.asarray(x)
    if x.ndim == 3 and x.shape == (210, 160, 3):
        gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[None, ...]  # (1,84,84)
    if x.ndim == 2:  # (H,W) -> (1,H,W)
        return x[None, ...]
    if x.ndim == 3:
        # Already CHW?
        if x.shape[0] in (1, 4) and x.shape[1] == x.shape[2]:
            return x
        # HWC square with channel 1 or 4?
        if x.shape[-1] in (1, 4) and x.shape[0] == x.shape[1]:
            return np.transpose(x, (2, 0, 1))
    raise ValueError(f"Unexpected obs shape {x.shape}; expected raw (210,160,3), (H,W), or square CHW/HWC with C∈{{1,4}}")


# ---- Frame stacker -----------------------------------------------------------
class FrameStacker:
    def __init__(self, k: int = 4):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, first_chw: np.ndarray) -> np.ndarray:
        """Seed the stack with k copies of the first (1,84,84) frame -> (4,84,84)."""
        assert first_chw.shape[0] == 1, f"expected (1,84,84), got {first_chw.shape}"
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(first_chw)
        return np.concatenate(list(self.frames), axis=0)

    def push(self, next_chw: np.ndarray) -> np.ndarray:
        """Append a new (1,84,84) frame and return stacked (4,84,84)."""
        assert next_chw.shape[0] == 1, f"expected (1,84,84), got {next_chw.shape}"
        self.frames.append(next_chw)
        return np.concatenate(list(self.frames), axis=0)


# ---- Training ----------------------------------------------------------------
def train(
    total_steps: int = 1_000_000,
    buffer_size: int = 100_000,
    batch_size: int = 32,
    learning_starts: int = 50_000,
    train_freq: int = 4,
    target_interval: int = 10_000,
    sticky: float = 0.0,  # kept for compatibility (unused by our wrapper)
    save_dir: str = "models",
    seed: int = 0,
    device: str = None,   # auto-select below
):
    os.makedirs(save_dir, exist_ok=True)

    # --- Env & RNG ---
    env = make_env(opponent="sticky_up_down", render_mode=None)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    # Normalize first obs and build 4-frame state
    first_chw = to_chw(obs).astype(np.uint8, copy=False)      # (1,84,84)
    stacker = FrameStacker(k=4)
    state = stacker.reset(first_chw)                          # (4,84,84)

    # Replay matches stacked shape
    rb = ReplayBuffer(capacity=buffer_size, obs_shape=state.shape)

    # --- Epsilon schedule (slow decay over whole run) ---
    eps = EpsilonScheduler(1.0, 0.01, decay_steps=0.7*total_steps)

    # --- Networks & learner ---
    n_actions = env.action_space.n
    q_online = QNetwork(n_actions=n_actions)
    q_target = QNetwork(n_actions=n_actions)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    L = Learner(q_online, q_target, n_actions, device=device)

    q_online.train()
    q_target.eval()

    # Helper: Q-values → NumPy without grad
    # in main.py
    def q_values_fn_numpy(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x_np).float().div(255.0).to(device)  # scale!
            return q_online(x_t).detach().cpu().numpy()

    steps = 0
    last_target_sync = 0
    tracker = ScoreTracker(ma_episodes=50)

    # Episode/match counters for summaries
    ep_idx = 0
    episode_steps = 0
    PRINT_POINTS = False  # set True if you want a line per point scored

    # Heartbeat so runs aren't silent
    def log_heartbeat():
        if steps % 10_000 == 0 and steps > 0:
            print(
                f"[train] steps={steps:,} | eps={eps.value(max(0, steps-learning_starts)):.3f} "
                f"| rb={len(rb):,} | {tracker.stats_str()}"
            )

    while steps < total_steps:
        # Current stacked state
        o_chw = state  # (4,84,84)

        # ε-greedy action (keep ε=1.0 before learning starts)
        epsilon = 1.0 if steps < learning_starts else eps.value(steps - learning_starts)
        a = select_action_eps_greedy(
            o_chw, n_actions, epsilon, rng, q_values_fn=q_values_fn_numpy
        )

        # Env step
        nobs, r, term, trunc, _ = env.step(int(a))
        tracker.step(float(r))

        if PRINT_POINTS and r != 0:
            who = "training" if r > 0 else "handcoded"
            print(f"[point] steps={steps:,} | {who} scored | score training:{tracker.you} handcoded:{tracker.cpu}")

        # Next stacked state
        no_chw_single = to_chw(nobs).astype(np.uint8, copy=False)  # (1,84,84)
        next_state = stacker.push(no_chw_single)                   # (4,84,84)

        # Store transition
        rb.add(o_chw, int(a), float(r), next_state, bool(term or trunc))

        # Bookkeeping
        obs = nobs
        state = next_state
        steps += 1
        episode_steps += 1
        log_heartbeat()

        # Episode end (natural only; no artificial cap)
        if term or trunc:
            you, cpu, ret = tracker.end_episode()
            ep_idx += 1
            winner = "RIGHT (training)" if ret > 0 else ("LEFT (handcoded)" if ret < 0 else "tie/timeout")

            # Verbose end-of-match line (kept)
            print(
                f"[match] ep={ep_idx} | result={winner} | final_score training:{you} handcoded:{cpu} "
                f"| return={ret:+.1f} | steps_in_match={episode_steps}"
            )
            # Rolling stats (kept)
            print(f"[episode] steps={steps:,} | {tracker.stats_str()}")

            # One-line scoreboard (requested)
            eps_now = 1.0 if steps < learning_starts else eps.value(steps - learning_starts)
            print(format_scoreboard(ep_idx, steps, eps_now, you, cpu, tracker))

            # Reset env + frame stack for next match
            obs, _ = env.reset()
            first_chw = to_chw(obs).astype(np.uint8, copy=False)
            state = stacker.reset(first_chw)
            episode_steps = 0

        # Learning step
        if steps > learning_starts and (steps % train_freq == 0):
            batch = rb.sample(batch_size)
            _ = L.train_step(batch)

        # Target sync
        if steps - last_target_sync >= target_interval:
            L.target_update()
            last_target_sync = steps

    env.close()
    torch.save(q_online.state_dict(), os.path.join(save_dir, "dqn_pong_nature.pt"))
    print(f"[done] saved to {os.path.join(save_dir, 'dqn_pong_nature.pt')}")


if __name__ == "__main__":
    train()
