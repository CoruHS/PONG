
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class TransitionBatch:
    obs: np.ndarray         # (B, C, H, W) uint8
    actions: np.ndarray     # (B,) int32
    rewards: np.ndarray     # (B,) float32
    next_obs: np.ndarray    # (B, C, H, W) uint8
    dones: np.ndarray       # (B,) bool


class ReplayBuffer:
    """
    Minimal circular replay buffer.
    Assumes observations are already (C,H,W) uint8 when added.
    """
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, int, int],
        rng: Optional[np.random.Generator] = None,
    ):
        self.capacity = int(capacity)
        C, H, W = obs_shape
        assert C > 0 and H > 0 and W > 0, f"Bad obs_shape={obs_shape}"
        self.obs = np.empty((self.capacity, C, H, W), dtype=np.uint8)
        self.next_obs = np.empty((self.capacity, C, H, W), dtype=np.uint8)
        self.actions = np.empty((self.capacity,), dtype=np.int32)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.dones = np.empty((self.capacity,), dtype=np.bool_)
        self.idx = 0
        self.full = False
        self.rng = rng or np.random.default_rng()

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        # Validate incoming shapes once in a cheap way
        if obs.shape != self.obs.shape[1:]:
            raise ValueError(f"obs shape {obs.shape} does not match buffer shape {self.obs.shape[1:]}")
        if next_obs.shape != self.next_obs.shape[1:]:
            raise ValueError(f"next_obs shape {next_obs.shape} does not match buffer shape {self.next_obs.shape[1:]}")
        self.obs[self.idx] = obs
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = bool(done)
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
