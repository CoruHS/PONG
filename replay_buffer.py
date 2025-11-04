import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, dtype=np.uint8):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.idx = 0
        self.full = False

    def add(self, s, a, r, s2, d):
        self.obs[self.idx] = s
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.next_obs[self.idx] = s2
        self.dones[self.idx] = d
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        n = len(self)
        idxs = np.random.randint(0, n, size=batch_size)
        return (self.obs[idxs],
                self.actions[idxs],
                self.rewards[idxs],
                self.next_obs[idxs],
                self.dones[idxs])
