import numpy as np

class EpsilonScheduler:
    """Linear anneal from eps_start -> eps_final over decay_steps."""
    def __init__(self, eps_start=1.0, eps_final=0.01, decay_steps=2_000_000):
        self.eps_start = float(eps_start)
        self.eps_final = float(eps_final)
        self.decay_steps = int(decay_steps)

    def value(self, t: int) -> float:
        if t >= self.decay_steps:
            return self.eps_final
        frac = t / self.decay_steps
        return self.eps_start + frac * (self.eps_final - self.eps_start)


def select_action_eps_greedy(
    obs_uint8_chw: np.ndarray,
    n_actions: int,
    epsilon: float,
    rng: np.random.Generator,
    q_values_fn=None,
) -> int:
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
