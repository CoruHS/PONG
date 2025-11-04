# train_env.py
# pip install -U "pettingzoo[atari]" gymnasium pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo.atari import pong_v3

# ======================
# Heuristic opponent
# ======================
class AtariPongHeuristic:
    """
    Simple opponent:
      - presses FIRE a few frames if no ball (kick off the serve)
      - tracks the ball with UP/DOWN, with a small deadzone
      - respects action masks if present
    meanings: list like ["NOOP","FIRE","UP","DOWN", ...]
    side: "left" or "right" (used only for paddle detection fallback)
    """
    def __init__(self, meanings, side: str, k_p: float = 0.4, deadzone: int = 3):
        self.meanings = meanings or []
        self.side = side
        self.deadzone = int(deadzone)
        self.k_p = float(k_p)

        # Map meanings to indices with safe fallbacks
        self.idx_NOOP = self._idx_or_default("NOOP", default=0)
        self.idx_FIRE = self._idx_or_default("FIRE", default=self.idx_NOOP)
        self.idx_UP   = self._idx_or_default("UP",   default=self.idx_NOOP)
        self.idx_DOWN = self._idx_or_default("DOWN", default=self.idx_NOOP)
        self._serve_countdown = 0

    def _idx_or_default(self, token, default=0):
        try:
            return self.meanings.index(token)
        except ValueError:
            return default

    def __call__(self, frame: np.ndarray, action_mask=None) -> int:
        # Expect RGB frame (H,W,3). If not, safe NOOP.
        if frame is None or frame.ndim != 3 or frame.shape[-1] != 3:
            return self.idx_NOOP

        H, W, _ = frame.shape

        # Fast grayscale on a 2x downsampled view
        small = frame[::2, ::2]
        gray = (0.299*small[...,0] + 0.587*small[...,1] + 0.114*small[...,2]).astype(np.float32)

        # Mask out the net (center vertical stripe)
        mid_x = gray.shape[1] // 2
        net_mask = np.ones_like(gray, dtype=bool)
        net_mask[:, max(0, mid_x-2): mid_x+3] = False

        # Bright-ish pixels ≈ ball
        bright = (gray > 200.0) & net_mask

        # Ignore a few cols near borders
        field_mask = np.ones_like(bright, dtype=bool)
        field_mask[:, :6] = False
        field_mask[:, -6:] = False

        ball_cands = np.argwhere(bright & field_mask)
        ball_y = None
        if ball_cands.size > 0:
            cy, cx = ball_cands.mean(axis=0)
            ball_y = float(cy) * 2.0  # undo downsample

        # Approx paddle center (simple brightness scan near edge)
        paddle_col = frame[:, :8, :] if self.side == "left" else frame[:, -8:, :]
        pgray = (0.299*paddle_col[...,0] + 0.587*paddle_col[...,1] + 0.114*paddle_col[...,2])
        rows = np.where(pgray > 180.0)[0]
        paddle_y = (rows.min() + rows.max()) / 2.0 if rows.size > 0 else H / 2.0

        # Control logic
        if ball_y is None:
            # Kick off serve periodically
            if self._serve_countdown <= 0:
                self._serve_countdown = 6
            act = self.idx_FIRE if self._serve_countdown > 0 else self.idx_NOOP
            self._serve_countdown = max(0, self._serve_countdown - 1)
        else:
            self._serve_countdown = 0
            error = ball_y - paddle_y
            if abs(error) <= self.deadzone:
                act = self.idx_NOOP
            else:
                act = self.idx_DOWN if error > 0 else self.idx_UP

        # Obey action mask if present
        if action_mask is not None:
            if not action_mask[act]:
                if action_mask[self.idx_NOOP]:
                    return self.idx_NOOP
                valid = np.flatnonzero(action_mask)
                return int(valid[0]) if valid.size > 0 else self.idx_NOOP

        return act


# ======================
# Helpers
# ======================
def _extract_frame(obs_any):
    """Return RGB frame regardless of Dict or raw array observation."""
    if isinstance(obs_any, dict):
        # PettingZoo Atari uses {"observation": (H,W,3), "action_mask": (n,)}
        return obs_any.get("observation", obs_any.get("rgb", None))
    return obs_any

def _extract_mask(obs_any):
    """Return action mask if present, else None."""
    if isinstance(obs_any, dict):
        return obs_any.get("action_mask")
    return None

def _pz_reset(env, seed=None):
    """
    Normalize PettingZoo parallel reset outputs.
    Returns: (obs_dict, infos)
    """
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


# ======================
# Gymnasium wrapper
# ======================
class PongLeftGym(gym.Env):
    """
    Single-agent Gymnasium wrapper:
      - Learner controls 'first_0' (left paddle).
      - Right paddle uses a moving heuristic (optionally noisy).
      - Returns ONLY left agent obs/reward/terminated/truncated/info.

    Render modes:
      - render_mode=None      : headless
      - render_mode="rgb_array": env.render() returns (H,W,3)
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, right_noise: float = 0.25, grayscale: bool = False):
        super().__init__()
        self.render_mode = render_mode
        self.right_noise = float(right_noise)

        # PettingZoo parallel env
        # Valid obs_type values are "rgb_image" (default) or "grayscale_image".
        if grayscale:
            self.env = pong_v3.parallel_env(
                render_mode=("rgb_array" if render_mode else None),
                obs_type="grayscale_image",
            )
        else:
            self.env = pong_v3.parallel_env(
                render_mode=("rgb_array" if render_mode else None)
                # default obs_type is "rgb_image"
            )

        # Observation space may be Dict(...) or Box
        obs_space_any = self.env.observation_space("first_0")
        if isinstance(obs_space_any, spaces.Dict):
            frame_space: spaces.Box = obs_space_any["observation"]
        else:
            frame_space: spaces.Box = obs_space_any
        self.observation_space = frame_space
        self.action_space = self.env.action_space("first_0")

        # Action meanings for heuristic:
        # In recent PettingZoo Atari, meanings are shared across agents.
        aec = pong_v3.env(obs_type="rgb_image")
        try:
            meanings = aec.unwrapped.get_action_meanings()
            if not isinstance(meanings, (list, tuple)):
                meanings = ["NOOP", "FIRE", "UP", "DOWN"]
            meanings_left = list(meanings)
            meanings_right = list(meanings)
        except Exception:
            meanings_left = ["NOOP", "FIRE", "UP", "DOWN"]
            meanings_right = ["NOOP", "FIRE", "UP", "DOWN"]
        finally:
            aec.close()

        self.right_policy = AtariPongHeuristic(meanings_right, side="right", k_p=0.35, deadzone=4)

        # Cache last full obs dict (for observe() fallback)
        self._last_obs_dict = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs_dict, _infos = _pz_reset(self.env, seed=seed)
        self._last_obs_dict = obs_dict
        left_obs = _extract_frame(obs_dict["first_0"])
        return left_obs, {}

    def step(self, action):
        # Right agent observation (prefer observe(); fall back to cached dict)
        try:
            right_obs_any = self.env.observe("second_0")
        except Exception:
            right_obs_any = None
        if right_obs_any is None:
            right_obs_any = self._last_obs_dict["second_0"]

        right_frame = _extract_frame(right_obs_any)
        right_mask = _extract_mask(right_obs_any)

        # Heuristic + optional noise
        right_act = self.right_policy(right_frame, right_mask)
        if np.random.random() < self.right_noise:
            if right_mask is not None:
                valid = [i for i, ok in enumerate(right_mask) if ok]
                if not valid:
                    valid = [0]
            else:
                valid = list(range(self.env.action_space("second_0").n))
            right_act = int(np.random.choice(valid))

        actions = {"first_0": int(action), "second_0": int(right_act)}
        obs_dict, rewards, terms, truncs, infos = self.env.step(actions)
        self._last_obs_dict = obs_dict

        left_obs = _extract_frame(obs_dict["first_0"])
        left_rew = float(rewards["first_0"])
        left_term = bool(terms["first_0"])
        left_trunc = bool(truncs["first_0"])
        left_info = infos.get("first_0", {})

        # Gymnasium step signature
        return left_obs, left_rew, left_term, left_trunc, left_info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.env.render()
        return None

    def close(self):
        self.env.close()


# ======== Quick smoke test ========
if __name__ == "__main__":
    env = PongLeftGym(render_mode="rgb_array", right_noise=0.2, grayscale=False)
    obs, info = env.reset(seed=0)
    done = False
    trunc = False
    total = 0.0
    for _ in range(2000):
        # Keep the learner "still" by taking NOOP if available
        a0 = 0 if env.action_space.contains(0) else env.action_space.sample()
        obs, r, done, trunc, info = env.step(a0)
        total += r
        if done or trunc:
            obs, info = env.reset()
    env.close()
    print("Smoke test reward:", total)
