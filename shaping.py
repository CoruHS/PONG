# shaping.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Heuristics for 84x84 grayscale Atari Pong after your preprocessing
H, W = 84, 84
LEFT_X, RIGHT_X = 8, 75          # approximate paddle columns in resized frames
EDGE_PAD = 12                    # ignore near-edges when searching for ball
TOP_PAD, BOT_PAD = 10, 8         # ignore scoreboard/top/bottom artifacts
BRIGHT = 200                     # threshold for white pixels

def _fold_bounce(y: float, height: int = H) -> float:
    """Reflect y within [0, H-1] for ideal wall bounces."""
    period = 2 * (height - 1)
    y_mod = y % period
    return (period - y_mod) if y_mod > (height - 1) else y_mod

def _find_ball(frame: np.ndarray) -> tuple[float, float] | None:
    """Find ball by bright pixels outside paddle/scoreboard regions."""
    m = frame > BRIGHT
    m[:, :EDGE_PAD] = False
    m[:, W - EDGE_PAD:] = False
    m[:TOP_PAD, :] = False
    m[H - BOT_PAD:, :] = False
    coords = np.argwhere(m)
    if coords.size == 0:
        return None
    y, x = coords.mean(axis=0)   # centroid
    return float(x), float(y)

def _find_paddle_center_y(frame: np.ndarray, side: str) -> float | None:
    if side == "left":
        xs = slice(3, EDGE_PAD)
    else:
        xs = slice(W - EDGE_PAD, W - 3)
    m = frame > BRIGHT
    m[:TOP_PAD, :] = False
    m[H - BOT_PAD:, :] = False
    m = m[:, xs]
    coords = np.argwhere(m)
    if coords.size == 0:
        return None
    y = coords[:, 0].mean()
    return float(y)

@dataclass
class VisionState:
    ball_x: float | None
    ball_y: float | None
    ball_vx: float | None
    ball_vy: float | None
    left_py: float | None
    right_py: float | None

def extract_state(obs_chw: np.ndarray) -> VisionState:
    """
    obs_chw: (4,84,84) uint8 stacked frames, channel -1 is most recent.
    Returns ball pos/vel (from last two frames) and paddle centers from last frame.
    """
    f_prev = obs_chw[-2].astype(np.uint8)
    f_curr = obs_chw[-1].astype(np.uint8)

    bx0 = by0 = bx1 = by1 = None
    b0 = _find_ball(f_prev)
    b1 = _find_ball(f_curr)
    if b0 is not None:
        bx0, by0 = b0
    if b1 is not None:
        bx1, by1 = b1

    vx = vy = None
    if (bx0 is not None) and (bx1 is not None):
        vx = bx1 - bx0
        vy = by1 - by0

    lp = _find_paddle_center_y(f_curr, "left")
    rp = _find_paddle_center_y(f_curr, "right")

    return VisionState(bx1, by1, vx, vy, lp, rp)

def _intercept_y(x0: float, y0: float, vx: float, vy: float, x_target: float) -> float | None:
    if vx == 0:
        return None
    t = (x_target - x0) / vx
    if t <= 0:
        return None
    y_raw = y0 + vy * t
    return _fold_bounce(y_raw, H)

def _err(paddle_y: float | None, y_star: float | None) -> float | None:
    if paddle_y is None or y_star is None:
        return None
    return abs(paddle_y - y_star)

@dataclass
class ShapingCfg:
    alpha_intercept: float = 0.10     # potential-based intercept shaping weight
    beta_contact: float = 0.05        # contact bonus weight
    lambda_decay: float = 0.90        # per-rally decay for contact bonus
    step_cost: float = 1e-4           # tiny time penalty
    reachable_margin_px: float = 10.0 # "you could have reached it" threshold
    miss_penalty: float = 0.40        # penalty on reachable miss

class PongShaper:
    """
    Stateless vision; stateful rally counters to decay contact rewards.
    Call step(prev_obs, next_obs, r_env_left, r_env_right, gamma) each env step.
    """
    def __init__(self, cfg: ShapingCfg = ShapingCfg()):
        self.cfg = cfg
        self.rally_hits_left = 0
        self.rally_hits_right = 0

    def _contact(self, s_prev: VisionState, s_next: VisionState, side: str) -> bool:
        # A contact flips inbound→outbound near paddle x.
        if (s_prev.ball_x is None) or (s_prev.ball_vx is None) or (s_next.ball_x is None) or (s_next.ball_vx is None):
            return False
        if side == "left":
            inbound_prev = s_prev.ball_vx < 0
            inbound_next = s_next.ball_vx < 0
            near = (abs((s_prev.ball_x or 0) - LEFT_X) < 6) or (abs((s_next.ball_x or 0) - LEFT_X) < 6)
        else:
            inbound_prev = s_prev.ball_vx > 0
            inbound_next = s_next.ball_vx > 0
            near = (abs((s_prev.ball_x or 0) - RIGHT_X) < 6) or (abs((s_next.ball_x or 0) - RIGHT_X) < 6)
        return inbound_prev and (not inbound_next) and near

    def step(self, prev_obs_chw: np.ndarray, next_obs_chw: np.ndarray,
             r_env_left: float, r_env_right: float, gamma: float) -> tuple[float, float, dict]:
        s_prev = extract_state(prev_obs_chw)
        s_next = extract_state(next_obs_chw)

        # Intercept potentials (only when inbound)
        # LEFT
        phi_prev_L = phi_next_L = 0.0
        e_prev_L = None
        if (s_prev.ball_x is not None) and (s_prev.ball_y is not None) and (s_prev.ball_vx is not None) and (s_prev.ball_vy is not None):
            inbound_prev_L = s_prev.ball_vx < 0
            if inbound_prev_L:
                y_star_prev_L = _intercept_y(s_prev.ball_x, s_prev.ball_y, s_prev.ball_vx, s_prev.ball_vy, LEFT_X)
                e_prev_L = _err(s_prev.left_py, y_star_prev_L)
                if e_prev_L is not None:
                    phi_prev_L = - e_prev_L / H
        if (s_next.ball_x is not None) and (s_next.ball_y is not None) and (s_next.ball_vx is not None) and (s_next.ball_vy is not None):
            inbound_next_L = s_next.ball_vx < 0
            if inbound_next_L:
                y_star_next_L = _intercept_y(s_next.ball_x, s_next.ball_y, s_next.ball_vx, s_next.ball_vy, LEFT_X)
                e_next_L = _err(s_next.left_py, y_star_next_L)
                if e_next_L is not None:
                    phi_next_L = - e_next_L / H

        # RIGHT
        phi_prev_R = phi_next_R = 0.0
        e_prev_R = None
        if (s_prev.ball_x is not None) and (s_prev.ball_y is not None) and (s_prev.ball_vx is not None) and (s_prev.ball_vy is not None):
            inbound_prev_R = s_prev.ball_vx > 0
            if inbound_prev_R:
                y_star_prev_R = _intercept_y(s_prev.ball_x, s_prev.ball_y, s_prev.ball_vx, s_prev.ball_vy, RIGHT_X)
                e_prev_R = _err(s_prev.right_py, y_star_prev_R)
                if e_prev_R is not None:
                    phi_prev_R = - e_prev_R / H
        if (s_next.ball_x is not None) and (s_next.ball_y is not None) and (s_next.ball_vx is not None) and (s_next.ball_vy is not None):
            inbound_next_R = s_next.ball_vx > 0
            if inbound_next_R:
                y_star_next_R = _intercept_y(s_next.ball_x, s_next.ball_y, s_next.ball_vx, s_next.ball_vy, RIGHT_X)
                e_next_R = _err(s_next.right_py, y_star_next_R)
                if e_next_R is not None:
                    phi_next_R = - e_next_R / H

        cfg = self.cfg
        r_shape_L = cfg.alpha_intercept * (gamma * phi_next_L - phi_prev_L)
        r_shape_R = cfg.alpha_intercept * (gamma * phi_next_R - phi_prev_R)

        # Contact bonuses (decayed)
        contact_L = self._contact(s_prev, s_next, "left")
        contact_R = self._contact(s_prev, s_next, "right")
        r_contact_L = r_contact_R = 0.0
        if contact_L:
            self.rally_hits_left += 1
            r_contact_L = cfg.beta_contact * (cfg.lambda_decay ** (self.rally_hits_left - 1))
        if contact_R:
            self.rally_hits_right += 1
            r_contact_R = cfg.beta_contact * (cfg.lambda_decay ** (self.rally_hits_right - 1))

        # Reset rally count when a point is scored
        if (r_env_left != 0.0) or (r_env_right != 0.0):
            self.rally_hits_left = 0
            self.rally_hits_right = 0

        # Reachable miss penalty (if you lost the point and the last intercept error was small)
        miss_L = (r_env_left < 0.0) and (e_prev_L is not None) and (e_prev_L <= cfg.reachable_margin_px)
        miss_R = (r_env_right < 0.0) and (e_prev_R is not None) and (e_prev_R <= cfg.reachable_margin_px)
        r_miss_L = -cfg.miss_penalty if miss_L else 0.0
        r_miss_R = -cfg.miss_penalty if miss_R else 0.0

        # Step cost
        r_step = -cfg.step_cost

        r_total_L = float(r_env_left + r_shape_L + r_contact_L + r_miss_L + r_step)
        r_total_R = float(r_env_right + r_shape_R + r_contact_R + r_miss_R + r_step)

        dbg = {
            "phi_prev_L": phi_prev_L, "phi_next_L": phi_next_L,
            "phi_prev_R": phi_prev_R, "phi_next_R": phi_next_R,
            "contact_L": contact_L, "contact_R": contact_R,
            "e_prev_L": e_prev_L, "e_prev_R": e_prev_R,
            "r_env_L": r_env_left, "r_env_R": r_env_right,
            "r_shape_L": r_shape_L, "r_shape_R": r_shape_R,
            "r_contact_L": r_contact_L, "r_contact_R": r_contact_R,
            "r_miss_L": r_miss_L, "r_miss_R": r_miss_R,
            "r_step": r_step,
        }
        return r_total_L, r_total_R, dbg
