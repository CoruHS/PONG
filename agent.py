"""
Environment utilities for multi-agent Pong (PettingZoo).

Key decisions:
- Use PettingZoo pong_v3 (parallel API)
- SuperSuit preprocessing: grayscale → resize(84×84) → frame-stack=4 → uint8
- No agent_indicator_v0 so channel count stays 4 (matches your networks/replay)
- Observations from the env are HWC (84,84,4). Use hwc_to_chw_uint8 to convert to CHW (4,84,84).

Install once:
    pip install "pettingzoo[atari]" "autorom[accept-rom-license]" supersuit
    AutoROM --accept-license
"""
from __future__ import annotations
from typing import Tuple

import numpy as np

from pettingzoo.atari import pong_v3
from supersuit import color_reduction_v0, resize_v1, frame_stack_v1, dtype_v0


def make_pong_env(render_mode: str = "rgb_array", stack: int = 4):
    """
    Create Pong with Atari-style preprocessing.

    Returns
    -------
    env : PettingZoo parallel_env
        Observations are np.uint8 shaped (84,84,stack) == (84,84,4) in HWC order.
    """
    env = pong_v3.parallel_env(render_mode=render_mode)
    env = color_reduction_v0(env, mode="B")   # grayscale (fast single channel)
    env = resize_v1(env, x_size=84, y_size=84)
    env = frame_stack_v1(env, stack)
    env = dtype_v0(env, np.uint8)              # keep compact; scale later in model/learner
    return env


def hwc_to_chw_uint8(obs_hwc: np.ndarray) -> np.ndarray:
    """Convert (H,W,C) → (C,H,W). Expects uint8 and C==4 after frame stacking."""
    if obs_hwc.ndim != 3:
        raise ValueError(f"Expected (H,W,C), got shape {obs_hwc.shape}")
    if obs_hwc.shape[2] != 4:
        raise ValueError(f"Expected C==4 after frame stack, got C={obs_hwc.shape[2]}")
    if obs_hwc.dtype != np.uint8:
        raise TypeError(f"Expected uint8 input, got {obs_hwc.dtype}")
    return np.transpose(obs_hwc, (2, 0, 1))


def chw_to_hwc_uint8(obs_chw: np.ndarray) -> np.ndarray:
    """Convert (C,H,W) → (H,W,C). Helpful for logging/visualization."""
    if obs_chw.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got shape {obs_chw.shape}")
    if obs_chw.shape[0] != 4:
        raise ValueError(f"Expected C==4, got C={obs_chw.shape[0]}")
    if obs_chw.dtype != np.uint8:
        raise TypeError(f"Expected uint8 input, got {obs_chw.dtype}")
    return np.transpose(obs_chw, (1, 2, 0))


def scale_chw_to_float(x_chw_uint8: np.ndarray) -> np.ndarray:
    """Map CHW uint8 → CHW float32 in [0,1]. Useful before passing to torch."""
    if x_chw_uint8.dtype != np.uint8:
        raise TypeError("scale_chw_to_float expects uint8 array")
    return x_chw_uint8.astype(np.float32, copy=False) / 255.0
