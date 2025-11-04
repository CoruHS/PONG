import numpy as np
import cv2
from collections import deque
import random
import torch

def set_seed(seed: int):
    import os, torch, numpy as np, random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def preprocess_frame(frame: np.ndarray, out_size: int = 84) -> np.ndarray:
    """
    Convert RGB (H,W,3) to uint8 grayscale (out_size,out_size).
    Crop/resize like Nature DQN (simple resize works fine for Pong).
    """
    # to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # resize to square 84x84
    proc = cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return proc.astype(np.uint8)

class FrameStacker:
    """
    Keeps a stack of last K preprocessed frames (uint8), returns (K,H,W) uint8.
    """
    def __init__(self, k: int, h: int, w: int):
        self.k = k
        self.h = h
        self.w = w
        self.deque = deque(maxlen=k)

    def reset(self, first_frame: np.ndarray):
        self.deque.clear()
        f = preprocess_frame(first_frame, self.h)
        for _ in range(self.k):
            self.deque.append(f)
        return self.get_state()

    def step(self, frame: np.ndarray):
        f = preprocess_frame(frame, self.h)
        self.deque.append(f)
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return np.stack(self.deque, axis=0)  # (k,h,w)

def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    t = min(1.0, step / float(decay_steps))
    return start + t * (end - start)

def to_tensor(batch_np, device):
    return torch.as_tensor(batch_np, device=device)
