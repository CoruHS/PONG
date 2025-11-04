import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Tuple

from models import make_q_network, NatureDQN  # NatureDQN kept for back-compat


@dataclass
class AgentState:
    policy: torch.nn.Module
    target: torch.nn.Module
    optim: torch.optim.Optimizer
    step: int = 0


class DQNAgent:
    """
    Double DQN with target network, Huber loss, gradient clipping.
    Accepts uint8 replay batches (normalized inside the network).
    """
    def __init__(self,
                 obs_shape: Tuple[int, int, int],
                 n_actions: int,
                 cfg,
                 arch: str = "nature"):
        c, h, w = obs_shape
        device_str = cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
        self.device = torch.device(device_str)

        # Networks (explicit keyword args to avoid channel/order bugs)
        self.policy = make_q_network(arch, n_actions=n_actions, in_channels=c).to(self.device)
        self.target = make_q_network(arch, n_actions=n_actions, in_channels=c).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        # Training hyperparams
        self.gamma = cfg.gamma
        self.criterion = nn.SmoothL1Loss()  # Huber
        self.optim = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.train_freq = cfg.train_freq
        self.target_sync_interval = cfg.target_sync_interval
        self.step = 0

    @torch.no_grad()
    def act(self, state_uint8, epsilon: float) -> int | None:
        """
        Epsilon-greedy action.
        Returns None to signal caller to sample random action when exploring.
        """
        if torch.rand(()) < epsilon:
            return None
        x = torch.as_tensor(state_uint8, device=self.device).unsqueeze(0)  # (1,C,H,W) uint8
        q = self.policy(x)
        return int(torch.argmax(q, dim=1).item())

    def optimize(self, batch, clip_grad: bool = True) -> float:
        """
        One Double DQN update from a replay batch of:
          s(uint8), a(int64), r(float32), s2(uint8), d(bool/float)
        """
        s, a, r, s2, d = batch

        s  = torch.as_tensor(s,  device=self.device)  # (B,C,H,W) uint8
        s2 = torch.as_tensor(s2, device=self.device)  # (B,C,H,W) uint8
        a  = torch.as_tensor(a,  device=self.device, dtype=torch.long)       # (B,)
        r  = torch.as_tensor(r,  device=self.device, dtype=torch.float32)    # (B,)
        d  = torch.as_tensor(d,  device=self.device, dtype=torch.float32)    # (B,)

        # Current Q(s, a)
        q = self.policy(s)                              # (B, A)
        q_sa = q.gather(1, a.unsqueeze(1)).squeeze(1)   # (B,)

        # Target: Double DQN
        with torch.no_grad():
            next_q_policy = self.policy(s2)             # (B, A)
            next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)  # (B,1)
            next_q_target = self.target(s2).gather(1, next_actions).squeeze(1)  # (B,)
            target = r + (1.0 - d) * self.gamma * next_q_target               # (B,)

        loss = self.criterion(q_sa, target)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optim.step()

        self.step += 1
        if self.step % self.target_sync_interval == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "optim": self.optim.state_dict(),
            "step": self.step
        }, path)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.target.load_state_dict(ckpt["target"])
        self.optim.load_state_dict(ckpt["optim"])
        self.step = ckpt.get("step", 0)
