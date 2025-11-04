import torch
import torch.nn as nn
from typing import Literal


def _orthogonal_(m: nn.Module, gain: float = 1.0):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class NatureCNN(nn.Module):
    """
    Feature extractor used in the Nature DQN paper.

    Expects input (B, C, 84, 84) where C is the frame stack (e.g., 4).
    Accepts either uint8 [0..255] or float [0..1]; auto-normalizes if uint8.
    Outputs a (B, 512) feature vector.
    """
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),          nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
        )
        self.apply(lambda m: _orthogonal_(m, gain=nn.init.calculate_gain("relu")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize if coming in as uint8 stacked frames
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        return self.fc(self.conv(x))


class QNetwork(nn.Module):
    """
    Plain Q-network: NatureCNN backbone + linear head to n_actions.
    """
    def __init__(self, *, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.backbone = NatureCNN(in_channels)
        self.head = nn.Linear(512, n_actions)
        _orthogonal_(self.head, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


class DuelingQNetwork(nn.Module):
    """
    Dueling architecture: splits into Value and Advantage streams.
    Q(a) = V + (A(a) - mean(A))
    """
    def __init__(self, *, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.backbone = NatureCNN(in_channels)
        self.value = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )
        self.apply(lambda m: _orthogonal_(m, gain=nn.init.calculate_gain("relu")))
        _orthogonal_(self.value[-1], gain=1.0)
        _orthogonal_(self.advantage[-1], gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        v = self.value(feats)               # (B, 1)
        a = self.advantage(feats)           # (B, A)
        a = a - a.mean(dim=1, keepdim=True) # zero-mean advantage
        return v + a


def make_q_network(arch: Literal["nature", "dueling"] = "nature",
                   *,
                   n_actions: int,
                   in_channels: int) -> nn.Module:
    if arch == "nature":
        return QNetwork(n_actions=n_actions, in_channels=in_channels)
    if arch == "dueling":
        return DuelingQNetwork(n_actions=n_actions, in_channels=in_channels)
    raise ValueError(f"Unknown arch: {arch}")


# Back-compat if other files import NatureDQN
NatureDQN = QNetwork
