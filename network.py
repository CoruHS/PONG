import torch, torch.nn as nn

class NatureCNN(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(True),
        )

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64*7*7, 512), nn.ReLU(True))
    def forward(self, x):                   # x: (B,4,84,84) in [0,1]
        return self.fc(self.conv(x))
class QNetwork(nn.Module):
    def __init__(self, n_actions:int, in_channels = 4):
        super().__init__()
        self.backbone = NatureCNN(in_channels)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        return self.head(self.backbone(x))
