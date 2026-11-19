"""Dueling convolutional Q-network for the Objective 3 prototype."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from env2048 import NUM_CHANNELS


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class DuelingCNN(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, width=128, n_blocks=4, n_actions=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(n_blocks)])
        self.value_head = nn.Sequential(
            nn.Conv2d(width, 32, 1), nn.ReLU(inplace=True),
            nn.Flatten(), nn.LazyLinear(128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Conv2d(width, 32, 1), nn.ReLU(inplace=True),
            nn.Flatten(), nn.LazyLinear(128), nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        # x: (B, C, H, W), last channel is the in-bounds mask
        mask = x[:, -1:, :, :]  # (B,1,H,W)
        feat = self.stem(x)
        feat = self.blocks(feat)
        feat = feat * mask  # zero out padded cells before the heads pool over them
        v = self.value_head(feat)  # (B,1)
        a = self.adv_head(feat)  # (B,4)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


def masked_argmax(q, legal_mask):
    """q, legal_mask: (B, 4)."""
    neg_inf = torch.finfo(q.dtype).min
    masked_q = torch.where(legal_mask, q, torch.full_like(q, neg_inf))
    return masked_q.argmax(dim=1)
