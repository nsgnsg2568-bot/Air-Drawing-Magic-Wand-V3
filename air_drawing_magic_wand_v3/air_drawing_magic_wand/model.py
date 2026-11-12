"""A small CNN for 28x28 sketches (QuickDraw style)."""
import torch.nn as nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class QuickDrawCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            block(1, 32), nn.MaxPool2d(2),     # 28 -> 14
            block(32, 64), nn.MaxPool2d(2),    # 14 -> 7
            block(64, 128), nn.MaxPool2d(2),   # 7  -> 3
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
