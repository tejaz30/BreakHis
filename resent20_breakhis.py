import torch
import torch.nn as nn
import torch.nn.functional as F

# === Basic Residual Block for ResNet-20 ===
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# === ResNet-20 Modified for BreakHis ===
class ResNet20(nn.Module):
    def __init__(self, num_classes=2):  # Binary classification
        super(ResNet20, self).__init__()
        self.in_planes = 16

        # first conv for 224x224 inputs - like BreakHis
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # ResNet-20 has (3, 3, 3) blocks
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)  # 🔁 MODIFIED: 64 → 2 for binary classification

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))     # Input: [B, 3, 224, 224]
        out = self.layer1(out)                    # -> [B, 16, H, W]
        out = self.layer2(out)                    # -> [B, 32, H/2, W/2]
        out = self.layer3(out)                    # -> [B, 64, H/4, W/4]
        out = self.avgpool(out)                   # -> [B, 64, 1, 1]
        out = out.view(out.size(0), -1)           # -> [B, 64]
        out = self.fc(out)                        # -> [B, 2]
        return out
