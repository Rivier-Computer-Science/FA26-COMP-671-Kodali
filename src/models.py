import torch.nn as nn

class MLP(nn.Module):
    """Baseline 2: MLP on flattened low-res pixels."""
    def __init__(self, num_classes, size=32, hidden=(1024, 512, 256), p=0.3):
        super().__init__()
        layers, d = [nn.Flatten()], 3 * size * size
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]; d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    """Two 3x3 conv-BN layers; skip connection toggled by `residual` (controlled ablation)."""
    def __init__(self, cin, cout, stride, residual):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False);     self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.short = None
        if residual and (stride != 1 or cin != cout):
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + (x if self.short is None else self.short(x))
        return self.relu(out)

class ConvNet(nn.Module):
    """Identical depth/width for CNN (residual=False) and ResNet (residual=True)."""
    def __init__(self, num_classes, residual, width=32, blocks_per_stage=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(2))
        layers, cin = [], width
        for i, w in enumerate([width, width * 2, width * 4]):
            for b in range(blocks_per_stage):
                stride = 2 if (b == 0 and i > 0) else 1
                layers.append(Block(cin, w, stride, residual)); cin = w
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(cin, num_classes))
    def forward(self, x): return self.head(self.stages(self.stem(x)))

def build_model(name, num_classes, cfg):
    if name == "mlp":
        return MLP(num_classes, cfg["mlp_img_size"])
    if name in ("cnn", "resnet"):
        return ConvNet(num_classes, residual=(name == "resnet"),
                       width=cfg["base_width"], blocks_per_stage=cfg["blocks_per_stage"])
    raise ValueError(name)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
