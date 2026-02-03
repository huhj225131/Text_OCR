import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1 , downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=stride,
                               padding="same",
                               bias = False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding="same")
        self.bn2  = nn.BatchNorm2d(num_features=out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out  = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return out

class ResNetBackBone(nn.Module):
    "[B, 1,480,960] -> "
    def __init__(self, d_model = 384):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride= 2,
            padding ="same",
            bias=True
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, d_model, 2, stride=1)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride= stride,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
            layers.append(ResBlock(in_channels=in_channels,
                                   out_channels=out_channels,
                                   stride=stride,
                                   downsample=downsample))
            for _ in range(blocks - 1):
                layers.append(ResBlock(in_channels=out_channels,
                                       out_channels=out_channels))
            return nn.Sequential(*layers)
    def forward(self, x :torch.Tensor):
        assert tuple(x.shape[-2:]) == (480, 960)

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x)  
        return x