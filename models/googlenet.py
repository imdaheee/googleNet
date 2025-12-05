# models/googlenet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block"""
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InceptionBlock(nn.Module):
    """
    Inception v1 block (GoogLeNet)

    구조:
    - branch1: 1x1 conv
    - branch2: 1x1 conv -> 3x3 conv
    - branch3: 1x1 conv -> 5x5 conv
    - branch4: 3x3 maxpool -> 1x1 conv
    """
    def __init__(self,
                 in_channels,
                 c1x1,
                 c3x3_reduce, c3x3,
                 c5x5_reduce, c5x5,
                 pool_proj):
        super().__init__()

        # 1x1 branch
        self.branch1 = ConvBNReLU(
            in_channels, c1x1,
            kernel_size=1,
        )

        # 1x1 -> 3x3 branch
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, c3x3_reduce, kernel_size=1),
            ConvBNReLU(c3x3_reduce, c3x3, kernel_size=3, padding=1),
        )

        # 1x1 -> 5x5 branch
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, c5x5_reduce, kernel_size=1),
            ConvBNReLU(c5x5_reduce, c5x5, kernel_size=5, padding=2),
        )

        # 3x3 maxpool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # channel 방향으로 concat
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return out


class AuxClassifier(nn.Module):
    """
    Auxiliary classifier used during training.

    입력: Inception 4a 또는 4d의 feature map
    출력: 클래스별 logits
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # paper 기준: 5x5 avg pool, stride 3
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBNReLU(in_channels, 128, kernel_size=1)

        # 입력 이미지 224x224 기준일 때, 4a/4d에서 feature map 크기 ≈ 14x14
        # -> AvgPool(5,3) 거치면 4x4 정도 => 128 * 4 * 4
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) implementation.

    기본 입력 크기: 3 x 224 x 224
    """
    def __init__(self, num_classes=1000, use_aux=True):
        super().__init__()
        self.use_aux = use_aux

        # ---------- stem ----------
        self.conv1 = ConvBNReLU(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = ConvBNReLU(64, 64, kernel_size=1)
        self.conv3 = ConvBNReLU(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---------- Inception 3a, 3b ----------
        self.inception3a = InceptionBlock(
            in_channels=192,
            c1x1=64,
            c3x3_reduce=96, c3x3=128,
            c5x5_reduce=16, c5x5=32,
            pool_proj=32,
        )
        self.inception3b = InceptionBlock(
            in_channels=256,
            c1x1=128,
            c3x3_reduce=128, c3x3=192,
            c5x5_reduce=32, c5x5=96,
            pool_proj=64,
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---------- Inception 4a ~ 4e ----------
        self.inception4a = InceptionBlock(
            in_channels=480,
            c1x1=192,
            c3x3_reduce=96, c3x3=208,
            c5x5_reduce=16, c5x5=48,
            pool_proj=64,
        )
        self.inception4b = InceptionBlock(
            in_channels=512,
            c1x1=160,
            c3x3_reduce=112, c3x3=224,
            c5x5_reduce=24, c5x5=64,
            pool_proj=64,
        )
        self.inception4c = InceptionBlock(
            in_channels=512,
            c1x1=128,
            c3x3_reduce=128, c3x3=256,
            c5x5_reduce=24, c5x5=64,
            pool_proj=64,
        )
        self.inception4d = InceptionBlock(
            in_channels=512,
            c1x1=112,
            c3x3_reduce=144, c3x3=288,
            c5x5_reduce=32, c5x5=64,
            pool_proj=64,
        )
        self.inception4e = InceptionBlock(
            in_channels=528,
            c1x1=256,
            c3x3_reduce=160, c3x3=320,
            c5x5_reduce=32, c5x5=128,
            pool_proj=128,
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---------- Inception 5a, 5b ----------
        self.inception5a = InceptionBlock(
            in_channels=832,
            c1x1=256,
            c3x3_reduce=160, c3x3=320,
            c5x5_reduce=32, c5x5=128,
            pool_proj=128,
        )
        self.inception5b = InceptionBlock(
            in_channels=832,
            c1x1=384,
            c3x3_reduce=192, c3x3=384,
            c5x5_reduce=48, c5x5=128,
            pool_proj=128,
        )

        # ---------- Auxiliary classifiers ----------
        if self.use_aux:
            # Inception 4a 출력 채널 수: 512
            self.aux1 = AuxClassifier(in_channels=512, num_classes=num_classes)
            # Inception 4d 출력 채널 수: 528
            self.aux2 = AuxClassifier(in_channels=528, num_classes=num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        # ---------- Final classifier ----------
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # ----- stem -----
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        # ----- 3a, 3b -----
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # ----- 4a -----
        x = self.inception4a(x)

        aux1 = None
        if self.use_aux and self.training and self.aux1 is not None:
            aux1 = self.aux1(x)

        # ----- 4b, 4c, 4d -----
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        aux2 = None
        if self.use_aux and self.training and self.aux2 is not None:
            aux2 = self.aux2(x)

        # ----- 4e -----
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # ----- 5a, 5b -----
        x = self.inception5a(x)
        x = self.inception5b(x)

        # ----- classifier -----
        x = self.global_avg_pool(x)      # (N, 1024, 1, 1)
        x = torch.flatten(x, 1)          # (N, 1024)
        x = self.dropout(x)
        x = self.fc(x)                   # (N, num_classes)

        if self.use_aux and self.training:
            # 학습 모드일 때: main + aux1 + aux2 모두 반환
            return x, aux1, aux2
        else:
            # 평가 모드일 때: main logits만 반환
            return x


def googlenet(num_classes=1000, use_aux=True):
    """
    Helper function to create GoogLeNet instance.

    사용 예:
        model = googlenet(num_classes=10, use_aux=True)
    """
    return GoogLeNet(num_classes=num_classes, use_aux=use_aux)
