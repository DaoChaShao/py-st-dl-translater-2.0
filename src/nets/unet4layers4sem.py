#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 02:58
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   unet4layers4sem.py
# @Desc     :   


from torch import nn, cat, randn
from torchsummary import summary

from src.configs.cfg_unet import CONFIG4UNET

WIDTH: int = 64


class DoubleConv(nn.Module):
    """ A double convolutional layer block for UNet input processing """

    def __init__(self, in_channels: int, out_channels: int, mid_channels=None, height=None, width=None):
        super().__init__()
        self._C = in_channels
        self._H = height
        self._W = width
        if not mid_channels:
            mid_channels = out_channels

        # Setup input layers using nn.Sequential
        self._layers = nn.Sequential(
            # The 1st conv layer: keep original size
            nn.Conv2d(self._C, mid_channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            # The 2nd conv layer: keep original size
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

        # Initialise parameters
        self._layers.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self._layers(x)

    def summary(self):
        # input size: (batch, channels, height, width)
        if self._H is None or self._W is None:
            raise ValueError("Height and Width must be specified for model summary.")

        summary(self, (self._C, self._H, self._W))
        print(f"Model Summary for {self.__class__.__name__}")
        print("=" * WIDTH)
        print(self)
        print("=" * WIDTH)
        print()


class DownSampler(nn.Module):
    """ Encoding down-sampling block for UNet model """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._pool = nn.Sequential(
            # Down-sample by 2 using MaxPool
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            DoubleConv(in_channels, out_channels),

            # Alternative: Down-sample by 2 strides convolution
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True),
            # nn.Dropout(0.3),
            # nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(),
        )

    def forward(self, x):
        return self._pool(x)

    def summary(self):
        print("=" * WIDTH)
        print(self)
        print("=" * WIDTH)
        print()


class UpSampler(nn.Module):
    """ Decoding up-sampling block for UNet model """

    def __init__(self, decoder_channels: int, encoder_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            # Bilinear InterpolationUpsample without changing channel count
            self._up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            # ConvTranspose2d will reduce decoder channels
            self._up = nn.ConvTranspose2d(decoder_channels, encoder_channels, kernel_size=2, stride=2)
        in_channels: int = decoder_channels + encoder_channels
        out_channels: int = encoder_channels
        self._conv = DoubleConv(in_channels, out_channels)

    def forward(self, decoder_feat, encoder_feat):
        # Decode the features: upsample/transpose
        decoder_out = self._up(decoder_feat)

        # Find the size difference and pad if needed to match encoder spatial size (batch, channels, height, width)
        diff_H = encoder_feat.size()[2] - decoder_out.size()[2]
        diff_W = encoder_feat.size()[3] - decoder_out.size()[3]

        # Pad the decoder output to match encoder feature size: [left, right, top, bottom]
        padded = nn.functional.pad(decoder_out, [diff_W // 2, diff_W - diff_W // 2, diff_H // 2, diff_H - diff_H // 2])
        # Fusing encoder and decoder features - channel-wise concatenation (Skip Connections)
        out = cat([encoder_feat, padded], dim=1)

        # Apply the convolutional layers
        return self._conv(out)

    def summary(self):
        print("=" * WIDTH)
        print(self)
        print("=" * WIDTH)
        print()


class OutConv(nn.Module):
    """ Output convolutional layer for UNet model """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._output = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self._output(x)

    def summary(self):
        print("=" * WIDTH)
        print(self)
        print("=" * WIDTH)
        print()


class Standard4LayersUNetClassification(nn.Module):
    def __init__(self,
                 in_channels: int, n_classes: int,
                 height: int, width: int,
                 bilinear: bool = True, channels: int = 64):
        super().__init__()
        self._C = in_channels
        self._amount = n_classes
        self._H = height
        self._W = width
        self._B = bilinear

        # Encoder Architecture
        self._inc = DoubleConv(self._C, channels, height=self._H, width=self._W)
        self._down_i = DownSampler(channels, channels * 2)
        self._down_ii = DownSampler(channels * 2, channels * 4)
        self._down_iii = DownSampler(channels * 4, channels * 8)
        # Bottleneck layer: reduce channels by factor if bilinear
        self._down_iv = DownSampler(channels * 8, channels * 16)
        # Decoder Architecture
        self._up_i = UpSampler(channels * 16, channels * 8, bilinear=self._B)
        self._up_ii = UpSampler(channels * 8, channels * 4, bilinear=self._B)
        self._up_iii = UpSampler(channels * 4, channels * 2, bilinear=self._B)
        self._up_iv = UpSampler(channels * 2, channels, bilinear=self._B)

        self._outc = OutConv(channels, self._amount)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        out = self._inc(x)

        down_i = self._down_i(out)
        down_ii = self._down_ii(down_i)
        down_iii = self._down_iii(down_ii)
        down_iv = self._down_iv(down_iii)

        up_i = self._up_i(down_iv, down_iii)
        up_ii = self._up_ii(up_i, down_ii)
        up_iii = self._up_iii(up_ii, down_i)
        up_iv = self._up_iv(up_iii, out)

        # NOTE: Return raw logits (no softmax); use argmax(dim=1) during validation.
        logits = self._outc(up_iv)

        return logits

    def summary(self):
        # input size: (batch, channels, height, width)
        summary(self, (self._C, self._H, self._W))
        print(f"Model Summary for {self.__class__.__name__}")
        print("=" * WIDTH)
        print(self)
        print("=" * WIDTH)
        print()


if __name__ == "__main__":
    for bilinear in [True, False]:
        print(f"Testing with bilinear={bilinear}")

        model = Standard4LayersUNetClassification(
            in_channels=3, n_classes=1,
            height=CONFIG4UNET.PREPROCESSOR.IMAGE_HEIGHT,
            width=CONFIG4UNET.PREPROCESSOR.IMAGE_WIDTH,
            bilinear=bilinear
        )
        x = randn(1, 3, 256, 256)
        output = model(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")  # out: [1, 10, 256, 256]
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
