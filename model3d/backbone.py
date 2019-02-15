import torch
import torch.nn as nn


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock3d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvTrans3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvTrans3d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class UpBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock3d, self).__init__()
        self.up_conv = ConvTrans3d(in_ch, out_ch)
        self.conv = ConvBlock3d(2 * out_ch, out_ch)

    def forward(self, x, down_features):
        x = self.up_conv(x)
        x = torch.cat([x, down_features], dim=1)
        x = self.conv(x)
        return x


def maxpool():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool