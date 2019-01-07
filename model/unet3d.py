

import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(self, in_ch, out_ch, degree=1):
        super(UNet3D, self).__init__()

        chs = [4, 8, 16, 32, 64] * degree

        self.downLayer1 = ConvBlock3d(in_ch, chs[0])
        self.downLayer2 = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
                                   ConvBlock3d(chs[0], chs[1]))

        self.downLayer3 = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
                                   ConvBlock3d(chs[1], chs[2]))

        self.downLayer4 = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
                                   ConvBlock3d(chs[2], chs[3]))

        self.bottomLayer = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
                                   ConvBlock3d(chs[3], chs[4]))

        self.upLayer1 = UpBlock(chs[4], chs[3])
        self.upLayer2 = UpBlock(chs[3], chs[2])
        self.upLayer3 = UpBlock(chs[2], chs[1])
        self.upLayer4 = UpBlock(chs[1], chs[0])

        self.outLayer = ConvBlock3d(chs[0], out_ch)

    def forward(self, x):
        x1 = self.downLayer1(x)
        x2 = self.downLayer2(x1)
        x3 = self.downLayer3(x2)
        x4 = self.downLayer4(x3)
        x5 = self.bottomLayer(x4)

        x = self.upLayer1(x5, x4)
        x = self.upLayer2(x, x3)
        x = self.upLayer3(x, x2)
        x = self.upLayer4(x, x1)

        x = self.outLayer(x)

        return x


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock3d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
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
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.up_conv = ConvTrans3d(in_ch, out_ch)
        self.conv = ConvBlock3d(2 * out_ch, out_ch)

    def forward(self, x, down_features):
        x = self.up_conv(x)
        x = torch.cat([x, down_features], dim=1)
        x = self.conv(x)
        return x


# test case
if __name__ == "__main__":
    net = UNet3D(4, 1)

    x = torch.randn(4, 4, 16, 64, 64)  # batch size = 4
    print ('input data')
    print (x.shape)

    if torch.cuda.is_available():
        net = net.cuda()
        x = x.cuda()

    y = net(x)
    print ('output data')
    print (y.shape)  # (4, 1, 16, 64, 64)




