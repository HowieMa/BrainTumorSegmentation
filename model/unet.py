import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()

        chs = [64, 128, 256, 512, 1024]

        self.down1 = nn.Sequential(Conv3x3(in_ch, chs[0]))
        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(chs[0], chs[1]))
        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(chs[1], chs[2]))
        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(chs[2], chs[3]))
        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(chs[3], chs[4]))

        self.up1 = Up(chs[4], chs[3])
        self.up2 = Up(chs[3], chs[2])
        self.up3 = Up(chs[2], chs[1])
        self.up4 = Up(chs[1], chs[0])

        self.out = Conv3x3(chs[0], out_ch)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.bottom(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out(x)
        return x


class Conv3x3(nn.Module):
    def __init__(self,in_chl, out_chl):
        super(Conv3x3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chl, out_chl,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chl),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chl, out_chl, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chl),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Up(nn.Module):
    def __init__(self, in_chl, out_chl):
        super(Up, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_chl, out_chl, kernel_size=2, stride=2)
        self.conv3x3 = Conv3x3(2 * out_chl, out_chl)

    def forward(self, inputs, down_features):
        out = self.up_conv(inputs)
        out_cat = torch.cat([out, down_features], dim=1)
        out = self.conv3x3(out_cat)
        return out


if __name__ == "__main__":
    net = UNet(1, 3)
    a = torch.randn(4, 1, 64, 64)
    b = net(a)
    print b.shape

