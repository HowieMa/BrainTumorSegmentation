# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class conv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(conv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv3d(in_ch,out_ch,(3,3,3),padding=(1,1,1)),

            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
        )

    def forward(self, x):
        x=self.conv(x)
        return x


class ResBlock(nn.Module):
    """

    """
    def __init__(self,in_ch, out_ch, d=1):
        """

        :param in_ch:
        :param out_ch:
        :param d:
        """
        super(ResBlock,self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (3, 3, 1), padding=(d, d, 0), dilation=(d, d, 1)),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
            nn.Conv3d(out_ch, out_ch, (3, 3, 1), padding=(d, d, 0), dilation=(d, d, 1)),
            nn.BatchNorm3d(out_ch),
            nn.PReLU()
        )

    def forward(self, x):
        x1 = self.resblock(x)       #
        x = x + x1
        return x


class FuseLayer(nn.Module):
    """
    inter slice kernel 1 * 1 * 3
    1x1x3 convolution output channel Co
    Green block in Fig.2 of paper
    """
    def __init__(self, in_ch, out_ch):
        super(FuseLayer, self).__init__()
        self.ant = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1, 3), padding=(0, 0, 1), dilation=1),
            nn.BatchNorm3d(out_ch),     # batch normalization layer
            nn.PReLU(),                 # action layer
        )

    def forward(self, x):
        x1 = self.ant(x)
        return x1


class ResBlock3(nn.Module):
    def __init__(self, in_ch, out_ch, flag):
        super(ResBlock3, self).__init__()
        if flag == 1:  #
            self.block = nn.Sequential(
                ResBlock(in_ch,  out_ch, 1),  #
                ResBlock(out_ch, out_ch, 2),
                ResBlock(out_ch, out_ch, 3),
                FuseLayer(out_ch, out_ch)
            )
        else:
            self.block = nn.Sequential(
                ResBlock(in_ch,  out_ch, 3),
                ResBlock(out_ch, out_ch, 2),
                ResBlock(out_ch, out_ch, 1),
                FuseLayer(out_ch, out_ch)
            )

    def forward(self, x):
        x1 = self.block(x)
        return x1


class ResBlock2(nn.Module):
    def __init__(self, in_ch, out_ch, flag):
        super(ResBlock2,self).__init__()
        self.flag = flag
        self.block = nn.Sequential(
            ResBlock(in_ch, out_ch),
            ResBlock(out_ch, out_ch),
            FuseLayer(out_ch, out_ch),
        )
        self.pooling = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=(3, 3, 1),
                      stride=(2, 2, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
        )

    def forward(self, x):
        x1 = self.block(x)
        out = self.pooling(x1)
        if self.flag == 1:
            return x1, out
        else:
            return out


class inconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(inconv,self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(3, 3, 1),
                               stride=(2, 2, 1), padding=(1, 1, 0), output_padding=(1, 1, 0)),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch,  out_classes,flag):
        super(up, self).__init__()

        self.conv1 = nn.Conv3d(in_ch, out_classes, (3, 3, 1), padding=(1, 1, 0))

        if flag == 2:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_classes, (3, 3, 1), padding=(1, 1, 0)),
                inconv(out_classes, out_classes),
            )
        if flag == 4:  # * 4
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_classes, (3, 3, 1), padding=(1, 1, 0)),
                inconv(out_classes, out_classes),
                inconv(out_classes, out_classes),
            )

        if flag==1:
            self.conv = self.conv1

    def forward(self, x):

        x=self.conv(x)
        return x


class WNET(nn.Module):
    def __init__(self, n_channels, out_ch, n_classes):
        super(WNET, self).__init__()
        self.conv = conv(n_channels, out_ch)
        self.block0 = ResBlock2(out_ch, out_ch, 0)
        self.block1 = ResBlock2(out_ch, out_ch, 1)
        self.block2 = ResBlock3(out_ch, out_ch, 1)
        self.block3 = ResBlock3(out_ch, out_ch, 0)

        self.up0 = up(out_ch, n_classes, 2)
        self.up1 = up(out_ch, n_classes*2, 4)
        self.up2 = up(out_ch, n_classes*4, 4)
        self.out = up(7*n_classes, n_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.block0(x)
        x0, x = self.block1(x)
        x0 = self.up0(x0)
        x = self.block2(x)

        x1 =self.up1(x)
        x = self.block3(x)
        x = self.up2(x)
        x = torch.cat([x0, x1, x], dim=1)
        x = self.out(x)
        return F.sigmoid(x)


class ENET(nn.Module):
    def __init__(self, n_channels, out_ch, n_classes):
        super(ENET, self).__init__()

        self.conv = conv(n_channels, out_ch)
        self.block0 = ResBlock2(out_ch, out_ch, 1)  #
        self.block1 = ResBlock2(out_ch, out_ch, 1)
        self.block2 = ResBlock3(out_ch, out_ch, 1)
        self.block3 = ResBlock3(out_ch, out_ch, 0)

        self.up0 = up(out_ch, n_classes, 1)
        self.up1 = up(out_ch, n_classes * 2, 2)
        self.up2 = up(out_ch, n_classes * 2, 2)

        self.out = up(5 * n_classes, n_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        x,_ = self.block0(x)
        x0, x = self.block1(x)
        x0 = self.up0(x0)
        x = self.block2(x)
        x1 = self.up1(x)

        x = self.block3(x)
        x = self.up2(x)

        x = torch.cat([x0, x1, x], dim=1)

        x = self.out(x)
        return F.sigmoid(x)


if __name__ =='__main__':
    x = torch.ones(1, 1, 24, 24, 24)
    print ('test wnet............')
    print ('shape of X ')
    print x.shape

    net = WNET(1, 32, 4)
    if torch.cuda.is_available():
        net = net.cuda()
        x = x.cuda()

    y = net(x)
    print ('shape of Y ')
    print (y.shape)

    print ('test Enet.............')
    print ('shape of X ')
    print x.shape
    net = ENET(1, 32, 4)
    if torch.cuda.is_available():
        net = net.cuda()
        x = x.cuda()

    y = net(x)
    print ('shape of Y ')
    print (y.shape)




