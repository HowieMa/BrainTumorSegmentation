from src.utils import *
from backbone import *

from unet3d import *


class StackedUnet3D(nn.Module):
    def __init__(self,in_ch=1, out_ch=2, degree=16):
        super(StackedUnet3D, self).__init__()

        self.unet3d1 = UNet3D(in_ch=in_ch, out_ch=out_ch, degree=degree)
        self.unet3d2 = UNet3D(in_ch=in_ch + out_ch, out_ch=out_ch, degree=degree)
        self.unet3d3 = UNet3D(in_ch=in_ch + out_ch, out_ch=out_ch, degree=degree)
        self.unet3d4 = UNet3D(in_ch=in_ch + out_ch, out_ch=out_ch, degree=degree)

    def forward(self, x):
        """
        :param x: images  5D Tensor  B  *  modal(4)  *  16  *  W  *  H
        :return:
        stacked output    6D Tensor  B  *  modal(4) * output(2)  *  16  *  W  *  H
        """
        Flair = x[:, 0:1, :, :, :]   # Batch Size * 1 * volume_size * height * width
        T1 = x[:, 1:2, :, :, :]
        T1c = x[:, 2:3, :, :, :]
        T2 = x[:, 3:4, :, :, :]

        out1 = self.unet3d1(Flair)
        out2 = self.unet3d2(torch.cat([out1, T1], dim=1))
        out3 = self.unet3d3(torch.cat([out2, T1c], dim=1))
        out4 = self.unet3d4(torch.cat([out3, T2], dim=1))

        return torch.stack([out1, out2, out3, out4], dim=1)


# test case
if __name__ == "__main__":
    net = StackedUnet3D(in_ch=1, out_ch=2, degree=8)
    print"total parameter:" + str(netSize(net))  # 6483800

    x = torch.randn(4, 4, 16, 192, 192)  # batch size = 2
    print ('input data')
    print (x.shape)

    if torch.cuda.is_available():
        net = net.cuda()
        x = x.cuda()

    y = net(x)
    print ('output data')
    print (y.shape)  # (2, 4, 2, 16, 192, 192)



