from model.unet3d import UNet3D

from data_loader.brats15 import Brats15DataLoader
from torch.utils.data import DataLoader
from torch.autograd import Variable

import configparser
import os

import torch
import torch.nn as nn
import torch.optim as optim


cuda_available = torch.cuda.is_available()
epochs = 20
save_dir = 'ckpt/'
device_ids = [0]


# build dataset
data = Brats15DataLoader(data_dir='data/train/')
train_dataset = DataLoader(dataset=data, batch_size=1, shuffle=True)


net = UNet3D(in_ch=4, out_ch=1)  # multi-mode

if cuda_available:
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=device_ids)


def train():
    optimizer = optim.Adam(params=net.parameters(), lr=0.001, betas=(0.9, 0.999))

    criterion = nn.BCELoss(size_average=True)

    net.train()

    for epoch in range(0, epochs):
        print ('epoch....................................' + str(epoch))
        for step, (images, labels) in enumerate(train_dataset):
            images = Variable(images.cuda() if cuda_available else images)
            labels = Variable(labels.cuda() if cuda_available else labels)

            optimizer.zero_grad()
            predicts = net(images)

            loss = criterion(predicts, labels)
            print ('step... %f  loss... %f ' % (step, float(loss)) )

            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), os.path.join(save_dir, 'unet3d_{:d}.pth'.format(epoch)))

    print ('done!')


if __name__ =='__main__':
    train()


