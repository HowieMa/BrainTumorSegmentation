from model.unet3d import UNet3D
from data_loader.brats15 import Brats15DataLoader
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os

import torch
import torch.nn as nn
import torch.optim as optim


cuda_available = torch.cuda.is_available()
epochs = 2000
save_dir = 'ckpt/'


# multi-GPU
device_ids = [0, 1, 2, 3]


# Hyper Parameter
data_dir = '/home/haoyum/download/BRATS2015_Training'
conf='/home/haoyum/download/BrainTumorSegmentation/config/train15.conf'
learning_rate = 0.001
batch_size = 4

# build dataset
data = Brats15DataLoader(data_dir=data_dir, task_type='wt', conf=conf)
train_dataset = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)


net = UNet3D(in_ch=4, out_ch=2)  # multi-mode

if cuda_available:
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=device_ids)


def train():
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    softMax = nn.Softmax()
    net.train()
    for epoch in range(0, epochs):
        print ('epoch....................................' + str(epoch))
        for step, (images, labels) in enumerate(train_dataset):

            images = Variable(images.cuda() if cuda_available else images)
            # 5D tensor Batch_Size * 4(modal) * 16(volume_size) * height * weight
            labels = Variable(labels.cuda() if cuda_available else labels)
            # 5D tensor Batch_Size * 1        * 16(volume_size) * height * weight
            optimizer.zero_grad()
            predicts = net(images)

            loss = 0
            volume_size = images.shape[2]
            for i in range(volume_size):
                predict = predicts[:, :, i, :, :]
                label = labels[:, 0, i, :, :].long()
                loss += criterion(predict, label)

            print 'step...' + str(step)
            print 'loss...' + str(float(loss))

            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'unet3d_{:d}.pth'.format(epoch)))

    print ('done!')


if __name__ =='__main__':
    train()


