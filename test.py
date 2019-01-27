from model.unet3d import UNet3D
from data_loader.brats15 import Brats15DataLoader
from src.utils import *

from torch.utils.data import DataLoader
from torch.autograd import Variable

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ********** Hyper Parameter **********
data_dir = '/home/haoyum/download/BRATS2015_Training'
conf_test = 'config/test15.conf'
batch_size = 4
save_dir = 'ckpt/'
device_ids = [0, 1, 2, 3]       # multi-GPU
cuda_available = torch.cuda.is_available()

model_epo = []

# build dataset
data = Brats15DataLoader(data_dir=data_dir, task_type='wt', conf=conf_test)
test_dataset = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)


def load_model(model):
    # build model
    net = UNet3D(in_ch=4, out_ch=2)  # multi-mode
    if torch.cuda.is_available():
        net = net.cuda()
        net = nn.DataParallel(net)  # multi-Gpu

    save_path = os.path.join('ckpt/unet3d_' + str(model)+'.pth')
    state_dict = torch.load(save_path)
    net.load_state_dict(state_dict)
    return net


def test():
    for model in model_epo:
        net = load_model(model)
        net.eval()
        dice_all = []
        for step, (images, labels) in enumerate(test_dataset):
            images = Variable(images.cuda() if cuda_available else images)
            # 5D tensor   Batch_Size * 4(modal) * 16(volume_size) * height * weight
            labels = Variable(labels.cuda() if cuda_available else labels)
            # 5D tensor   Batch_Size * 1        * 16(volume_size) * height * weight

            predicts = net(images)
            # 5D tensor   Batch_Size * 2 * 16(volume_size) * height * weight

            predicts = F.softmax(predicts, dim=1)
            # 5D tensor   Batch_Size * 2 * 16(volume_size) * height * weight
            predicts = (predicts[:, 0, :, :, :] > 0.5).long()
            # 4D Long tensor   Batch_Size * 16(volume_size) * height * weight

            d = dice(predicts, labels[:, 0, :, :, :].long())
            dice_all.append(d)

        print 'model epoch' + str(model)
        print sum(dice_all)/(len(dice_all)*1.0)


if __name__ =='__main__':
    test()


