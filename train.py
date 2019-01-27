from model.unet3d import UNet3D
from model.multi_unet import IVD_Net_asym
from src.utils import *
from data_loader.brats15 import Brats15DataLoader


from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ********** Hyper Parameter **********
data_dir = '/home/haoyum/download/BRATS2015_Training'
conf_train = 'config/train15.conf'
conf_test = 'config/test15.conf'
learning_rate = 0.001
batch_size = 4
epochs = 2000
save_dir = 'ckpt/'
device_ids = [0, 1, 2, 3]       # multi-GPU
cuda_available = torch.cuda.is_available()

log_train = open('log_train.txt', 'w')
log_test = open('log_test.txt', 'w')


# ******************** data preparation  ********************
print 'train data ....'
train_data = Brats15DataLoader(data_dir=data_dir, task_type='wt', conf=conf_train)
print 'test data .....'
test_data = Brats15DataLoader(data_dir=data_dir, task_type='wt', conf=conf_test)

# data loader
train_dataset = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_dataset = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


# ******************** build model ********************
net = UNet3D(in_ch=4, out_ch=2)  # multi-modal =4, out binary classification one-hot

# init model weight
net.apply(weights_init)

if cuda_available:
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=device_ids)


def run():
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        print ('epoch....................................' + str(epoch))
        train_loss = []
        test_loss = []
        # *************** train model ***************
        net.train()
        for step, (images, labels) in enumerate(train_dataset):
            images = Variable(images.cuda() if cuda_available else images)
            # 5D tensor   Batch_Size * 4(modal) * 16(volume_size) * height * weight
            labels = Variable(labels.cuda() if cuda_available else labels)
            # 5D tensor   Batch_Size * 1        * 16(volume_size) * height * weight
            optimizer.zero_grad()
            predicts = net(images)
            # 5D tensor   Batch_Size * 2 * 16(volume_size) * height * weigh
            loss_train = criterion(predicts, labels[:, 0, :, :, :].long())
            train_loss.append(float(loss_train))
            loss_train.backward()
            optimizer.step()

            # ****** save image of step 0 for each epoch ******
            if step == 0:
                save_train_slice(images, predicts, labels, epoch)

        # ***************** calculate test loss *****************
        net.eval()
        for step, (images, labels) in enumerate(test_dataset):
            images = Variable(images.cuda() if cuda_available else images)
            # 5D tensor   Batch_Size * 4(modal) * 16(volume_size) * height * weight
            labels = Variable(labels.cuda() if cuda_available else labels)
            # 5D tensor   Batch_Size * 1        * 16(volume_size) * height * weight

            predicts = net(images)
            # 5D tensor   Batch_Size * 2 * 16(volume_size) * height * weight
            loss_test = criterion(predicts, labels[:, 0, :, :, :].long())
            test_loss.append(float(loss_test))

            predicts = F.softmax(predicts, dim=1)
            # 5D tensor   Batch_Size * 2 * 16(volume_size) * height * weight
            predicts = (predicts[:, 0, :, :, :] > 0.5).long()
            # 4D Long tensor   Batch_Size * 16(volume_size) * height * weight

            dice(predicts, labels[:, 0, :, :, :].long())

        # **************** save loss for one batch ****************
        log_train.write(str(sum(train_loss)/(len(train_loss) * 1.0)) + '\n')
        log_test.write(str(sum(test_loss) / (len(test_loss) * 1.0)) + '\n')

        # **************** save model ****************
        if epoch % 50 == 0:
            torch.save(net.state_dict(),
                       os.path.join(save_dir, 'unet3d_{:d}.pth'.format(epoch)))

    print ('done!')


if __name__ == '__main__':
    run()


