from model.unet3d import UNet3D
from model.multi_unet import Multi_Unet
from src.utils import *
from data_loader.brats15_v2 import Brats15DataLoader


from torch.utils.data import DataLoader
from torch.autograd import Variable

import sys
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
epochs = 200
save_dir = 'ckpt_'
device_ids = [0, 1, 2, 3]       # multi-GPU
cuda_available = torch.cuda.is_available()

model = sys.argv[1]
print model

if not os.path.exists(save_dir + model + '/'):
    os.mkdir(save_dir + model + '/')

# ******************** build model ********************
if model == '3dunet':
    net = UNet3D(in_ch=4, out_ch=2)  # multi-modal =4, out binary classification one-hot
elif model == 'multi_unet':
    net = Multi_Unet(1, 2, 32)
else:
    exit('wrong model!')


if cuda_available:
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=device_ids)

# ******************** log file ********************
log_train = open(model + 'log_train.txt', 'w')
log_test = open(model + 'log_test.txt', 'w')
log_test_dice = open(model + 'log_test_dice.txt', 'w')


# ******************** data preparation  ********************
print 'train data ....'
train_data = Brats15DataLoader(data_dir=data_dir, task_type='wt', conf=conf_train, is_train=True)
print 'test data .....'
test_data = Brats15DataLoader(data_dir=data_dir, task_type='wt', conf=conf_test, is_train=False)

# data loader
train_dataset = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_dataset = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


def run():
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    best_dice = -1
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        print ('epoch....................................' + str(epoch))
        train_loss = []
        test_loss = []
        train_dice = []
        test_dice = []
        # *************** train model ***************
        print 'train ....'
        net.train()
        for step, (images_vol, labels_vol, subject) in enumerate(train_dataset):
            for i in range(len(images_vol)):        # 144/16 = 9
                images = Variable(images_vol[i].cuda() if cuda_available else images_vol[i])
                # 5D tensor   Batch_Size * 4(modal) * 16 * 192 * 192
                labels = Variable(labels_vol[i].cuda() if cuda_available else labels_vol[i])
                # 5D tensor   Batch_Size * 1        * 16 * 192 * 192

                optimizer.zero_grad()
                predicts = net(images)
                # 5D tensor   Batch_Size * 2 * 16(volume_size) * height * weigh
                loss_train = criterion(predicts, labels[:, 0, :, :, :].long())
                train_loss.append(float(loss_train))
                loss_train.backward()
                optimizer.step()

                predicts = F.softmax(predicts, dim=1)
                # 5D float Tensor   Batch_Size * 2 * 16(volume_size) * height * weight
                predicts = (predicts[:, 1, :, :, :] > 0.5).long()
                # 4D Long  Tensor   Batch_Size * 16(volume_size) * height * weight
                d = dice(predicts, labels[:, 0, :, :, :].long())
                train_dice.append(d)
            # ****** save image of step 0 for each epoch ******
            if step == 0:
                save_train_slice(images, predicts, labels[:, 0, :, :, :], epoch, save_dir=save_dir + model + '/')

        # ***************** calculate test loss *****************
        print 'test ....'
        net.eval()
        for step, (images_vol, labels_vol, subject) in enumerate(test_dataset):
            for i in range(len(images_vol)):  # 144/16 = 9
                images = Variable(images_vol[i].cuda() if cuda_available else images_vol[i])
                # 5D tensor   Batch_Size * 4(modal) * 16 * 192 * 192
                labels = Variable(labels_vol[i].cuda() if cuda_available else labels_vol[i])
                # 5D tensor   Batch_Size * 1        * 16 * 192 * 192

                predicts = net(images)
                # 5D tensor   Batch_Size * 2 * 16(volume_size) * height * weight
                loss_test = criterion(predicts, labels[:, 0, :, :, :].long())
                test_loss.append(float(loss_test))

                predicts = F.softmax(predicts, dim=1)
                # 5D float Tensor   Batch_Size * 2 * 16(volume_size) * height * weight
                predicts = (predicts[:, 1, :, :, :] > 0.5).long()
                # 4D Long  Tensor   Batch_Size * 16(volume_size) * height * weight
                d = dice(predicts, labels[:, 0, :, :, :].long())
                test_dice.append(d)

        # **************** save loss for one batch ****************
        print 'train_loss ' + str(sum(train_loss) / (len(train_loss) * 1.0))
        print 'test_loss ' + str(sum(test_loss) / (len(test_loss) * 1.0))
        print 'train_dice ' + str(sum(train_dice) / (len(train_dice) * 1.0))
        print 'test_dice ' + str(sum(test_dice) / (len(test_dice) * 1.0))

        log_train.write(str(sum(train_loss)/(len(train_loss) * 1.0)) + '\n')
        log_test.write(str(sum(test_loss) / (len(test_loss) * 1.0)) + '\n')
        log_test_dice.write(str(sum(test_dice) / (len(test_dice) * 1.0)) + '\n')

        if sum(test_dice) / (len(test_dice) * 1.0) > best_dice:
            best_dice = sum(test_dice) / (len(test_dice) * 1.0)
            best_epoch = epoch

        # **************** save model ****************
        if epoch % 10 == 0:
            torch.save(net.state_dict(),
                       os.path.join(save_dir + model + '/', 'epoch_{:d}.pth'.format(epoch)))

    print '***********************************************************'
    print 'Best Dice coefficient is '
    print best_dice
    print 'Best epoch is '
    print best_epoch
    print '***********************************************************'
    print ('done!')


if __name__ == '__main__':
    run()

log_train.close()
log_test.close()
log_test_dice.close()
