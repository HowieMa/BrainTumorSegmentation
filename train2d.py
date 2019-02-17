from model2d.unet2d import UNet2D
from model2d.multi_unet2d import Multi_Unet

from src.utils import *
from data_loader.brats15_2d import Brats15DataLoader


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
batch_size = 32
epochs = 30
save_dir = 'ckpt_'
device_ids = [0, 1, 2, 3]       # multi-GPU
cuda_available = torch.cuda.is_available()

model = sys.argv[1]
print model

if not os.path.exists(save_dir + model + '/'):
    os.mkdir(save_dir + model + '/')

# ******************** build model ********************
if model == '2dunet':
    net = UNet2D(in_ch=4, out_ch=2, degree=64)  # multi-modal =4, out binary classification one-hot
elif model == 'multi_unet':
    net = Multi_Unet(1, 2, 32)
else:
    exit('wrong model!')


if cuda_available:
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=device_ids)

# ******************** log file ********************
log_train = open(model + '_log_train.txt', 'w')
log_test = open(model + '_log_test.txt', 'w')
log_train_dice = open(model + '_log_train_dice.txt', 'w')
log_test_dice = open(model + '_log_test_dice.txt', 'w')


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

    for epoch in range(1, epochs + 1):
        print ('epoch....................................' + str(epoch))
        train_loss = []
        test_loss = []
        train_dice = []
        test_dice = []
        # *************** train model ***************
        print 'train ....'
        net.train()
        for step, (image, label, index) in enumerate(train_dataset):
            image = Variable(image.cuda() if cuda_available else image)
            # 4D tensor   Batch_Size * 4(modal) * 192 * 192
            label = Variable(label.cuda() if cuda_available else image)
            # 4D tensor   Batch_Size * 1 * 192 * 192

            optimizer.zero_grad()

            predicts = net(image)   # 4D tensor   Batch_Size * 2 * 192 * 192
            loss_train = criterion(predicts, label[:, 0, :, :].long())
            train_loss.append(float(loss_train))
            loss_train.backward()
            optimizer.step()

            # ****** calculate dice ******
            predicts = F.softmax(predicts, dim=1)
            # 4D float Tensor   Batch_Size * 2 * height * weight
            predicts = (predicts[:, 1, :, :] > 0.5).long()
            # 3D Long  Tensor   Batch_Size *     height * weight

            d = dice(predicts, label[:, 0, :, :].long())
            train_dice.append(d)

            # ****** save sample image for each epoch ******
            if step == 0:
                save_train_images(image, predicts, label[:, 0, :, :], index, epoch, save_dir=save_dir + model + '/')

        # ***************** calculate test loss *****************
        print 'test ....'
        net.eval()
        for step, (image, label, subject) in enumerate(test_dataset):
            image = Variable(image.cuda() if cuda_available else image)
            # 4D tensor   Batch_Size * 4(modal) * 192 * 192
            label = Variable(label.cuda() if cuda_available else image)
            # 4D tensor   Batch_Size * 1 * 192 * 192

            predicts = net(image)   # 4D tensor   Batch_Size * 2 * 192 * 192
            loss_test = criterion(predicts, label[:, 0, :, :].long())
            test_loss.append(float(loss_test))

            # ****** calculate dice ******
            predicts = F.softmax(predicts, dim=1)
            # 4D float Tensor   Batch_Size * 2 * height * weight
            predicts = (predicts[:, 1, :, :] > 0.5).long()
            # 3D Long  Tensor   Batch_Size *     height * weight
            d = dice(predicts, label[:, 0, :, :].long())
            test_dice.append(d)

        # **************** save loss for one batch ****************
        print 'train_loss ' + str(sum(train_loss) / (len(train_loss) * 1.0))
        print 'test_loss ' + str(sum(test_loss) / (len(test_loss) * 1.0))
        print 'train_dice ' + str(sum(train_dice) / (len(train_dice) * 1.0))
        print 'test_dice ' + str(sum(test_dice) / (len(test_dice) * 1.0))

        log_train.write(str(sum(train_loss)/(len(train_loss) * 1.0)) + '\n')
        log_train_dice.write(str(sum(train_dice) / (len(train_dice) * 1.0)) + '\n')
        log_test.write(str(sum(test_loss) / (len(test_loss) * 1.0)) + '\n')
        log_test_dice.write(str(sum(test_dice) / (len(test_dice) * 1.0)) + '\n')

        # **************** save model ****************
        if epoch % 10 == 0:
            torch.save(net.state_dict(),
                       os.path.join(save_dir + model + '/', 'epoch_{:d}.pth'.format(epoch)))

    print '***********************************************************'
    print ('done!')


if __name__ == '__main__':
    run()

log_train.close()
log_test.close()
log_test_dice.close()
