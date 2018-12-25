# coding:utf-8
"""
对 whole tumor 任务
基于原始 150 * 240 * 240 图，抠出非零区域.
据此寻找label 是 1,2,3,4的部分

对 tumor core 任务
从原始图中抠出 label 是1，2，3，4 的区域
寻找label 是 1,3,4的部分

对 enhanced tumor 任务
从原始图中抠出 label 是1，3，4 的区域
寻找label 是 4 的部分

对Flair 数据，范围是 R+; T1 范围是 R; T1C 范围是 R+; T2 范围是 R+;

"""


import os
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.misc


from src.utils import *
ddd = ['flair','t1','t1c','t2']

class Brats15DataLoader(Dataset):
    def __init__(self, data_dir, direction='axial', volume_size=16,
                 num_class=4, task_type='wt', with_gt=True):
        self.data_dir = data_dir  #
        self.num_class = num_class
        self.img_lists = []

        self.volume_size = volume_size
        self.with_gt = with_gt

        self.task_type = task_type
        self.direction = direction
        self.margin = 5

        subjects = os.listdir(self.data_dir)        # [brats_2013_pat0001_1, ...]
        for sub in subjects:
            if sub == '.DS_Store':
                continue
            self.img_lists.append(os.path.join(self.data_dir, sub))
        print ('**** total number of data is ' + str(len(self.img_lists)))

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, item):
        # ********** get file dir **********
        subject = self.img_lists[item]
        files = os.listdir(subject)  # [XXX.Flair, XXX.T1, XXX.T1c, XXX.T2, XXX.OT]

        multi_mode_dir = []
        label_dir = ""
        for f in files:
            if f == '.DS_Store':
                continue
            if 'O.OT.' not in f:
                multi_mode_dir.append(f)
            else:
                label_dir = f

        bbmin = [0, 0, 0]                           # default bounding box
        bbmax = [155 - 1, 240 - 1, 240 - 1]
        # ********** get 4 mode images **********
        multi_mode_imgs = []    # list size :4      item size: 150 * 240 * 240
        for mod_dir in multi_mode_dir:
            path = os.path.join(subject, mod_dir)  # absolute directory
            img = load_mha_as_array(path + '/' + mod_dir + '.mha')
            multi_mode_imgs.append(img)

            if 'Flair.' in mod_dir:  # get non zero bounding box based on Flair image
                bbmin, bbmax = get_ND_bounding_box(img, self.margin)

        # ********** get label **********
        label_dir = os.path.join(subject, label_dir) + '/' + label_dir + '.mha'
        label = load_mha_as_array(label_dir)    #

        # *********** image pre-processing *************
        # step1 ********* crop none-zero images and labels *********
        for i in range(len(multi_mode_imgs)):
            multi_mode_imgs[i] = crop_with_box(multi_mode_imgs[i], bbmin, bbmax)
            print str(i) + '....'
            print multi_mode_imgs[i].max()
            print multi_mode_imgs[i].min()
            print multi_mode_imgs[i].mean()
            print 'norm'
            multi_mode_imgs[i] = normalize_one_volume(multi_mode_imgs[i])
            print multi_mode_imgs[i].max()
            print multi_mode_imgs[i].min()
            print multi_mode_imgs[i][multi_mode_imgs[i]!=0].mean()

        label = crop_with_box(label, bbmin, bbmax)


        # step2 ********* transfer images to different direction *********
        multi_mode_imgs = transpose_volumes(multi_mode_imgs, self.direction)
        label = transpose_volumes([label], self.direction)[0]

        # step3 ********** get bounding box based on task **********
        if self.task_type == 'wt':
            label = get_whole_tumor_labels(label)
            # for whole tumor task,bouding box is self
            bbmin = [0, 0, 0]
            bbmax = [label.shape[0]-1, label.shape[1]-1, label.shape[2]-1]

        if self.task_type == 'tc':
            no_zero_label = get_whole_tumor_labels(label)
            bbmin, bbmax = get_ND_bounding_box(no_zero_label, self.margin)
            label = get_tumor_core_labels(label)

        # ********** crop image and label based on bounding box **********
        for i in range(len(multi_mode_imgs)):
            multi_mode_imgs[i] = crop_with_box(multi_mode_imgs[i], bbmin, bbmax)
        volume = np.asarray(multi_mode_imgs)

        label = crop_with_box(label, bbmin, bbmax)
        label = label[np.newaxis, :, :, :]          # from 3D to 4D

        # ********** change data type from numpy to torch.Tensor **********
        volume = torch.from_numpy(volume)
        label = torch.from_numpy(label)
        return volume.float(), label.float()

    def get_slices(self, img, label):
        """
        get volume randomly
        :param img:
        :param label:
        :return:
        """
        start = np.random.randint(0, img.shape[0] - self.volume_size + 1)
        img = img[start: start + self.volume_size, :, :]
        label = label[start: start + self.volume_size, :, :]
        return img, label


# test case
if __name__ =="__main__":
    slice = 60
    data_dir = '../data/train/'
    print ('**** whole tumor task *************')
    brats15 = Brats15DataLoader(data_dir=data_dir,task_type='wt')
    volume, labels = brats15[0]
    print ('image size ......')
    print (volume.shape)                # (4, 16, 240, 240)

    print ('label size ......')
    print (labels.shape)             # (1, 16, 240, 240)

    print ('get sample of images')
    for i in range(4):
        sample_img = volume[i, slice, :, :]        # size 1 * 240 * 240
        sample_img = np.squeeze(np.asarray(sample_img))
        print sample_img.shape
        scipy.misc.imsave('img/img_%s_wt.jpg' % ddd[i] , sample_img)

    print ('get sample of labels')
    sample_label = labels[0, slice, :, :]       # size 1 * 240 * 240
    sample_label = np.squeeze(np.asarray(sample_label))
    print sample_label.shape
    scipy.misc.imsave('img/label_wt.jpg', sample_label)

    print ('\n**** tumor core task *************')
    brats15 = Brats15DataLoader(data_dir=data_dir, task_type='wt', direction='sagittal')
    volume, labels = brats15[0]
    print ('image size ......')
    print (volume.shape)                # (4, 16, 240, 240)
    print ('label size ......')
    print (labels.shape)             # (1, 16, 240, 240)

    print ('get sample of images')
    sample_img = volume[0, slice, :, :]        # size 1 * 240 * 240
    sample_img = np.squeeze(np.asarray(sample_img))
    print sample_img.shape
    scipy.misc.imsave('img/img_tc.jpg', sample_img)

    print ('get sample of labels')
    sample_label = labels[0, slice, :, :]       # size 1 * 240 * 240
    sample_label = np.squeeze(np.asarray(sample_label))
    print sample_label.shape
    scipy.misc.imsave('img/label_tc.jpg', sample_label)

















