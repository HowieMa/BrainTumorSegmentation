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

ddd = ['flair', 't1', 't1c', 't2']


class Brats15DataLoader(Dataset):
    def __init__(self, data_dir, is_train, direction='axial', task_type='wt',
                 conf='../config/train15.conf'):
        self.data_dir = data_dir  #
        self.img_lists = []
        self.volume_size = 16
        self.data_box = [144, 192, 192]      # max 145
        self.margin = 0
        self.is_train = is_train        # True or False

        self.task_type = task_type    # whole tumor, tumor core,
        self.direction = direction    # 'axial', 'sagittal', or 'coronal'

        train_config = open(conf).readlines()
        for data in train_config:
            self.img_lists.append(os.path.join(self.data_dir, data.strip('\n')))

        print('~' * 50)
        print ('******** Loading data from disk ********')
        self.data = []

        for subject in self.img_lists:
            volume, label = self.get_subject(subject)   # 4 * 144 * 192 * 192
            length = volume.shape[1]                    # 144
            for i in range(length):
                name = subject + '=' + str(i)
                self.data.append([volume[:, i, :, :], label[:, i, :, :], name])

        print ('********  Finish loading data  ********')
        print ('********  Total number of 2D images is ' + str(len(self.data)))
        print('~' * 50)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # ********** get file dir **********
        [image2d, label2d,name] = self.data[index]  # get whole data for one subject

        # ********** change data type from numpy to torch.Tensor **********
        image2d = torch.from_numpy(image2d).float()  # Float Tensor 4 * 144 * 192 * 192
        label2d = torch.from_numpy(label2d).float()    # Float Tensor 4 * 144 * 192 * 192

        return image2d, label2d, name

    def get_subject(self, subject):
        """
        get
        :param subject: absolute dir
        :return:
        volume  4D numpy    4 * 144 * 192 * 192
        label   4D numpy    4 * 144 * 192 * 192
        """
        # **************** get file ****************
        files = os.listdir(subject)  # [XXX.Flair, XXX.T1, XXX.T1c, XXX.T2, XXX.OT]
        multi_mode_dir = []
        label_dir = ""
        for f in files:
            if f == '.DS_Store':
                continue
            # if is data
            if 'Flair' in f or 'T1' in f or 'T2' in f:
                multi_mode_dir.append(f)
            elif 'OT.' in f:        # if is label
                label_dir = f

        bbmin = [0, 0, 0]  # default bounding box
        bbmax = [155 - 1, 240 - 1, 240 - 1]
        # ********** load 4 mode images **********
        multi_mode_imgs = []  # list size :4      item size: 150 * 240 * 240
        for mod_dir in multi_mode_dir:
            path = os.path.join(subject, mod_dir)  # absolute directory
            img = load_mha_as_array(path + '/' + mod_dir + '.mha')
            multi_mode_imgs.append(img)
            if 'Flair.' in mod_dir:  # get non zero bounding box based on Flair image
                bbmin, bbmax = get_ND_bounding_box(img, self.margin)

        # ********** get label **********
        label_dir = os.path.join(subject, label_dir) + '/' + label_dir + '.mha'
        label = load_mha_as_array(label_dir)  #

        # *********** image pre-processing *************
        # step1 ****** resize images and labels to 160 * 192 * 192 *******
        for i in range(len(multi_mode_imgs)):
            multi_mode_imgs[i] = crop_with_box(multi_mode_imgs[i], bbmin, bbmax, self.data_box)
            multi_mode_imgs[i] = normalize_one_volume(multi_mode_imgs[i])
        label = crop_with_box(label, bbmin, bbmax, self.data_box)   # (144, 192, 192)

        # step2 ********* transfer to different direction *********
        multi_mode_imgs = transpose_volumes(multi_mode_imgs, self.direction)  # list
        label = transpose_volumes([label], self.direction)[0]

        if self.direction == 'sagittal' or self.direction == 'coronal':
            self.data_box = [192, 144, 192]

        # step3 ********** get bounding box based on task **********
        if self.task_type == 'wt':
            label = get_whole_tumor_labels(label)
            # for whole tumor task, bouding box is self
            bbmin = [0, 0, 0]
            bbmax = [label.shape[0], label.shape[1], label.shape[2]]

        elif self.task_type == 'tc':
            # for tumor core task, bounding box is the whole tumor box
            no_zero_label = get_whole_tumor_labels(label)
            bbmin, bbmax = get_ND_bounding_box(no_zero_label, self.margin)
            label = get_tumor_core_labels(label)

        # ********** crop image and label based on bounding box **********
        for i in range(len(multi_mode_imgs)):
            multi_mode_imgs[i] = crop_with_box(multi_mode_imgs[i], bbmin, bbmax, self.data_box)
        volume = np.asarray(multi_mode_imgs)

        label = crop_with_box(label, bbmin, bbmax, self.data_box)  # 3D label
        label = label[np.newaxis, :, :, :]  # from 3D to 4D label

        return volume, label


# test case
if __name__ =="__main__":
    vol_num = 4
    data_dir = '../data_sample/'
    conf = '../config/sample15.conf'
    print ('**** whole tumor task *************')
    brats15 = Brats15DataLoader(data_dir=data_dir, task_type='wt',
                                direction='coronal',
                                conf=conf, is_train=False)
    image2d, label2d, index = brats15[60]

    print ('image size ......')
    print (image2d.shape)             # (4,  192, 192)

    print ('label size ......')
    print (label2d.shape)             # (1,  192, 192)

    print index

    print ('get sample of images')
    for i in range(4):
        scipy.misc.imsave('img2d/img_%s_wt.jpg' % ddd[i], image2d[i, :, :])

    print ('get sample of labels')
    scipy.misc.imsave('img2d/label_wt.jpg', label2d[0, :, :]) # 192 * 192


















