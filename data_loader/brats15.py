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
    def __init__(self, data_dir, direction='axial', volume_size=16,
                 num_class=4, task_type='wt', conf='../config/train15.conf',
                 with_gt=True):
        self.data_dir = data_dir  #
        self.num_class = num_class
        self.img_lists = []

        self.volume_size = volume_size
        self.with_gt = with_gt

        self.task_type = task_type
        self.direction = direction          # 'axial', 'sagittal', or 'coronal'
        self.margin = 5

        train_config = open(conf).readlines()
        for data in train_config:
            self.img_lists.append(os.path.join(self.data_dir, data.strip('\n')))

        print ('**** total number of data is ' + str(len(self.img_lists)))

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, item):
        # ********** get file dir **********
        subject = self.img_lists[item]  # absolute dir
        files = os.listdir(subject)  # [XXX.Flair, XXX.T1, XXX.T1c, XXX.T2, XXX.OT]

        multi_mode_dir = []
        label_dir = ""
        for f in files:
            if f == '.DS_Store':
                continue
            if 'Flair' in f or 'T1' in f or 'T2' in f:
                multi_mode_dir.append(f)
            elif 'OT.' in f:
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
        minbox = [16, 128, 128]
        for i in range(len(multi_mode_imgs)):
            multi_mode_imgs[i] = crop_with_box(multi_mode_imgs[i], bbmin, bbmax, minbox)
            multi_mode_imgs[i] = normalize_one_volume(multi_mode_imgs[i])

        label = crop_with_box(label, bbmin, bbmax, minbox)
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
            multi_mode_imgs[i] = crop_with_box(multi_mode_imgs[i], bbmin, bbmax, minbox)
        volume = np.asarray(multi_mode_imgs)

        label = crop_with_box(label, bbmin, bbmax, minbox)  # 3D label
        label = label[np.newaxis, :, :, :]          # from 3D to 4D label
        # ********** get slice from whole images **********
        volume, label = self.get_slices(volume, label)

        # ********** change data type from numpy to torch.Tensor **********

        volume = torch.from_numpy(volume)  # modal(4) * volume_size(16) * height * weight
        label = torch.from_numpy(label)    # modal(4) * volume_size(16) * height * weight
        return volume.float(), label.float()

    def get_slices(self, img, label):
        """
        get volume randomly
        :param img:
        :param label:
        :return:
        """
        start = np.random.randint(0, img.shape[1] - self.volume_size + 1)

        Wstart = np.random.randint(0, img.shape[2] - 128 + 1)
        Hstart = np.random.randint(0, img.shape[3] - 128 + 1)

        img = img[:, start: start + self.volume_size,
              Wstart:Wstart+128, Hstart:Hstart+128]
        label = label[:, start: start + self.volume_size,
                Wstart:Wstart+128, Hstart:Hstart+128]
        return img, label


# test case
if __name__ =="__main__":
    slice = 10
    data_dir = '../data/train/'
    conf = '../config/sample15.conf'
    print ('**** whole tumor task *************')
    brats15 = Brats15DataLoader(data_dir=data_dir, task_type='wt', conf=conf)
    volume, labels = brats15[0]
    print ('image size ......')
    print (volume.shape)                # (4, 16, 128, 128)

    print ('label size ......')
    print (labels.shape)             # (1, 16, 128, 128)

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

    # print ('\n**** tumor core task *************')
    # brats15 = Brats15DataLoader(data_dir=data_dir, task_type='wt',
    #                             conf=conf, direction='sagittal')
    # volume, labels = brats15[0]
    # print ('image size ......')
    # print (volume.shape)                # (4, 16, 240, 240)
    # print ('label size ......')
    # print (labels.shape)             # (1, 16, 240, 240)
    #
    # print ('get sample of images')
    # for i in range(4):
    #     sample_img = volume[i, slice, :, :]        # size 1 * 240 * 240
    #     sample_img = np.squeeze(np.asarray(sample_img))
    #     print sample_img.shape
    #     scipy.misc.imsave('img/img_%s_tc.jpg' % ddd[i] , sample_img)
    #
    # print ('get sample of labels')
    # sample_label = labels[0, slice, :, :]       # size 1 * 240 * 240
    # sample_label = np.squeeze(np.asarray(sample_label))
    # print sample_label.shape
    # scipy.misc.imsave('img/label_tc.jpg', sample_label)
    #
    #















