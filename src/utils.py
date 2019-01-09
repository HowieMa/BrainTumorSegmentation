# coding:utf-8

import SimpleITK as sitk
import numpy as np


def load_mha_as_array(img_name):
    """
    get the numpy array of brain mha image
    :param img_name: absolute directory of 3D mha images
    :return:
        nda  type: numpy    size: 150 * 240 * 240
    """
    img = sitk.ReadImage(img_name)
    nda = sitk.GetArrayFromImage(img)
    return nda


def get_ND_bounding_box(volume, margin):
    """
    找出输入原始三维图片非零区域的边界
    :param volume:  type:np.array      size: 150 * 240 * 240
    :param margin:  type int           预留边界
    :return:
    idx_min         type: list          [minx, miny, minz]
    idx_max         type: list          [maxx, maxy, maxz]
    """
    input_shape = volume.shape  # volume 150 * 240 * 240
    if (type(margin) is int):
        margin = [margin] * len(input_shape)

    indxes = np.nonzero(volume)  # 每一维度中非0 的数据的位置
    idx_min = []    # type list  [minx, miny, minz]
    idx_max = []    # type list  [maxx, maxy, maxz]
    for i in range(len(input_shape)):  # i = 0, 1, 2
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())
    # resize bounding box with margin
    for i in range(len(input_shape)):  # i = 0, 1, 2
        idx_min[i] = max(idx_min[i] - margin[i], 0)   # 考虑预留边界
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)

    return idx_min, idx_max


def crop_with_box(volume, min_idx, max_idx, MinBox):
    """
    crop image with bounding box
    :param volume:      type: 3D numpy.array
    :param min_idx:     type: list          [minx, miny, minz]
    :param max_idx:     type: list          [maxx, maxy, maxz]
    :return:
    output  cropped volume
    """
    # ensure we have at least a bounding box of size 16 * 128 * 128
    for i in range(3):
        if max_idx[i] - min_idx[i] < MinBox[i]:
            mid = (max_idx[i] + min_idx[i]) / 2.0
            min_idx[i] = mid - MinBox[i]/2
            max_idx[i] = mid + MinBox[i]/2

    output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                           range(min_idx[1], max_idx[1] + 1),
                           range(min_idx[2], max_idx[2] + 1))]
    return output


def get_whole_tumor_labels(label):
    """
    whole tumor in patient data is label 1 + 2 + 3 + 4
    :param label:  numpy array      size : 155 * 240 * 240  value 0-4
    :return:
    label 1 * 155 * 240 * 240
    """
    label = (label > 0) + 0  # label 1,2,3,4
    return label


def get_tumor_core_labels(label):
    """
    tumor core in patient data is label 1 + 3 + 4
    :param label:  numpy array      size : 155 * 240 * 240  value 0-4
    :return:
    label 155 * 240 * 240
    """

    label = (label == 1) + (label == 3) + (label == 4) + 0  # label 1,3,4 = 1
    return label


def transpose_volumes(volumes, slice_direction):
    """
    transpose a list of volumes
    inputs:
        volumes: a list of nd volumes
        slice_direction: 'axial', 'sagittal', or 'coronal'
    outputs:
        tr_volumes: a list of transposed volumes
    """
    if (slice_direction == 'axial'):
        tr_volumes = volumes

    elif(slice_direction == 'sagittal'):
        tr_volumes = [np.transpose(x, (2, 0, 1)) for x in volumes]

    elif(slice_direction == 'coronal'):
        tr_volumes = [np.transpose(x, (1, 0, 2)) for x in volumes]

    else:
        print('undefined slice direction:', slice_direction)
        tr_volumes = volumes
    return tr_volumes


def normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]  # ignore background
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std

    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out


def oneHotLabel(label):
    """
    change 3D label to 4D one hot label
    :param label: 3D numpy
    :return: 4D numpy
    """
    background = 1 - label
    return np.stack((label, background))
