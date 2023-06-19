# coding: utf-8
import os
import random

import cv2 as cv
import numpy as np
import torch

from . import augmentation
from torch.utils.data import Dataset
from .utils import multi2binary
from collections import defaultdict


class BaseDataset(Dataset):
    def __init__(self, aug_params=None, transform=None, if_test=False, cls_num=4):

        self.aug_params = aug_params
        self.transform = transform
        self.if_test = if_test
        self.cls_num = cls_num

    def label_statistic(self):
        cls_count = np.zeros(self.cls_num).astype(np.int64)
        binary_count = np.zeros(2 ** self.cls_num).astype(np.int64)
        for label in self.labels_list:
            for i in range(len(label)):
                cls_count[i] += label[i]
            binary_label = multi2binary(label, self.cls_num)
            binary_count[binary_label] += 1

        for i in range(self.cls_num):
            print("Class {}: {}".format(str(i), cls_count[i]))
        print("Summary: {}".format(np.sum(cls_count)))
        return cls_count, binary_count

    def label_weights_for_balance(self, C=100.0):
        cls_count, binary_count = self.label_statistic()
 
        labels_weight_list = []
        for label in self.labels_list:
            num_label, num_occur = 0.0, 0.0
            for i in range(len(label)):
                num_occur += cls_count[i] * label[i]
                num_label += label[i]
            if num_label == 0:
                weight = C / binary_count[0]
            else:
                weight = C / (num_occur / num_label)
            # binary_label = multi2binary(label, self.cls_num)
            # weight = C / float(binary_count[binary_label])
            labels_weight_list.append(weight)
            # labels_weight_list.append(C / float(cls_count[label]))
        return labels_weight_list


class MultiDataset(BaseDataset):
    """Multi-modal Dataset"""
    def __init__(
        self, img_f_path_list, img_o_path_list, labels_f_list=None, labels_o_list=None,
        aug_params=None, transform=None, if_test=False, cls_num=4, if_eval=False, queue_size=1):
        super(MultiDataset, self).__init__(aug_params, transform, if_test, cls_num)
        self.cls_num = cls_num
        self.queue_size = queue_size

        self.img_f_path_list = img_f_path_list
        self.img_o_path_list = img_o_path_list
        if not self.if_test:
            self.labels_f_list = labels_f_list
            self.labels_list = labels_o_list

        def multi2binary(label_onehot, num_classes):
            root = 2 ** (num_classes - 1)
            result = 0
            for i in range(num_classes):
                result += label_onehot[i] * root
                root /= 2
            return result

        self.class_list = defaultdict(list)
        for idx in range(len(self.labels_f_list)):
            class_id = multi2binary(self.labels_f_list[idx], cls_num)
            self.class_list[class_id].append(img_f_path_list[idx])

        self.class_list_o = defaultdict(list)
        for idx in range(len(self.labels_list)):
            class_id = multi2binary(self.labels_list[idx], cls_num)
            self.class_list_o[class_id].append(img_o_path_list[idx])

        # pair
        self.eye_list = defaultdict(list)
        for img_f_path in self.img_f_path_list:
            img_f_filename = os.path.split(img_f_path)[-1]
            eye_id = '-'.join(img_f_filename.split('-')[0:2])
            self.eye_list[eye_id].append(img_f_path)

        self.if_eval = if_eval

    def __getitem__(self, index):
        img_o_path = self.img_o_path_list[index]
        img_o_filename = os.path.split(img_o_path)[-1]
        img_o = (cv.imread(img_o_path) / 255.).astype(np.float32)

        label_onehot = np.array(self.labels_list[index])
        if not self.if_eval:
            if multi2binary(label_onehot, self.cls_num) in self.class_list:
                img_f_path = random.choice(self.class_list[multi2binary(label_onehot, self.cls_num)])
                label_onehot_f = label_onehot
            else:
                idx_r = random.randint(0, len(self.img_f_path_list) - 1)
                img_f_path = self.img_f_path_list[idx_r]
                label_onehot_f = self.labels_f_list[idx_r]

            img_o_path_add = random.choice(self.class_list_o[multi2binary(label_onehot, self.cls_num)])

            label_onehot_f = np.array(label_onehot_f)
            img_f = (cv.imread(img_f_path) / 255.).astype(np.float32)
            img_o_add = (cv.imread(img_o_path_add) / 255.).astype(np.float32)

            if self.aug_params:
                aug = augmentation.OurAug(self.aug_params)
                img_f = aug.process(img_f)
                img_o_add = aug.process(img_o_add)

            if self.transform:
                img_f = self.transform(img_f)
                img_o_add = self.transform(img_o_add)

        if self.aug_params:
            aug = augmentation.OurAug(self.aug_params)
            img_o = aug.process(img_o)

        if self.transform:
            img_o = self.transform(img_o)

        if not self.if_eval:
            return (img_f, img_o, img_o_add), (label_onehot, label_onehot_f), img_o_filename
        else:
            return img_o, label_onehot, img_o_filename


    def __len__(self):
        return len(self.img_o_path_list)


class SingleDataset(BaseDataset):
    def __init__(
            self, imgs_path_list, labels_list=None,
            aug_params=None, transform=None, if_test=False, cls_num=4):
        super(SingleDataset, self).__init__(aug_params, transform, if_test, cls_num)
        self.imgs_path_list = imgs_path_list
        if not self.if_test:
            self.labels_list = labels_list

    def __getitem__(self, index):

        img_path = self.imgs_path_list[index]
        img_filename = os.path.split(img_path)[-1]
        img = (cv.imread(img_path) / 255.).astype(np.float32)

        if self.aug_params is not None:
            myAug = augmentation.OurAug(self.aug_params)
            img = myAug.process(img)
        if self.transform:
            img = self.transform(img)

        if not self.if_test:
            label_onehot = np.array(self.labels_list[index])
        else:
            label_onehot = -1

        return img, label_onehot, img_filename

    def __len__(self):
        return len(self.imgs_path_list)