import os
import numpy as np
import torch
import cv2 as cv
from . import augmentation

def load_image(img_path, configs):
    raw_img = cv.imread(img_path)
    img = (raw_img / 255.).astype(np.float32)
    # pre-process
    myAug = augmentation.OurAug(configs.aug_params["onlyresize"])
    img = myAug.process(img)
    img = configs.transform(img)
    img = torch.unsqueeze(img, dim=0)
    return raw_img, img


# modality is cfp, oct, or mm
def read_imset(dataset, modality, data_path='VisualSearch'):
    imset_file = os.path.join(data_path, dataset, 'ImageSets', '%s.txt' % modality)
    imset = list(map(str.strip, open(imset_file).readlines()))
    return imset


def get_impath(img_id, preprocess=None, data_path='VisualSearch'):
    if preprocess is None: # use default
        preprocess = 'cfp-clahe-448x448' if img_id.startswith('f') else 'oct-median3x3-448x448'  
    return os.path.join(data_path, 'mmc-amd', 'ImageData', preprocess, '%s.jpg' % img_id)
