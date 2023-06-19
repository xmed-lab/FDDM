import os
import copy
import torch
import shutil
import importlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_config(config_filename):
    config_path = "configs.{}".format(config_filename.split('.')[0])
    module = importlib.import_module(config_path)
    return module.Config()


def splitprint():
    print("#" * 100)


def runid_checker(opts, if_syn=False):
    rootpath = opts.train_collection
    if if_syn:
        rootpath = opts.syn_collection
    valset_name = os.path.split(opts.val_collection)[-1]
    config_filename = opts.model_configs
    run_id = opts.run_id
    target_path = os.path.join(rootpath, "models", valset_name, config_filename, "run_" + str(run_id))
    if os.path.exists(target_path):
        if opts.overwrite:
            shutil.rmtree(target_path)
        else:
            print("'{}' exists!".format(target_path))
            return False
    os.makedirs(target_path)
    print("checkpoints are saved in '{}'".format(target_path))
    return True


def predict_dataloader(model, loader, device, net_name="mm-model", if_test=False):
    model.eval()
    predicts, predicts_fine = [], []
    scores = []
    scores_fine = []
    expects, expects_fine = [], []

    eye_level_predict = defaultdict(list)
    eye_level_expect = defaultdict(list)
    for i, (inputs, labels_onehot, imagenames) in enumerate(loader):
        with torch.no_grad():
            outputs, _ = model(inputs.to(device))
            outputs = torch.nn.Sigmoid()(outputs)
            eye_id = '-'.join(imagenames[0].split('-')[0:2])
            eye_level_predict[eye_id].extend(outputs.cpu().numpy().tolist())
            if eye_id not in eye_level_expect:
                eye_level_expect[eye_id] = labels_onehot.cpu().numpy()

        scores_fine.extend(outputs.cpu().numpy().astype(np.int64).tolist())
        predict_fine = torch.round(outputs).cpu().numpy().astype(np.int64).tolist()
        predicts_fine.extend(predict_fine)
        expects_fine.extend(labels_onehot.cpu().numpy().tolist())

    for eye_id in eye_level_predict:
        predict = np.array([np.max(np.array(eye_level_predict[eye_id])[:, i]) for i in range(11)])
        scores.append(predict)
        predict = np.int64(torch.from_numpy(predict).squeeze(0).cpu().numpy() >= 0.5).tolist()
        predicts.append(predict)
        expects.extend(eye_level_expect[eye_id])
    return predicts, scores, expects, predicts_fine, scores_fine, expects_fine
