# coding: utf-8
import torch
import numpy as np
import torch.nn as nn
from . import resnet
import random
import math
import torch.nn.functional as F

class CombineNet_linear_concatenate(nn.Module):
    def __init__(
            self, model1, model2, fc_ins, num_classes=4, heatmap=False):

        super(CombineNet_linear_concatenate, self).__init__()

        self.heatmap = heatmap
        self.model1 = model1
        self.model2 = model2
        self.fc_cat = nn.Linear(fc_ins // 2, num_classes)
        self.gap = nn.AvgPool2d(14, stride=1)

        self.conv1 = nn.Conv2d(fc_ins, fc_ins, kernel_size=3, stride=1, padding=1, groups=fc_ins, bias=False)
        self.bn1 = nn.BatchNorm2d(fc_ins)
        self.conv1_1 = nn.Conv2d(fc_ins, fc_ins // 2, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(fc_ins // 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def predict_per_instance(self, inputs):
        with torch.no_grad():
            output, fm_f, fm_o = self.forward(inputs[0], inputs[1])
        output = np.squeeze(torch.softmax(output, dim=1).cpu().numpy())
        pred = np.argmax(output)
        score = np.max(output)
        return pred, score

    def forward(self, x1, x2):
        x1, feat_x1, map_x1 = self.model1(x1)
        x2, feat_x2, map_x2 = self.model2(x2)
        # x = torch.cat([feat_x1, feat_x2], 1)
        map_x = torch.cat([map_x1, map_x2], 1)

        x = F.relu(self.bn1((self.conv1(map_x))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.gap(x)

        x = x.view(x.size(0), -1)
        x = self.fc_cat(x)
        return x, x1, x2


class MLP(nn.Module):
    def __init__(self, fc_ins=1000):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(fc_ins, fc_ins)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(fc_ins, fc_ins)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(fc_ins, fc_ins)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

