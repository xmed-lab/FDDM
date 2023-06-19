import os
import sys
import copy
import time
import torch
import torch.nn as nn
import argparse
import importlib
import numpy as np
from models import save_model, load_single_stream_model
from data.DataLoaders import MultiDataLoader, FundusDataLoader, OctDataLoader
from utils import AverageMeter, load_config, splitprint, runid_checker, predict_dataloader
import random
from metrics import multilabel_confusion_matrix, accuracy_score, sen_score, spe_score, f1_score
from metrics import confusion_matrix as cfm
from sklearn.metrics import roc_curve, auc, average_precision_score


label2disease = ['NOR', 'AMD', 'WAMD', 'DR', 'CSC', 'PED', 'MEM', 'FLD', 'EXU', 'CNV', 'RVO']


def parse_args():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--train_collection", type=str,
                        default='image_data/topcon-mm/train',
                        help="train collection path")
    parser.add_argument("--val_collection", type=str,
                        default='image_data/topcon-mm/val',
                        help="val collection path")
    parser.add_argument("--test_collection", type=str,
                        default='image_data/topcon-mm/test',
                        help="test collection path")
    parser.add_argument("--print_freq", default=20, type=int, help="print frequent (default: 20)")
    parser.add_argument("--model_configs", type=str, default='config_fundus.py',
                        help="filename of the model configuration file.")
    parser.add_argument("--run_id", default=0, type=int, help="run_id (default: 0)")
    parser.add_argument("--device", default=0, type=str, help="cuda:n or cpu (default: 0)")
    parser.add_argument("--num_workers", default=0, type=int, help="number of threads for sampling. (default: 0)")
    parser.add_argument("--checkpoint", default=None, type=str, help="checkpoint path")
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    args = parser.parse_args()
    return args


def validate(model, val_loader, selected_metric, device, cls_num, net_name="mm-model", verbose=True):
    if verbose:
        print("-" * 45 + "validation" + "-" * 45)
    predicts, scores, expects, eye_level_expect, _ = predict_dataloader(model, val_loader, device, net_name,
                                                                        if_test=False)
    predicts = np.array(predicts)
    expects = np.array(expects)
    scores = np.array(scores)

    results = {'overall': {}}
    for lb in label2disease:
        results[lb] = {}
    confusion_matrix = multilabel_confusion_matrix(expects, predicts)
    results['overall']['cm'] = confusion_matrix

    for i in range(cls_num):
        results[label2disease[i]]['spe'] = spe_score(confusion_matrix[i])
        results[label2disease[i]]['sen'] = sen_score(confusion_matrix[i])
        results[label2disease[i]]['acc'] = accuracy_score(confusion_matrix[i])
        results[label2disease[i]]['f1_score'] = f1_score(results[label2disease[i]]['spe'],
                                                         results[label2disease[i]]['sen'])

        predicts_specific = scores[:, i].tolist()
        expects_specific = expects[:, i].tolist()
        fpr, tpr, th = roc_curve(expects_specific, predicts_specific, pos_label=1)
        auc_specific = auc(fpr, tpr)
        results[label2disease[i]]['auc'] = auc_specific
        results[label2disease[i]]['ap'] = average_precision_score(expects_specific, predicts_specific)

    results["overall"]["sen"] = np.average([results[cls_name]["sen"] for cls_name in label2disease])
    results["overall"]["spe"] = np.average([results[cls_name]["spe"] for cls_name in label2disease])
    results["overall"]["f1_score"] = np.average([results[cls_name]["f1_score"] for cls_name in label2disease])
    results["overall"]["auc"] = np.average([results[cls_name]["auc"] for cls_name in label2disease])
    results["overall"]["map"] = np.average([results[cls_name]["ap"] for cls_name in label2disease])
    results["overall"]["acc"] = np.average([results[cls_name]["acc"] for cls_name in label2disease])

    print("cls\tsen\tspe\tf1\tauc\tmap\tacc")
    for lbl in label2disease:
        print("{cls}\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\t{auc:.4f}\t{ap:.4f}\t{acc:.4f}\t".format(cls=lbl,
                                                                                    sen=results[lbl]['sen'],
                                                                                    spe=results[lbl]['spe'],
                                                                                    f1=results[lbl]['f1_score'],
                                                                                    auc=results[lbl]['auc'],
                                                                                    ap=results[lbl]['ap'],acc=results[lbl]['acc']))
    print("overall\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\t{auc:.4f}\t"
          "{map:.4f}\t{acc:.4f}\n".format(
              sen=results["overall"]["sen"],
              spe=results["overall"]["spe"],
              f1=results["overall"]["f1_score"],
              auc=results["overall"]["auc"],
              map=results["overall"]["map"],
              acc=results["overall"]["acc"]),
          "confusion matrix:\n {}".format(results["overall"]["cm"]))
    return results["overall"]["map"], eye_level_expect


def adjust_learning_rate(optimizer, optim_params):
    optim_params['lr'] *= 0.5
    print('learning rate:', optim_params['lr'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']
    if optim_params['lr'] < optim_params['lr_min']:
        return True
    else:
        return False


def model_structure(net_name):
    print("load model {}".format(net_name))
    if net_name == 'cfp-model' or net_name == 'oct-model':
        return load_single_stream_model
    else:
        print("{} is not support.").format(net_name)
        return None


def select_dataloader(modality):
    print("initialize dataloader for {}".format(modality))
    if modality == "cfp":
        return FundusDataLoader
    elif modality == "oct":
        return OctDataLoader
    else:
        print("{} is not support.").format(modality)
        return None


def main(opts):
    # load model configs

    configs = load_config(opts.model_configs)

    # check that the save path is available
    if not runid_checker(opts, configs.if_syn):
        return
    splitprint()
    # cuda number
    device = torch.device("cuda" if (torch.cuda.is_available() and opts.device != "cpu") else "cpu")

    # get trainset and valset dataloaders for training
    data_initializer = select_dataloader(configs.modality)(opts, configs)
    train_loader, val_loader, test_loader = data_initializer.get_training_dataloader()

    # load model
    splitprint()
    # checkpoint = configs.checkpoint if len(configs.checkpoint) else None
    checkpoint = opts.checkpoint
    model = model_structure(configs.net_name)(configs, device, checkpoint)

    criterion = torch.nn.BCEWithLogitsLoss()
    if configs.train_params["optimizer"] == "sgd":
        optimizer_params = configs.train_params["sgd"]
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_params["lr"],
                                    momentum=optimizer_params["momentum"],
                                    weight_decay=optimizer_params["weight_decay"])

    tolerance = 0
    best_epoch = 0
    best_metric = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(configs.train_params["max_epoch"]):

        splitprint()
        print('Epoch {}/{}'.format(epoch + 1, configs.train_params["max_epoch"]))

        # train step
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for i, (inputs, labels_onehot, imagenames) in enumerate(train_loader):
            data_time.update(time.time() - end)
            labels_onehot = labels_onehot.float().to(device)
            optimizer.zero_grad()
            if configs.net_name in ["cfp-model", "oct-model"]:
                outputs, _ = model(inputs.to(device))
                inputs_size = inputs.size(0)
            else:
                print("model {} is not support.").format(configs.net_name)

            loss_cls = criterion(outputs, labels_onehot)
            loss = loss_cls
            loss.backward()
            optimizer.step()

            losses.update(loss, inputs_size)
            batch_time.update(time.time() - end)
            end = time.time()

            if i % opts.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                # 'Single Loss {loss_fundus:.4f}, {loss_oct:.4f}\t'
                # 'Classification Loss: {loss_cls:.4f}\t'
                .format(
                    epoch, i, len(train_loader),
                    batch_time=batch_time, data_time=data_time, loss=losses,
                    # loss_fundus=loss_fundus, loss_oct=loss_oct,
                    # loss_cls=loss_cls
                ))

        # val step
        model.eval()
        test_metric, _ = validate(model, test_loader, configs.train_params["best_metric"],
                                 device, configs.cls_num, configs.net_name, not configs.if_syn)
        model_wts = copy.deepcopy(model.state_dict())

        if test_metric > best_metric:
            best_epoch = epoch
            best_metric = test_metric
            best_model_wts = copy.deepcopy(model.state_dict())
            print("save the better weights, metric value: {}".format(best_metric))
            save_model(best_model_wts, opts, epoch, best_metric, if_syn=configs.if_syn, best_model=True)
            print("test metric value: {}".format(test_metric))
            tolerance = 0
        elif epoch > optimizer_params["lr_decay_start"]:
            tolerance += 1
            if tolerance % optimizer_params["tolerance_iter_num"] == 0:
                if_stop = adjust_learning_rate(optimizer, optimizer_params)
                print("best:", best_metric)
                if if_stop:
                    break

    save_model(model_wts, opts, epoch, best_metric, if_syn=configs.if_syn)
    print("validation metric value: {}".format(best_metric))


if __name__ == "__main__":
    opts = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.device
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.seed)
    main(opts)
