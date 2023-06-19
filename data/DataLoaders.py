# coding: utf-8

import os
import abc
import torch
import pickle
from .DataOrganizer import DataOrganizer


class BaseDataLoader(object):
    def __init__(self, opts, configs):
        self.opts = opts
        self.configs = configs

    @abc.abstractmethod
    def init_trainset_params(self):
        pass

    @abc.abstractmethod
    def init_valset_params(self):
        pass

    @abc.abstractmethod
    def init_testset_params(self):
        pass

    @abc.abstractmethod
    def init_camgenerating_params(self):
        pass

    def get_training_dataloader(self):
        print("train set statistics:")
        trainset_params = self.init_trainset_params()
        trainset, fundus_buffer = self.get_data(**trainset_params)
        train_dataloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.configs.train_params["batch_size"], shuffle=True,
            num_workers=self.opts.num_workers)
        print("-" * 100)
        print("validation set statistics:")
        valset_params = self.init_valset_params()
        valset, _ = self.get_data(**valset_params)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=1, num_workers=self.opts.num_workers)

        testset_params = self.init_testset_params()
        testset, _ = self.get_data(**testset_params)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=self.opts.num_workers)
        return train_dataloader, val_dataloader, test_dataloader, fundus_buffer

    def get_test_dataloader(self):
        testset_params = self.init_testset_params()
        testset = self.get_data(**testset_params)
        return torch.utils.data.DataLoader(testset, batch_size=1, num_workers=self.opts.num_workers)

    def get_camgenerating_dataloader(self):
        camgenerating_params = self.init_camgenerating_params()
        dataset = self.get_data(**camgenerating_params)
        print("-" * 100)
        print("validation set statistics:")
        dataset.label_statistic()
        return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=self.opts.num_workers)




class SingleDataLoader(BaseDataLoader):
    def __init__(self, opts, configs):
        super(SingleDataLoader, self).__init__(opts, configs)
        self.get_data = DataOrganizer.get_fundus_data

    def init_trainset_params(self):
        if not self.configs.if_syn:
            self.configs.syn_collection = None
        return {"collection": self.opts.train_collection,
                "aug_params": self.configs.aug_params["augmentation"],
                "transform": self.configs.transform,
                "if_test": False,
                "cls_num": self.configs.cls_num,
                "if_eval": False,
                "image_path": self.opts.train_collection}

    def init_valset_params(self):
        return {"collection": self.opts.val_collection,
                "aug_params": self.configs.aug_params["onlyresize"],
                "transform": self.configs.transform,
                "if_test": False,
                "cls_num": self.configs.cls_num,
                "if_eval": False,
                "image_path": self.opts.train_collection}

    def init_testset_params(self, if_test=True, if_eval=False):
        return {"collection": self.opts.test_collection,
                "aug_params": self.configs.aug_params["onlyresize"],
                "transform": self.configs.transform,
                "if_test": False,
                "cls_num": self.configs.cls_num,
                "if_eval": False,
                "image_path": self.opts.train_collection}

    def init_camgenerating_params(self):
        return {"collection": self.opts.collection,
                "aug_params": self.configs.aug_params["onlyresize"],
                "transform": self.configs.transform,
                "cls_num": self.configs.cls_num}


class FundusDataLoader(SingleDataLoader):
    def __init__(self, opts, configs):
        super(FundusDataLoader, self).__init__(opts, configs)
        self.get_data = DataOrganizer.get_fundus_data


class OctDataLoader(SingleDataLoader):
    def __init__(self, opts, configs):
        super(OctDataLoader, self).__init__(opts, configs)
        self.get_data = DataOrganizer.get_oct_data


class MultiDataLoader(BaseDataLoader):
    def __init__(self, opts, configs):
        super(MultiDataLoader, self).__init__(opts, configs)
        self.get_data = DataOrganizer.get_mm_data

    def init_trainset_params(self):
        if not self.configs.if_syn:
            self.configs.syn_collection = None
        return {"collection": self.opts.train_collection,
                "aug_params": self.configs.aug_params["augmentation"],
                "transform": self.configs.transform,
                "if_test": False,
                "cls_num": self.configs.cls_num,
                "if_eval": False,
                "loosepair": self.configs.loosepair,
                "if_syn": self.configs.if_syn,
                "syn_collection": self.configs.syn_collection,
                "image_path": self.opts.train_collection}

    def init_valset_params(self):
        return {"collection": self.opts.val_collection,
                "aug_params": self.configs.aug_params["onlyresize"],
                "transform": self.configs.transform,
                "if_test": False,
                "cls_num": self.configs.cls_num,
                "if_eval": True,
                "loosepair": False,
                "if_syn": False,
                "syn_collection": None,
                "image_path": self.opts.train_collection}

    def init_testset_params(self, if_test=True, if_eval=False):
        return {"collection": self.opts.test_collection,
                "aug_params": self.configs.aug_params["onlyresize"],
                "transform": self.configs.transform,
                "if_test": False,
                "cls_num": self.configs.cls_num,
                "if_eval": True,
                "loosepair": False,
                "if_syn": False,
                "syn_collection": None,
                "image_path": self.opts.train_collection}

    def init_camgenerating_params(self):
        return {"collection": self.opts.collection,
                "aug_params": self.configs.aug_params["onlyresize"],
                "transform": self.configs.transform,
                "cls_num": self.configs.cls_num}

    def aa_get_camgenerating_dataloader(self):
        moda = self.configs.modality
        assert moda in ['fundus', 'oct', 'multi'], "choose a modality in ['fundus', 'oct', 'multi']"
        camgenerating_params = self.init_camgenerating_params()
        if moda == 'multi':
            dataset = self.get_data(**camgenerating_params)
        elif moda == 'fundus':
            dataset = DataOrganizer.get_fundus_data(**camgenerating_params)
        else:
            dataset = DataOrganizer.get_oct_data(**camgenerating_params)
        print("###########################################\ndataset statistics:")
        dataset.label_statistic()
        return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=self.opts.num_workers)

    @staticmethod
    def get_dataset_for_eval(collection):
        return DataOrganizer.get_mm_data(collection=collection, if_eval=True)
