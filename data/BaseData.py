import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler


class BaseData:
    def __init__(self, args, pt_data=None):
        self.args = args
        self.id2label, self.label2id = self._read_labels()
        self.features = torch.load(os.path.join(self.args.data_path, self.args.dataset_name, 'features.pt'))
        if pt_data:
            self.train_data = pt_data['train_data']
            self.val_data = pt_data['val_data']
            self.test_data = pt_data['test_data']

    def _read_labels(self):
        id2label = None
        label2id = None
        if self.args.dataset_name == "FewRel":
            id2label = json.load(open(os.path.join(self.args.data_path, self.args.dataset_name, 'id2label.json')))
            rel2name = json.load(open(os.path.join(self.args.data_path, self.args.dataset_name, 'rel2name.json')))
            id2label = [rel2name[rel] for rel in id2label]
            label2id = {label: i for i, label in enumerate(id2label)}
        elif "TACRED" in self.args.dataset_name:
            id2label = json.load(open(os.path.join(self.args.data_path, self.args.dataset_name, 'id2rel.json')))
            rel2name = open(os.path.join(self.args.data_path, self.args.dataset_name, 'rel2name.txt')).readlines()
            rel2name = [item.strip() for item in rel2name]
            id2label = rel2name
            label2id = {label: i for i, label in enumerate(id2label)}
        return id2label, label2id

    def _filter_key(self, datas, not_need_key):
        filter_datas = []
        for data in datas:
            filter_data = {}
            for key in data:
                if key not in not_need_key:
                    filter_data[key] = data[key]
            filter_datas.append(filter_data)
        return filter_datas

    def read_and_preprocess(self, **kwargs):
        raise NotImplementedError

    def filter(self, labels, split='train', not_need_key=['intent', 'event_type']):
        if not isinstance(labels, list):
            labels = [labels]
        if isinstance(labels[0], str):
            labels = [self.label2id[label] for label in labels]
        split = split.lower()
        res = []
        for label in labels:
            if split == 'train':
                res += self._filter_key(self.train_data[label], not_need_key)
            elif split in ['dev', 'val']:
                res += self._filter_key(self.val_data[label], not_need_key)
            elif split == 'test':
                res += self._filter_key(self.test_data[label], not_need_key)

        return res


class BaseDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, dict):
            res = []
            for key in data.keys():
                res += data[key]
            data = res
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

