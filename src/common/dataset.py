import random
import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle as pkl
from Bio import pairwise2
from Bio.Align import substitution_matrices
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils import feature_generate

##you can set your custom dataset here

class GFP_dataset(Dataset):
    def __init__(self, data_path, state = "train", num_prefix_tokens=64, max_len=1024):
        self.data_path = data_path
        self.state = state
        self.num_prefix = num_prefix_tokens
        self.max_len = max_len
        self.restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
        self.vocab =  [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
            'S', 'T', 'W', 'Y', 'V',"X", "m"
        ]
        self.vocab_set = set(self.vocab)
        self.restype_order =  {restype: i for i, restype in enumerate(self.vocab)}
        self.load_data()
        
    def load_data(self):
        df = pd.read_csv(self.data_path)
        targs = df["fitness"].to_numpy()
        targs = list(torch.from_numpy(targs))
        self.targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)
        labels = df["label"].to_numpy()
        labels = list(torch.from_numpy(labels))
        self.labels = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in labels], 0)
        self.seqs = list(df["seq"])
        train_size = int(len(self.seqs) * 0.85)
        if self.state == "test":
            self.data_targets = self.targets
            self.data_seqs = self.seqs
            self.data_labels = self.labels
        else:
            train_targets, valid_targets, train_seqs, valid_seqs, train_labels, valid_labels = train_test_split(self.targets, self.seqs, self.labels, train_size=train_size, random_state=42)
            if self.state == "train":
                self.data_targets = train_targets
                self.data_seqs = train_seqs
                self.data_labels = train_labels
            else:
                self.data_targets = valid_targets
                self.data_seqs = valid_seqs
                self.data_labels = valid_labels
        self.data_seqs = [seq.replace("J", "") for seq in self.data_seqs]
    def create_feature(self, sequence):
        save_dict = {}
        seq_len = len(sequence)
        fake_protoken = np.full(seq_len, 7)
        aatype = np.array([self.restype_order.get(i, self.restype_order["X"]) for i in sequence])
        residue_index = np.arange(1,seq_len+1)
        save_dict["seq_len"] = seq_len
        save_dict["aatype"] = aatype
        save_dict["label"] = fake_protoken
        save_dict["residue_index"] = residue_index
        data_dict = feature_generate(save_dict, num_prefix_tokens = self.num_prefix, max_len=self.max_len, mask=False)
        return data_dict
        
        
    def __len__(self):
        return len(self.data_seqs)

    def __getitem__(self, idx):
        sequence = self.data_seqs[idx]
        fitness = self.data_targets[idx]
        feature_dict = self.create_feature(sequence)
        return {
            'input': feature_dict,
            'fitness': fitness
        }