"""data.py

DataLoader, Reads in the training pairs of MeSH concepts and the labels
"""
import os
import logging
import re
from functools import partial

import code

from lxml import etree

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from BMET.utils import sequence_mask
from BMET.vocab import WvModel
import BMET.config as cfg

logger = logging.getLogger(__name__)


class MeshPairDataset(Dataset):
    def __init__(self, examples, mesh_tree, wv_model, mode='trn'):
        """Labelled MeSh pair Dataset"""
        self.meshes = mesh_tree.meshes
        # Remove examples with the mesh not found in mesh_tree.meshes
        # examples = [ex for ex in examples
        #             if ex[1] in self.meshes and ex[2] in self.meshes]
        cutoff = int(len(examples) * .9)
        self.pairs = examples[:cutoff] if mode == 'trn' else examples[cutoff:]
        self.embs = wv_model

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        y, msh1, msh2, context = self.pairs[idx]
        inp = [self.embs['[CLS]']]
        segs = [0]
        for i, m in enumerate((msh1, msh2)):
            txt = self.meshes[m].name.lower()
            inp_ = [self.embs[m.lower()+'mesh']] + \
                [self.embs[t] for t in txt.split()] + [self.embs['[SEP]']]
            inp += inp_
            segs += [i] * len(inp_)
        # Append context
        if cfg.ADD_CONTEXT:
            inp_ = [self.embs[t] for t in context] + [self.embs['[SEP]']]
            inp += inp_
            segs += [0] * len(inp_)

        return y, inp, segs

def batchify(batch):
    pad_ = partial(pad_sequence, batch_first=True, padding_value=0)
    labels = torch.tensor([rec[0] for rec in batch]).cuda()
    inputs = pad_([torch.tensor(rec[1]) for rec in batch]).cuda()
    src_lens = [len(rec[1]) for rec in batch]
    segs = pad_([torch.tensor(rec[2]) for rec in batch]).cuda()
    mask_inp = sequence_mask(src_lens, inputs.size(1)).cuda()

    return labels, inputs, segs, mask_inp
