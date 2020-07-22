"""model.py

Defines MeshRelClassifier model which predicts the relevancy between MeSH
concepts. Relevancy is defined as the co-occurrence of the terms in the same
document (REL model) or the closeness in the MeSH tree (SIM model)"""

import code
import logging

import torch
from torch import nn

from transformers import BertForSequenceClassification, BertConfig

import BMET.config as cfg

logger = logging.getLogger(__name__)


class MeshRelClassifier(nn.Module):
    def __init__(self, wv):
        super(MeshRelClassifier, self).__init__()
        self.wv_mdl = wv
        config = BertConfig(
            vocab_size=wv.vocab_size,
            hidden_size=wv.dim,
            num_hidden_layers=2
        )
        self.bert = BertForSequenceClassification(config)
        self.load_embeddings_from_vocab()
        self.is_emb_frozen = True
        self.we = self.bert.bert.embeddings.word_embeddings


    def forward(self, data):
        tgt, inp, segs, masks = data
        outputs = self.bert(inp, attention_mask=masks, token_type_ids=segs,
                            labels=tgt)
        return outputs

    def load_embeddings_from_vocab(self):
        logger.info('Loading pretrained word embeddings..')
        emb_dim = self.wv_mdl.emb.size(1)
        we = nn.Sequential(
            nn.Embedding.from_pretrained(self.wv_mdl.emb, freeze=True,
                                         padding_idx=0, sparse=True),
            nn.Linear(emb_dim, emb_dim, bias=False).requires_grad_(False)
        )
        we[1].weight.copy_(torch.eye(emb_dim))
        self.bert.bert.embeddings.word_embeddings = we


    def unfreeze_embs(self):
        logger.info('Embeddings layer is now unfreezed')
        # for param in self.bert.bert.embeddings.word_embeddings[1].parameters():
        #     param.requires_grad = True
        self.we[1].weight.requires_grad = True
        self.is_emb_frozen = False


    def next_annealing_stage(self):
        """embeddings mapping table is updated with fine-tuned
        representations and the projection layer is reset to the identity"""
        logger.info('next_annealing_stage')
        emb_org = self.we[0].weight
        proj = self.we[1].weight
        fine_tuned = torch.mm(emb_org, proj)
        with torch.no_grad():
            self.we[0].weight.copy_(fine_tuned)
            self.we[1].weight.copy_(torch.eye(self.wv_mdl.emb.size(1)))

    def save_model(self, fpath, update_emb=False):
        logging.info('Save current model and embeddings...')
        # Saving tuned embeddings to a file
        fine_tuned = torch.mm(self.we[0].weight, self.we[1].weight)
        if update_emb:
            logging.info('-updating embeddings..')
            with torch.no_grad():
                self.we[0].weight.copy_(fine_tuned)
                self.we[1].weight.copy_(torch.eye(self.wv_mdl.emb.size(1)))
        with open(fpath, 'w') as f:
            f.write('{} {}\n'.format(*fine_tuned.size()))
            for i, vec in enumerate(fine_tuned):
                f.write('{} {}\n'.format(
                    self.wv_mdl[i],
                    ' '.join(['{:.6f}'.format(v) for v in vec.tolist()])
                ))
        logging.info('Fine-tuned embeddings are saved ({})'.format(fpath))
