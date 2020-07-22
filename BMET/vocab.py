import re
import logging
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WvModel:
    """
    WordVector Model, all-in-all class contains a pretrained word vectors and
    its vocabulary functions
    """

    def __init__(self, emb_path, vocab_size=0, dim=0, bert_symbols=None,
                 normalize_emb=False):
        self.vocab_size = vocab_size
        self.dim = dim
        self.sym2idx = defaultdict(lambda: len(self.sym2idx))
        self.idx2sym = dict()
        self.emb = None
        self.patt_mesh = re.compile(r'[d|c][0-9]+mesh')
        # BERT symbols
        self.bert_symbols = ('[PAD]', '[UNK]', '[SEP]', '[CLS]', '[MASK]') \
            if bert_symbols is None else bert_symbols
        # Load vocabulary and embeddings from pretrained-embeddings
        self.__build_vocab_from_pretrained(emb_path, normalize=normalize_emb)

    def __len__(self):
        return len(self.sym2idx)

    def __build_vocab_from_pretrained(self, fpath, normalize=False):
        """
        Read MeSH codes and regular words from a pre-trained .vec file.
        Vocabulary is indexed in the following order:
        (special symbols, mesh codes, regular words up to specified size)
        """
        logging.info('Loading word vectors from {}...'.format(fpath))
        emb_list = []
        with open(fpath, encoding='utf-8') as fd:
            vs, dim = map(int, fd.readline().split())
            if self.dim == 0:
                self.dim = int(dim/12) * 12
            assert self.dim <= dim, \
                "Configured 'mdl_dim' is greater than vocab dim"

            # Add BERT symbols first
            for sym in self.bert_symbols:
                self.idx2sym[self.sym2idx[sym]] = sym
            emb_ = (0.01 * np.random.randn(len(self.bert_symbols), self.dim))
            emb_list.extend(emb_.tolist())

            reg_words = OrderedDict()
            for line in fd:
                tokens = line.rstrip().split()
                sym, vec = tokens[0], list(map(float, tokens[1:]))
                if sym in self.bert_symbols:
                    # Special symbols saved in tuned vectors
                    emb_list[self.sym2idx[sym]] = vec[:self.dim]
                elif self.patt_mesh.match(sym):
                    # Add entity codes with no condition
                    self.idx2sym[self.sym2idx[sym]] = sym
                    emb_list.append(vec[:self.dim])
                else:  # Add regular words later once MeSHes are done
                    reg_words[sym] = vec[:self.dim]
            # Regular words
            for k, v in reg_words.items():
                if self.vocab_size == 0 or len(self) < self.vocab_size:
                    self.idx2sym[self.sym2idx[k]] = k
                    emb_list.append(v)
        # redefine sym2idx to handle OOVs; OOVs are mapped to [UNK]
        self.sym2idx = defaultdict(lambda: self.sym2idx['[UNK]'], self.sym2idx)
        self.vocab_size = len(self)
        self.emb = torch.FloatTensor(emb_list)
        if normalize:
            self.emb = F.normalize(self.emb, p=2, dim=1)

        logger.info('(%s, %s) terms loaded in WvModel', len(self), self.dim)

    def get_nearest_neighbors(self, v, k=5):
        if isinstance(v, str):
            words = v.split()
            if len(words) > 1:
                v = torch.mean(
                    torch.stack([self.emb[self[w]] for w in words]), dim=0
                )
            else:
                v = self.emb[self[v]]
        # scores, ids = torch.topk(torch.mv(self.emb, v), k)
        scores, ids = torch.topk(
            F.cosine_similarity(self.emb, v.repeat(self.emb.size(0), 1)), k
        )
        return scores.tolist(), [self.idx2sym[i.item()] for i in ids]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.idx2sym[item]
        elif isinstance(item, str):
            return self.sym2idx[item]
        else:
            raise KeyError

