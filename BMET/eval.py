import code
from collections import defaultdict
from os.path import join as pjoin
import csv
import logging

from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from BMET.vocab import WvModel
import BMET.config as cfg

logger = logging.getLogger(__name__)


class EmbEvaluator:
    """A class for evaluation pretrained/fine-tuned embeddings via intrinsic methods"""
    def __init__(self, mdl=None, eval_dir=cfg.D_EVAL, eval_sets=list(cfg.F_EVAL.keys())):
        self.eval_sets = eval_sets
        self.eval_dir = eval_dir
        self.reference = self.read_eval_sets()
        if isinstance(mdl, WvModel):
            self.wv_mdl = mdl
        else:
            self.wv_mdl = WvModel(mdl, normalize_emb=True)

    def read_eval_sets(self):
        """ Read evaluation datasets and store them in a unified format. """
        ref = defaultdict(list)
        logger.info('Reading evaluation datasets...')
        for ds, (filename, mapping) in cfg.F_EVAL.items():
            with open(pjoin(self.eval_dir, filename)) as f:
                csv_reader = csv.reader(map(lambda line: line.lower(), f))
                next(csv_reader)  # Skip the header row
                for rec in csv_reader:
                    ex = [None] * 5
                    for i, v in enumerate(mapping):
                        if i < 2:  # terms
                            ex[i] = rec[mapping[i]]
                        elif i < 4:  # mesh terms
                            if rec[mapping[i]] != 'none':
                                ex[i] = rec[mapping[i]] + 'mesh'
                            else:
                                ex[i] = 'none'
                        else:  # score values
                            ex[i] = float(rec[mapping[i]])
                    ref[ds].append(ex)
        return ref

    def eval(self, we=None, rtn_corr=[]):
        """
        Evaluate the embeddings on specified benchmarks

        `we` and `rtn_corr` is given while in training. In that case, update
        the WvModel embeddings with the current embeddings and evaluate"""
        code.interact(local=locals())
        if we is not None:
            # Update WvModel embeddings to current vectors in training
            embs = torch.mm(we[0].weight, we[1].weight).cpu().detach()
            # Normalize
            embs = F.normalize(embs, p=2, dim=1)
        else:
            embs = self.wv_mdl.emb

        # evaluate with a reference set
        corr_sum = 0
        for ds in self.eval_sets:
            logger.info('Evaluating on %s', ds)
            ref = self.reference[ds]
            oovs = set()
            mat_size = torch.Size((len(ref), embs.size(1)))
            mat = [torch.empty(mat_size) for _ in range(4)]

            for i, rec in enumerate(ref):
                for m, ph in enumerate(rec[:4]):
                    idx = [self.wv_mdl[t] for t in ph.split()]
                    if 1 in idx:
                        oovs.update(ph.split())
                    v = torch.mean(
                        torch.stack([embs[j] for j in idx]), dim=0
                    )
                    mat[m][i] = v
            logger.debug('OOVs: %s', oovs)

            list(map(lambda x: F.normalize(x), mat))
            # Term
            gt = [rec[4] for rec in ref]
            scores_t = torch.einsum('ij, ij -> i', mat[0], mat[1]).numpy()
            rho, pval = spearmanr(gt, scores_t)
            if ds in rtn_corr:
                corr_sum += rho
            logger.info('on words: rho ({:.3f}) pvalue ({})'.format(rho, pval))

            scores_m = torch.einsum('ij, ij -> i', mat[2], mat[3]).numpy()
            rho, pval = spearmanr(gt, scores_m)
            if ds in rtn_corr:
                corr_sum += rho
            logger.info('on mesh: rho ({:.3f}) pvalue ({})'.format(rho, pval))

            rho, pval = spearmanr(gt, self.compute_RRF(scores_t, scores_m))
            logger.info('on rrf: rho ({:.3f}) pvalue ({})'.format(rho, pval))
            if ds  in rtn_corr:
                corr_sum += rho

        if len(rtn_corr) != 0:
            return corr_sum
        return

    def compute_RRF(self, list1, list2, k=60):
        assert len(list1) == len(list2), "lengths of the lists must agree"
        rankings = [None, None]
        for i, l in enumerate((list1, list2)):
            rank_idx = sorted(range(len(l)), key=lambda e: l[e], reverse=True)
            ranking = [0] * len(l)
            for j, x in enumerate(rank_idx):
                ranking[x] = j
            rankings[i] = ranking
        rrf_scores = []
        for r1, r2 in zip(*rankings):
            s = 1 / (k + r1 + 1)
            s += 1 / (k + r2 + 1)
            rrf_scores.append(s)
        return rrf_scores


if __name__ == '__main__':
    ee = EmbEvaluator()