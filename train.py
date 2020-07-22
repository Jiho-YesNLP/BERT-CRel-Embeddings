"""train.py

Fine-tune embeddings in a simple relevance classification task with MeSH concepts
"""
import code
import logging
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AdamW, get_linear_schedule_with_warmup

import BMET.config as cfg
from BMET.data import MeshPairDataset, batchify
from BMET.utils import MeSHTree, TrainingLogs, generate_examples
from BMET.vocab import WvModel
from BMET.eval import EmbEvaluator
from BMET.model import MeshRelClassifier


logger = logging.getLogger(__name__)

def set_defaults():
    # Random seed
    random.seed(cfg.RSEED)
    np.random.seed(cfg.RSEED)
    torch.manual_seed(cfg.RSEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(ds, mdl, optim, sch, stats, evaluator):
    mdl.train()
    # we = mdl.bert.bert.embeddings.word_embeddings
    trn_it = DataLoader(ds['trn'], batch_size=cfg.BSZ, shuffle=True,
                          collate_fn=batchify)
    for batch in trn_it:
        optim.zero_grad()
        loss, logits = mdl(batch)
        loss.backward()
        optim.step()
        nn.utils.clip_grad_norm_(mdl.parameters(), 8)
        stats.update('train', loss.item())

        # Log
        if stats.steps % cfg.LOG_INTERVAL == 0:
            stats.lr = optim.param_groups[0]['lr']
            loss_ = stats.report_trn()
            # Step to next annealing stage
            if (mdl.is_emb_frozen and stats.lr <= cfg.MIN_LR) or\
                    stats.lr <= cfg.MIN_LR_FT:
                stats.stage += 1
                if mdl.is_emb_frozen:
                    mdl.unfreeze_embs()
                    sch.min_lrs = [cfg.MIN_LR_FT]
                    optim.param_groups[0]['lr'] = cfg.LR_FT
                else:
                    mdl.next_annealing_stage()
                    optim.param_groups[0]['lr'] = cfg.MIN_LR_FT + \
                        + (cfg.LR_FT - cfg.MIN_LR_FT) * .9**stats.stage
        if stats.steps % cfg.VAL_INTERVAL == 0:
            loss_ = validate(ds['val'], mdl, stats)
            sch.step(loss_)
            # Benchmark
            if not mdl.is_emb_frozen:
                corr = evaluator.eval(we=mdl.we,
                                      rtn_corr=['UMNSRS-sim', 'UMNSRS-rel',
                                                'MayoSRS'])
            else:
                corr = -1.
            if corr > stats.best_intr_score:
                mdl.save_model(cfg.F_OUT, update_emb=True)
                stats.best_intr_score = corr

def validate(val_ds, mdl, stats):
    val_it = DataLoader(val_ds, batch_size=cfg.BSZ, shuffle=True, collate_fn=batchify)
    with torch.no_grad():
        mdl.eval()
        for i, batch in enumerate(val_it):
            loss, _ = mdl(batch)
            stats.update('valid', loss.item())
            if i > cfg.VAL_MAX_STEPS:
                avg_loss = stats.report_val()
                return avg_loss

if __name__ == '__main__':
    # Logger
    logging.basicConfig(
        level=cfg.LOGGING_MODE,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M',
        handlers=[logging.FileHandler('train.log'),
                  logging.StreamHandler()]
    )

    # Configuration
    set_defaults()

    # Read MeSH descriptors
    logger.info('Reading MeSH tree structure...')
    msh_tr = MeSHTree(cfg.F_MESH)

    # Load pretrained static embeddings
    embs = WvModel(cfg.F_WV_EMBS)
    evaluator = EmbEvaluator(embs, eval_sets=['UMNSRS-sim', 'UMNSRS-rel'])
    # Read training dataset
    logger.info('Loading examples from %s...', cfg.F_DATA)
    exs = generate_examples(cfg.F_DATA)
    ds = {k: MeshPairDataset(exs, msh_tr, embs, mode=k)
          for k in ['trn', 'val']}

    # Model
    # -------------------------------------------------------------------
    model = MeshRelClassifier(embs)
    model.to(cfg.DEVICE)
    optimizer = AdamW(
        [p for n, p in model.named_parameters()
         if not n.startswith('bert.bert.embeddings.word_embeddings[0]')],
        lr=cfg.LR
    )

    # num_training_steps = int(len(ds['trn']) / cfg.BSZ) * cfg.EPOCHS
    # num_warmup_steps = int(num_training_steps * 0.05)
    # sch = get_linear_schedule_with_warmup(optimizer,
    #                                       num_warmup_steps,
    #                                       num_training_steps)
    sch = ReduceLROnPlateau(optimizer, patience=cfg.SCH_PATIENCE,
                            factor=cfg.SCH_FACTOR, min_lr=cfg.MIN_LR,
                            verbose=True)
    running_stats = TrainingLogs()

    for epoch in range(1, cfg.EPOCHS + 1):
        running_stats.epoch = epoch
        logger.info('*** Epoch %s ***', epoch)
        train(ds, model, optimizer, sch, running_stats, evaluator)
