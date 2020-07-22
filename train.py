"""train.py

Fine-tune embeddings in simple relevancy classification tasks of MeSH concepts
"""
import argparse
import logging
import code

from lxml import etree

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AdamW

from BMET.model import MeshRelClassifier
from BMET.data import MeshPairDataset, batchify
from BMET.utils import WvModel, TrainingStats
from BMET.eval import EmbEvaluator

logger = logging.getLogger(__name__)


def set_defaults():
    """Set default configurations"""
    args.use_cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')


def train(args, tr_ds, vl_ds, mdl, optim, sch, stats):
    mdl.train()
    validity = [0, 0]
    we = mdl.bert.bert.embeddings.word_embeddings
    for batch in DataLoader(tr_ds, batch_size=args.batch_size,
                            collate_fn=batchify):
        optim.zero_grad()
        loss, logits = mdl(batch)
        validity[0] += args.batch_size
        validity[1] += (logits.argmax(1) ^ batch[0]).sum().item()
        loss.backward()
        optim.step()
        nn.utils.clip_grad_norm_(mdl.parameters(), 10)
        stats.update('train', loss.item())
        if stats.steps == 0:
            continue
        if stats.steps % args.log_interval == 0:
            print(validity)
            validity = [0, 0]
            stats.lr = optimizer.param_groups[0]['lr']
            if stats.is_emb_frozen and stats.steps > 10000:
                stats.is_emb_frozen = False
                for param in we[1].parameters():
                    param.requires_grad = True

            loss_ = stats.report_tr()
            sch.step(loss_)
            evaluator.eval_in_loop(we)
        if stats.steps % args.eval_interval == 0:
            validate(args, vl_ds, mdl, stats)


def validate(args, ds, mdl, stats):
    mdl.eval()
    # it = DataLoader(ds, batch_size=args.batch_size, sampler=RandomSampler(ds),
    it = DataLoader(ds, batch_size=args.batch_size, collate_fn=batchify)
    for i, batch in enumerate(it):
        loss, logits = mdl(batch)
        stats.update('valid', loss.item())
        if len(stats.valid_loss) > 10000:
            stats.report_vl()
            break
    mdl.train()


if __name__ == '__main__':
    # Configuration -----------------------------------------------------------
    parser = argparse.ArgumentParser(
        'Fine-tuning BMET embeddings for relationships between MeSH concepts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Runtime environmnet
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--debug', action='store_true',
                         help='Run in debug mode (= verbose)')
    runtime.add_argument('--epochs', type=int, default=5,
                         help='Number of epochs to train')
    runtime.add_argument('--log_interval', type=int, default=200,
                         help='Log interval for training')
    runtime.add_argument('--eval_interval', type=int, default=10000,
                         help='Log interval for validation')
    mdl = parser.add_argument_group('Model Configuration')
    mdl.add_argument('--mdl_dim', type=int, default=396,
                     help='BERT model hidden_size; dim of word embeddings '
                     'should be consistent with this')
    mdl.add_argument('--vocab_size', type=int, default=200000,
                     help='Number of words in pretrained vocabulary')
    mdl.add_argument('--batch_size', type=int, default=16,
                     help='Number of examples in running train/valid steps')
    mdl.add_argument('--lr', type=float, default=5e-5,
                     help='Learning rate')
    args = parser.parse_args()

    # Logger
    log_lvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_lvl,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )

    # Default settings --------------------------------------------------------
    set_defaults()
    mesh_file = 'data/mesh/desc2020.xml'
    exs_file = 'data/models/BMET-toy.vec'
    exs_file = 'data/models/BMET-ft-06337.vec'

    # Read MeSH definitions
    mesh_dict = dict()
    data = etree.parse(open(mesh_file))
    for rec in data.getiterator('DescriptorRecord'):
        ui = rec.find('DescriptorUI').text
        name = rec.find('DescriptorName/String').text
        note_elm = rec.find('ConceptList/Concept/ScopeNote')
        note = note_elm.text.strip() if note_elm is not None else ''
        mesh_dict[ui] = {'name': name, 'note': note}
    del(data)
    # Read pre-trained WvModel
    vocab = WvModel(exs_file, vocab_size=args.vocab_size, dim=args.mdl_dim)
    evaluator = EmbEvaluator(['UMNSRS-sim-mod', 'UMNSRS-rel-mod'],
                             eval_dir='data/eval/', wv_=vocab)

    # Model -------------------------------------------------------------------
    model = MeshRelClassifier(vocab)
    model.to(args.device)
    # criterion = nn.BCELoss(reduction='none')
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.8,
                                  min_lr=5e-6)

    # Read training datasets
    train_ds = MeshPairDataset('data/bmet-mt.train', mesh_dict, vocab)
    valid_ds = MeshPairDataset('data/bmet-mt.valid', mesh_dict, vocab)

    # Train -------------------------------------------------------------------
    logger.info('Start training REL model')
    model.train()
    stats = TrainingStats()
    try:
        for epoch in range(1, args.epochs + 1):
            stats.epoch = epoch
            logger.info('*** Epoch {} ***'.format(epoch))
            train(args, train_ds, valid_ds, model, optimizer, scheduler, stats)

            break
    except KeyboardInterrupt:
        code.interact(local=locals())
