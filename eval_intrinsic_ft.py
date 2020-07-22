#!/usr/bin/env python3
"""Script for Intrinsic evaluations (todo. currently for BMET-ft only)"""
import argparse
import logging
import code

from BMET.eval import EmbEvaluator


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wv_file', type=str, default='',
                        help='Path to a pre-trained embeddings vec file')
    args = parser.parse_args()

    ds = ['UMNSRS-sim-mod', 'UMNSRS-rel-mod', 'MayoSRS']
    ds = ['UMNSRS-sim-mod', 'UMNSRS-rel-mod', 'MayoSRS',
          'MiniMayoSRS-p', 'MiniMayoSRS-c',
          'Pedersen-p', 'Pedersen-c', 'Hliaoutakis']

    # Logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )

    # Load evaluator
    evaluator = EmbEvaluator(eval_sets=ds, eval_dir='data/eval/',
                             wv_=args.wv_file)
    evaluator.eval()
    # code.interact(local=locals())
