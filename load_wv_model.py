"""load_wv_model.py

Load pre-trained static embeddings into WvModel, which provides analytic
tools.
"""
import logging
import code
import sys

from BMET.utils import WvModel, sequence_mask

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )

    # Read MeSH definitions
    mdl = WvModel(sys.argv[1])
    code.interact(local=locals())
