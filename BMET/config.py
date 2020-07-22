import logging
from time import time

import torch

EXP_ID = int(time())

# GPU
USE_CUDA = True and torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# Paths
# -----------------------------------------------------------------

# MeSH descriptor file
F_MESH = 'data/mesh/desc2020.xml'
# Pretrained static embeddings file
F_WV_EMBS = 'data/models/BMET-ft-5-396-30-5_c.vec'
# F_WV_EMBS = 'data/models/BMET-ft-5-768-30-3.vec'
# F_WV_EMBS = 'data/models/BMET-ft-toy.vec'
# Pickled training examples
F_DATA = 'data/bmet_data.pkl'
# Reference records in [t1, t2, m1, m2, score1]
F_EVAL = {
    'UMNSRS-sim': ('UMNSRS_sim_mesh.csv', (2, 3, 6, 7, 0)),
    'UMNSRS-rel': ('UMNSRS_rel_mesh.csv', (2, 3, 6, 7, 0)),
    'UMNSRS-sim-mod': ('UMNSRS_sim_mod_mesh.csv', (2, 3, 6, 7, 0)),
    'UMNSRS-rel-mod': ('UMNSRS_rel_mod_mesh.csv', (2, 3, 6, 7, 0)),
    'MayoSRS': ('MayoSRS_mesh.csv', (3, 4, 5, 6, 0)),
    'MiniMayoSRS-p': ('MiniMayoSRS_mesh.csv', (4, 5, 6, 7, 0)),
    'MiniMayoSRS-c': ('MiniMayoSRS_mesh.csv', (4, 5, 6, 7, 1)),
    'Pedersen-p': ('Pedersen2007_mesh.csv', (0, 1, 2, 3, 4)),
    'Pedersen-c': ('Pedersen2007_mesh.csv', (0, 1, 2, 3, 5)),
    'Hliaoutakis': ('Hliaoutakis2005_mesh.csv', (0, 1, 2, 3, 4))
}
D_EVAL = 'data/eval/'
D_PM = 'data/pubmed/'
F_OUT = 'data/models/BMET-tuned-{}.vec'.format(EXP_ID)

# Runtime
# -----------------------------------------------------------------

LOGGING_MODE = logging.INFO
LR = 3e-5               # learning rate for model training
MIN_LR = 2e-5           # minimum learning rate for model training
LR_FT = 3e-5            # learning rate for fine-tuning
MIN_LR_FT = 1e-5       # minimum learning rate for fine-tuning
EPOCHS = 5              # number of epochs
BSZ = 12                # batch size in training loop
RSEED = 12345           # random seed
LOG_INTERVAL = 1000     # training log interval (2000)
VAL_INTERVAL = 5000     # validation interval (6000)
VAL_MAX_STEPS = 2000   # maximum number of steps for validation (2000)
SCH_PATIENCE = 2
SCH_FACTOR = 0.9

# Model
# -----------------------------------------------------------------

# Add contextual string to the input concept pairs
ADD_CONTEXT = False