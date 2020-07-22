"""gen_rel_pairs.py

Generate training datasets for MSHREL model; pairs of MeSH concepts in
documents"""

import logging
import os
import gzip
import multiprocessing as mp
import glob
from collections import Counter
import random
import pickle

import code
from tqdm import tqdm
from lxml import etree

from BMET.utils import MeSHTree, MeSH

logger = logging.getLogger()

# PATHS
PT_FILE = 'data/pubtator/bioconcepts2pubtatorcentral.offset.gz'
MESH_FILE = 'data/mesh/desc2020.xml'
PM_OUT_FILE = 'data/bmet-pm'
MT_OUT_FILE = 'data/bmet-mt'
PM_DIR = 'data/pubmed/'


def read_mesh_descriptors():
    logger.info('Reading MeSH descriptors...')
    mesh = dict()
    data = etree.parse(open(MESH_FILE))
    for rec in data.getiterator('DescriptorRecord'):
        ui = rec.find('DescriptorUI').text
        name = rec.find('DescriptorName/String').text
        mesh[ui] = name
    return mesh


def read_pubmed_file(fpath):
    meshes = []
    data = etree.parse(gzip.open(fpath))
    for doc in data.iterfind('PubmedArticle'):
        meshes_in_doc = \
            [m.get('UI') for m in
             doc.findall('.//MeshHeadingList/MeshHeading/DescriptorName')] + \
            [m.get('UI') for m in
             doc.findall('./ChemicalList/Chemical/NameofSubstance')]
        meshes.append([t for t in meshes_in_doc if t.startswith('D')])
    return meshes


def read_mesh_pubmed(num_files=10):
    logger.info('Reading PubMed articles for MeSH descriptors...')
    pm_files = sorted(glob.glob(PM_DIR + 'pubmed*.gz'))
    pbar = tqdm(total=num_files)
    mesh_cnt = Counter()
    meshes = []

    def cb_update_counter(res):
        for d in res:
            mesh_cnt.update(d)
        meshes.extend(res)
        pbar.update()
    p = mp.Pool(10)

    # for f in pm_files:
    for f in random.sample(pm_files, num_files):
        p.apply_async(read_pubmed_file, (f,), callback=cb_update_counter)
    p.close()
    p.join()
    pbar.close()

    return meshes, mesh_cnt


def generate_examples_pm(mesh_def, mesh_lists, ignored_meshes={}):
    pos = []
    neg = []
    rnd_idx = random.choices(range(len(mesh_def)),
                             k=sum([v for k, v in mesh_cnt.items()]))
    mesh_keys = list(mesh_def.keys())
    for doc_mesh in mesh_lists:
        selected = set(doc_mesh).difference(ignored_meshes)
        len_ = len(selected)
        if len_ >= 2:
            # positive
            offset = random.randint(1, len_-1)
            for i, e in enumerate(selected):
                e_pos = list(selected)[(i+offset) % len_]
                e_rnd = mesh_keys[rnd_idx[len(pos)]]
                if not (e in mesh_def and e_pos in mesh_def):
                    continue
                else:
                    pos.append((1, e, e_pos))
                    neg.append((0, e, e_rnd))
    return pos + neg


def generate_examples_mt(mesh_def, mt):
    pos = []
    neg = []
    mesh_keys = list(mesh_def.keys())
    tree_numbers = []
    for cat in mt.root.children:
        mt.inorder_traversal(cat, tree_numbers, return_type='node')
    for i, tn in enumerate(tree_numbers):
        me = mt.nodes[tn].mesh.ui
        family = mt.find_proximal_meshes(tn)
        pos_ = [(1, me, node) for node in family if me != node]
        neg_ = [(0, me, node) for node in
                random.choices(mesh_keys, k=len(pos_))]
        pos.extend(pos_)
        neg.extend(neg_)
    return pos + neg


if __name__ == '__main__':
    # Logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )

    mesh_all = read_mesh_descriptors()
    # Generate training datasets from the mesh distribution in PubMed articles
    if not os.path.exists(PM_OUT_FILE + '.train'):
        mesh_in_pm, mesh_cnt = read_mesh_pubmed(num_files=50)
        common_meshes = {k for k, v in mesh_cnt.most_common(25)}
        exs = generate_examples_pm(mesh_all, mesh_in_pm,
                                   ignored_meshes=common_meshes)

        logger.info('Saving examples in {}...'.format(PM_OUT_FILE))
        random.shuffle(exs)
        pickle.dump(exs[:int(len(exs)*.8)], open(PM_OUT_FILE + '.train', 'wb'))
        pickle.dump(exs[int(len(exs)*.8):], open(PM_OUT_FILE + '.valid', 'wb'))

    # Generate training dataset from the MeSH tree structure
    if not os.path.exists(MT_OUT_FILE + '.train'):
        mt = MeSHTree('./data/mesh/desc2020.xml')
        exs = generate_examples_mt(mesh_all, mt)

        logger.info('Saving examples in {}...'.format(MT_OUT_FILE))
        random.shuffle(exs)
        pickle.dump(exs[:int(len(exs)*.8)], open(MT_OUT_FILE + '.train', 'wb'))
        pickle.dump(exs[int(len(exs)*.8):], open(MT_OUT_FILE + '.valid', 'wb'))

