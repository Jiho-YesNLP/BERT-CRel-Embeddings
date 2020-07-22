"""gen_finetune_exs.py

This script generates datasets for training the BMET fine-tuning model.
Examples are in
todo. finish this part
"""
import code
import csv
import glob
import pickle
import multiprocessing as mp
import random
import gzip
from collections import Counter
from itertools import combinations
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from lxml import etree
from tqdm import tqdm

from BMET.utils import MeSHTree
import BMET.config as cfg


def read_eval_terms(eval_files):
    """Read evaluation datasets and return sets of words and meshes"""
    words = set()
    meshes = set()
    for fp, mapping in eval_files:
        with open(fp) as f:
            csv_reader = csv.reader(map(lambda line: line.lower(), f))
            next(csv_reader)  # Skip the header line
            for rec in csv_reader:
                words.add(rec[mapping[0]])
                words.add(rec[mapping[1]])
                meshes.add(rec[mapping[2]].upper())
                meshes.add(rec[mapping[3]].upper())
    return (words, meshes)

def mp_read_pm_file(fpath):
    """Read meshes, doc_id, and doc_title from the PubMed documents."""

    out = []
    data = etree.parse(gzip.open(fpath))
    # remove stopwords and punctuation from context strings
    rem_tokens = set(stopwords.words('english') + list(string.punctuation))
    for d in data.iterfind('PubmedArticle'):
        if random.random() < 0.005:
            meshes = [m.get('UI') for m in
                    d.findall('.//MeshHeadingList/MeshHeading/DescriptorName')]
            title = d.find('.//ArticleTitle').text
            keywords = [k.text for k in d.findall('.//KeywordList/Keyword')]
            context = (title + ' ' + ' '.join(keywords)).lower()
            context = [t for t in word_tokenize(context) if t not in rem_tokens]
            out.append((meshes, context))
    return out


def gen_pm_examples():
    """Generate MeSH pairs with relevance labels from the PubMed metadata
    returned docs are in the following tuple format:
    [([meshes ...], doc_id, doc_title), ...]
    """
    # Read MeSH from random sample of PubMed documents
    print('Reading PubMed documents...')
    pm_files = sorted(glob.glob(cfg.D_PM + 'pubmed*.gz'))
    n_files = 0
    exs = []
    mesh_counter = Counter()

    # Callback function
    def cb_acc_report(res):
        nonlocal n_files, exs
        n_files += 1
        for d in res:
            mesh_counter.update(d[0])
            exs.append(d)
        print('n_files {}, n_docs {}\r'.format(n_files, len(exs)), end='')

    # mp_read_pm_file(pm_files[0])  # debug
    pool = mp.Pool(mp.cpu_count()-2)
    for f in pm_files:
        pool.apply_async(mp_read_pm_file, (f,), callback=cb_acc_report)
    pool.close()
    pool.join()

    return exs


def gen_mt_examples(mt):
    """Generate MeSH pairs with proximal distance"""
    print('Reading pairs from MeSH tree...')
    exs = []
    tree_numbers = []
    # remove stopwords and punctuation from context strings
    rem_tokens = set(stopwords.words('english') + list(string.punctuation))
    for cat in mt.root.children:
        mt.inorder_traversal(cat, tree_numbers, return_type='node')
    for i, tn in enumerate(tqdm(tree_numbers)):
        me = mt.nodes[tn].mesh.ui
        family = mt.find_mesh_family(tn)
        context = mt.nodes[tn].mesh.note.lower()
        context = [t for t in word_tokenize(context) if t not in rem_tokens]
        exs.append((family, context))

    # Adding PharmacologicalAction and SeeAlso relations
    for ui, me in mt.meshes.items():
        family = me.pa + me.sa
        if len(family) > 0:
            for c in family:
                context = me.note.lower()
                context = [t for t in word_tokenize(context) if t not in rem_tokens]
                exs.append(([ui, c], context))
    return exs


if __name__ == '__main__':
    msh_tr = MeSHTree(cfg.F_MESH)

    exs = []
    # Generate training examples from the PubMed MeSH labels
    exs += gen_pm_examples()
    # validate MeSH keys
    for ex in exs:
        for m in ex[0]:
            if m not in msh_tr.meshes and m != 'D005260' and m != 'D008297':
                print(ex)
                del ex
    # Generate training examples from concept relationships in MeSH tree
    exs += gen_mt_examples(msh_tr)

    pickle.dump(exs, open(cfg.F_DATA, 'wb'))

