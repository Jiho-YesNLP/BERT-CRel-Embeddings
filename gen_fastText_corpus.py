"""gen_fastText_corpus.py

This script generates training datasets for the fastText inputs .We use two
textual resources: PubTator and MeSH definitions.  Using the PubTator dataset,
we utilize the MeSH annonations on the PubMed articles. In order for fastText
to capture all of the MeSH terms, we include the entire definitions of the MeSH
descriptors and seriealize them using the relationships among the concepts in
its hierarchical structure.  """

import logging
import gzip
import multiprocessing as mp
import re
from functools import partial
import random

from lxml import etree

from BMET.utils import MeSHTree, MeSH

logger = logging.getLogger(__name__)

# Paths
PT_FILE = 'data/pubtator/bioconcepts2pubtatorcentral.offset.gz'
MESH_FILE = 'data/mesh/desc2020.xml'
MUST_INCLUDE_FILE = 'data/must_include_docs.txt'
EXTRA_DOCS_FILE = 'data/must_include_docs_ext.txt'
FT_OUT = 'data/PT_corpus'

# Global
ENT_INDICATOR = 'εmesh_'
CPU_MAX = min(80, mp.cpu_count() - 1)


def interpolate_meshes(text, entities):
    """Interpolate entities into the text"""
    if len(entities) == 0:
        return text
    ret_txt = ''
    pointer = 0
    for ent_line in entities:
        # We assume that entities are sorted by the starting position and each
        # record contains six fields
        try:
            doc_id, start_pos, end_pos, phrase, code_type, code_id = \
                ent_line.split('\t')
        except ValueError:  # Ignore records that mismatch the format
            continue
        start_pos = int(start_pos)
        end_pos = int(end_pos)
        if end_pos > len(text):
            break
        # If multiple codes are found, use the first one
        if '-' in code_id:  # MeSH code N/A
            continue
        if '|' in code_id:  # Multiple MeSH codes
            code_id = code_id.split('|')[0]
        if ':' in code_id:  # Most cases
            _, code = code_id.split(':')[:2]
        else:  # No prefix
            _, code = ('MESH', code_id)

        ent_label = ' ' + ENT_INDICATOR + code.lower() + ' '
        ret_txt += text[pointer:end_pos] + ent_label
        pointer = end_pos
    ret_txt += text[pointer:]  # The rest of the document
    return ret_txt


def mp_proc_docs(docs, include_docs=[]):
    # Parse document components
    r_title = re.compile(r'^((\d+)\|t\|)(.*)$')
    r_body = re.compile(r'^((\d+)\|a\|)(.*)$')
    r_annt = re.compile(r'^\d+\t\d+\t\d+\t[^\t]+\t(Chemical|Disease)\t.*')
    sample_ratio = 0.06

    out = []  # list of document texts
    for doc in docs:
        did = None
        title_lines = []
        body_lines = []
        mesh_lines = []
        for line in doc:
            m = r_title.match(line)
            if m:
                if did is None:
                    did = m.group(2)
                title_lines.append(m.group(3))
            m = r_body.match(line)
            if m and len(m.group(3)) > 0:
                body_lines.append(m.group(3))
            m = r_annt.match(line)
            if m:
                mesh_lines.append(m.group(0))
        if len(body_lines) == 0 or len(mesh_lines) == 0:
            continue
        if did not in include_docs and random.random() > sample_ratio:
            continue
        doc_text = ' '.join(title_lines + body_lines)
        doc_text = interpolate_meshes(doc_text, mesh_lines)
        out.append(doc_text)

    return '\n'.join(out)


def proc_pubtator():
    """Read PubTator documents, clean-up texts, and interpolate MeSH
    entities"""

    logger.info('Processing PubTator documents...')

    # Read list of PubMed doc ids that need to be included in training data
    with open(MUST_INCLUDE_FILE) as f:
        must_include_docs = f.read().split()

    b_cnt = 0

    def cb_write(res):
        with open(FT_OUT, 'a') as fout:
            fout.write(res)
    bsz = 40000  # batch of docs
    docs = []
    a_doc = []
    p = mp.Pool(CPU_MAX)
    mp_f = partial(mp_proc_docs, include_docs=must_include_docs)
    with gzip.open(PT_FILE, 'rt', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:  # End of file
                break
            if line == '\n':  # End of a document
                docs.append(a_doc)
                a_doc = []
                if len(docs) == bsz:
                    b_cnt += 1
                    print('batch {}=~774\r'.format(b_cnt), end='')
                    p.apply_async(mp_f, (docs,), callback=cb_write)
                    docs = []
            else:
                a_doc.append(line.rstrip())

        if len(docs) > 0:
            b_cnt += 1
            print('batch {}=~774\r'.format(b_cnt), end='')
            p.apply_async(mp_f, (docs, ), callback=cb_write)
            docs = []
    p.close()
    p.join()


def add_extra_document():
    """Some of the words in the evaluation datasets can't be found in the
    PubMed corpus. We add just enough number of documents that contain those
    word from outside (e.g., Wikipedia articles) in order for the fastText to
    add these words in its vocabulary."""
    with open(FT_OUT, 'a') as fout, open(EXTRA_DOCS_FILE) as fin:
        fout.write(fin.read())


def proc_mesh_desc():
    """Parse the MeSH descriptors file (XML) and construct a MeSH tree, then
    use it for generating training examples of the MeSH definitions."""
    logger.info('Reading MeSH descriptors...')

    trace = []
    t = MeSHTree(MESH_FILE)
    for cat in t.root.children:
        t.inorder_traversal(cat, trace)
    logging.info('Traversing {} nodes of the MeSH tree'.format(len(trace)))
    logging.info('Encoding and Tokenizing MeSH definitions...')
    with open(FT_OUT, 'a') as f:
        for mesh_ui in trace:
            mesh = t.meshes[mesh_ui]
            ex = ('{} {}{} {}'.format(mesh.name, ENT_INDICATOR,
                                      mesh_ui.lower(), mesh.note))
            f.write(ex + '\n')


if __name__ == '__main__':
    # Logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )

    # if not run yet
    proc_pubtator()
    add_extra_document()  # There are a few words that can't be found in PM
    proc_mesh_desc()
