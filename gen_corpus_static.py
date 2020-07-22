"""gen_corpus_static.py

This script generates training datasets for the fastText inputs .We use two
textual resources: PubTator and MeSH definitions.  Using the PubTator dataset,
we utilize the MeSH annonations on the PubMed articles. In order for fastText
to capture all of the MeSH terms, we include the entire definitions of the MeSH
descriptors and seriealize them using the relationships among the concepts in
its hierarchical structure.

Once we obtain text output for training, apply post-process for splitting word
boundaries and converting it to lowercase, such as,

`cat PT_corpus| sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > PT_corpus.train`
"""

import gzip
from collections import Counter
import multiprocessing as mp
import re
import random
from functools import partial

from tqdm import tqdm

from BMET.utils import MeSHTree

def parse_mesh_code(mesh):
    """Parse MeSH code from annotation lines"""
    if len(mesh) < 7:
        return None, None
    if ';' in mesh:  # (e.g. MESH:D003920;0.04073532087143564)
        mesh = mesh.split(';')[0]
    if '|' in mesh:  # Multiple codes, we use the primary one (i.e., the first)
        mesh = mesh.split('|')[0]
    if ':' in mesh:  # Most cases, single code (e.g., MESH:D003920)
        src, code_id = mesh.split(':')[:2]
    else:  # No prefix
        src, code_id = ('MESH', mesh)

    return (code_id + src).lower(), code_id


def affix_meshes(text, entities):
    """Interpolate entities into the text"""
    ret_txt = ''
    pointer = 0
    mesh_counted = []
    for ent_line in entities:
        # We assume that entities are sorted by the starting position and each
        # record contains six fields
        try:
            doc_id, start_pos, end_pos, phrase, code_type, mesh = \
                ent_line.split('\t')
        except ValueError:  # Ignore records that mismatch the format
            continue
        start_pos = int(start_pos)
        end_pos = int(end_pos)
        if end_pos > len(text):
            break
        c, msh_id = parse_mesh_code(mesh)
        if c is not None:
            ret_txt += text[pointer:end_pos] + ' ' + c + ' '
            mesh_counted.append(msh_id)
        pointer = end_pos
    ret_txt += text[pointer:]  # The rest of the document
    return ret_txt, mesh_counted


def mp_proc_docs(docs, meshes, include_docs=None):
    # Parse document components
    r_title = re.compile(r'^((\d+)\|t\|)(.*)$')
    r_body = re.compile(r'^((\d+)\|a\|)(.*)$')
    r_annt = re.compile(r'^\d+\t\d+\t\d+\t[^\t]+\t(Chemical|Disease)\t(.*)')
    sample_ratio = 0.01

    out = []  # sequence of document texts
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
        # Skip documents that either empty or contain no new MeSH code
        if include_docs is not None and did in include_docs:
            pass
        else:
            if len(body_lines) == 0 or len(mesh_lines) == 0:
                continue
            if random.random() > sample_ratio:
                continue

        doc_text = ' '.join(title_lines + body_lines)
        doc_text, meshes_update = affix_meshes(doc_text, mesh_lines)
        out.append(doc_text)

    return '\n'.join(out)

def proc_pubtator(msh_mh=None):
    """Read PubTator documents, clean-up texts, and insert the MeSH codes in document texts"""
    print('Processing PubTator documents...')
    a_doc = []
    docs = []
    bsz = 10000

    # Read a list of PubMed doc ids that need to be included in training data
    with open(F_MUST_INCLUDE_PM) as f:
        must_include_docs = f.read().split()

    pbar = tqdm(total=27000000)  # Approximate count
    p = mp.Pool(mp.cpu_count())
    mp_fn = partial(mp_proc_docs, include_docs=must_include_docs)

    def cb_write(res):
        nonlocal msh_mh, pbar
        text = res
        with open(F_OUT, 'a') as fout:
            fout.write(text)

    with gzip.open(F_PT, 'rt', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:  # End of file
                break
            if line == '\n':  # End of a document
                docs.append(a_doc)
                a_doc = []
                pbar.update()
                if len(docs) == bsz:
                    p.apply_async(mp_fn, (docs, msh_mh), callback=cb_write)
                    docs = []
            else:
                a_doc.append(line.rstrip())
        if len(docs) > 0:  # ignoring the remainings
            docs = []
    pbar.close()
    p.close()
    p.join()


def proc_mesh_desc(t):
    """Parse the MeSH descriptors file (XML) and construct a MeSH tree, then
    use it for generating training examples of the MeSH definitions."""
    trace = []
    for cat in t.root.children:
        t.inorder_traversal(cat, trace)
    print('Traversing {} nodes of the MeSH tree'.format(len(trace)))
    print('Encoding and Tokenizing MeSH definitions...')
    with open(F_OUT, 'a') as f:
        for mesh_ui in trace:
            mesh = t.meshes[mesh_ui]
            msh_code = (mesh_ui + 'MESH').lower()
            ex = '{} {} {}'.format(mesh.name, msh_code, mesh.note)
            f.write(ex + '\n')


def add_extra_documents():
    """Some of the words in the evaluation datasets can't be found in the
    PubMed corpus. We add just enough number of documents that contain those
    word from outside (e.g., Wikipedia articles) in order for the fastText to
    add these words in its vocabulary."""
    print('Adding extra documents...')
    with open(F_OUT, 'a') as fout, open(F_MUST_INCLUDE_EXTRA) as fin:
        fout.write(fin.read())


if __name__ == '__main__':
    # File Paths
    F_PT = 'data/pubtator/bioconcepts2pubtatorcentral.offset.gz'
    F_MESH = 'data/mesh/desc2020.xml'
    F_MUST_INCLUDE_PM = 'data/must_include_docs.txt'
    F_MUST_INCLUDE_EXTRA = 'data/must_include_docs_ext.txt'
    F_OUT = 'data/PT_corpus'

    # Read MeSH descriptors into a tree structure
    print('Reading MeSH tree structure...')
    msh_tr = MeSHTree(F_MESH)

    # Process the annotated PubMed documents (PubTator)
    proc_pubtator(msh_mh=set(msh_tr.meshes.keys()))
    # Process the MeSH definitions
    proc_mesh_desc(msh_tr)
    # Add additional texts for words not in Pubmed documents
    add_extra_documents()  # There are some words not found in PM
