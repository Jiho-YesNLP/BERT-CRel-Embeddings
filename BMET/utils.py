import code
import logging
import pickle
import random
from itertools import combinations
from collections import Counter

import torch
from lxml import etree
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, node_ui, mesh=None):
        self.node_ui = node_ui
        self.mesh = mesh
        self.children = list()


class MeSH:
    def __init__(self, ui, name, note='', pa=None, sa=None):
        self.ui = ui
        self.name = name
        self.note = note
        self.tree_numbers = []  # list of node_ui's
        self.pa = pa  # Parmacological Actions
        self.sa = sa  # See Also


class MeSHTree:
    """
    MeSH tree is constructed using the TreeNumbers of all codes.

    You can render a branch of the tree by
      a TreeNumber (e.g. `render_branch('D05.500.099')`) or
      a MeSH ui (e.g. `render_mesh('D007212')`).

    Also, you can find a traversal path from a node of the tree
      (e.g. `path = []; inorder_traversal('D05.500', path); print(path)`).
    """
    categories = {
        'E': 'Analytical, Diagnostic and Therapeutic Techniques and Equipment Category',
        'A': 'Anatomy Category',
        'I': 'Anthropology, Education, Sociology and Social Phenomena Category',
        # 'X': 'Check Tags Category',
        'D': 'Chemicals and Drugs Category',
        'H': 'Disciplines and Occupations Category',
        'C': 'Diseases Category',
        'Z': 'Geographical Locations Category',
        'N': 'Health Care Category',
        'K': 'Humanities Category',
        'L': 'Information Science Category',
        'B': 'Organisms Category',
        'M': 'Persons Category',
        # 'R': 'Pharmacological Actions Category',
        'G': 'Phenommodel and Processes Category',
        'F': 'Psychiatry and Psychology Category',
        'V': 'Publication Type Category',
        # 'Y': 'Subheadings Category',
        'J': 'Technology and Food and Beverages Category'
    }

    def __init__(self, desc_file):
        """
        Args
            desc_file: path to the MeSH descriptor XML file
        """
        self.root = Node('*', None)
        self.nodes = dict()  # set of nodes
        self.meshes = dict()  # set of meshes
        self.excluded_meshes = dict()  # meshes not in the tree
        self.read_mesh_tree(desc_file)

    def read_mesh_tree(self, fpath):
        logger.info('Reading MeSH descriptors file...')
        data = etree.parse(open(fpath))
        for rec in data.getiterator('DescriptorRecord'):
            descriptor_ui = rec.find('DescriptorUI').text
            descriptor_name = rec.find('DescriptorName/String').text
            scope_path = 'ConceptList/Concept[@PreferredConceptYN="Y"]/ScopeNote'
            scope_elm = rec.find(scope_path)
            scope_note = scope_elm.text.strip() if scope_elm is not None else ''
            pa_path = './/PharmacologicalAction/DescriptorReferredTo/DescriptorUI'
            sa_path = './/SeeRelatedDescriptor/DescriptorReferredTo/DescriptorUI'
            pa = [e.text for e in rec.findall(pa_path)]
            sa = [e.text for e in rec.findall(sa_path)]
            m = MeSH(descriptor_ui, descriptor_name, scope_note, pa, sa)
            tn_list = []
            try:
                for tn in rec.find('TreeNumberList'):
                    tn_list.append(tn.text)
                    self.add_child(m, tn.text)
            except TypeError:
                self.excluded_meshes[m.ui] = m
                continue
            finally:
                m.tree_numbers = tn_list

    def add_child(self, mesh, node_ui):
        if node_ui not in self.nodes:
            self.nodes[node_ui] = Node(node_ui, mesh)
        else:
            self.nodes[node_ui].mesh = mesh
            self.nodes[node_ui].node_ui = node_ui
        if mesh.ui not in self.meshes:
            self.meshes[mesh.ui] = mesh
        if len(node_ui.split('.')) == 1:  # A root node
            self.root.children.append(node_ui)
        else:
            parent_tn = '.'.join(node_ui.split('.')[:-1])
            if parent_tn not in self.nodes:
                self.nodes[parent_tn] = Node(parent_tn)
            self.nodes[parent_tn].children.append(node_ui)

    def inorder_traversal(self, node_ui, path, return_type='mesh'):
        children = self.nodes[node_ui].children
        for ch in children[:-1]:
            self.inorder_traversal(ch, path, return_type=return_type)
        if return_type == 'mesh':
            path.append(self.nodes[node_ui].mesh.ui)
        else:
            path.append(node_ui)
        if len(children) > 0:
            self.inorder_traversal(children[-1], path, return_type=return_type)

    def render_branch(self, node_ui):
        trace = []
        for i in range(len(node_ui.split('.'))):
            n = '.'.join(node_ui.split('.')[:i+1])
            trace.append(self.nodes[n].mesh.name)
        level = 0
        print('# {}'.format(self.categories[node_ui[0]]))
        for i, name in enumerate(trace):
            print('   ' * i, '-', name)
            level = i
        for n in self.nodes[node_ui].children:
            print('   ' * level, '     -', self.nodes[n].mesh.name)

    def render_mesh(self, mesh_ui):
        for tn in self.meshes[mesh_ui].tree_numbers:
            self.render_branch(tn)

    def find_mesh_family(self, tn):
        family = []
        if tn not in self.nodes:
            return family
        a = tn.split('.')
        # if len(a) >= 4:  # ggp
        #     family.append(self.nodes['.'.join(a[:-1])].mesh.ui)
        # if len(a) >= 3:  # gp
        #     family.append(self.nodes['.'.join(a[:-2])].mesh.ui)
        if len(a) >= 2:  # p
            p = self.nodes['.'.join(a[:-1])]
            family.append(p.mesh.ui)
            for ch in p.children:  # siblings
                family.append(self.nodes[ch].mesh.ui)
        for ch in self.nodes[tn].children:
            # for gch in self.nodes[ch].children:
                # family.append(self.nodes[gch].mesh.ui)  # gch
                # for ggch in self.nodes[gch].children:
                #     family.append(self.nodes[ggch].mesh.ui)  # ggch
            family.append(self.nodes[ch].mesh.ui)  # ch
        return family

    def __str__(self):
        return f'#MeSHes: {len(self.meshes)}, #Nodes: {len(self.nodes)}'


class TrainingLogs:
    def __init__(self):
        self.epoch = 0
        self.steps = 0
        self.stage = 0  # Embeddings annealing stages
        self.lr = 0
        self.train_loss = []
        self.train_loss_acc = []  # _acc: Accumulated for analysis purpose
        self.valid_loss = []
        self.valid_loss_acc = []
        self.accuracy_acc = []  # [(accuracy, steps), ...]
        self.is_emb_frozen = True
        self.aWordVector = None
        self.best_intr_score = 0

    def update(self, mode, loss):
        if mode == 'train':
            self.steps += 1
            self.train_loss.append(loss)
            self.train_loss_acc.append((loss, self.steps))
        elif mode == 'valid':
            self.valid_loss.append(loss)
            self.valid_loss_acc.append((loss, self.steps))

    def report_trn(self, aWV=None):
        avg_loss = sum(self.train_loss) / len(self.train_loss)
        self.aWordVector = aWV
        msg = (
            'epoch: {}, steps: {}, loss: {:.6f}, lr: {:.8f}'
            ''.format(self.epoch, self.steps, avg_loss, self.lr)
        )
        self.train_loss = []
        logger.info(msg)
        return avg_loss

    def report_val(self):
        avg_loss = sum(self.valid_loss) / len(self.valid_loss)
        msg = (
            'VAL// loss: {:.6f}'.format(avg_loss)
        )
        self.valid_loss = []
        logger.info(msg)
        return avg_loss

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    if isinstance(lengths, list):
        lengths = torch.tensor(lengths)
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def generate_examples(fp):
    """Read stored mesh-doc data and generate positive/negative concept
    pair examples"""
    data = pickle.load(open(fp, 'rb'))
    exs = []
    mesh_counter = Counter()
    # Find most common meshes (approximate)
    for ex in data:
        mesh_counter.update(ex[0])
    cutoff = 50
    mesh_common = [k for k, v in mesh_counter.most_common(cutoff)]
    mesh_neg_pool = [m for m in mesh_counter.most_common()[cutoff:]]

    # Generate pos examples
    print('Generating positive examples...')
    for ex in tqdm(data):
        # ex in ([meshes in group], context)
        for m1, m2 in combinations(ex[0], 2):
            if any(e in mesh_common for e in (m1, m2)):
                continue
            exs.append((1, m1, m2, ex[1]))

    # Generate neg examples
    print('Generating negative examples...')
    rnd_mesh_pool = random.choices(population=[m[0] for m in mesh_neg_pool],
                                   weights=[m[1] for m in mesh_neg_pool],
                                   k=len(exs))
    for i in trange(len(exs)):
        ex = exs[i]
        if i % 2 == 0:
            ex_neg = (0, rnd_mesh_pool[i], ex[2], ex[3])
        else:
            ex_neg = (0, ex[1], rnd_mesh_pool[i], ex[3])
        exs.append(ex_neg)
    print('{} examples generated'.format(len(exs)))
    random.shuffle(exs)
    return exs
