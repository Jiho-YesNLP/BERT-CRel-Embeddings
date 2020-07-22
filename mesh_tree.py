#!/usr/bin/env python3
from lxml import etree
import code

from BMET.utils import MeSHTree, MeSH

t = MeSHTree()
print('Reading MeSH descriptors file...')
data = etree.parse(open('./data/mesh/desc2019'))
for rec in data.getiterator('DescriptorRecord'):
    descriptor_ui = rec.find('DescriptorUI').text
    descriptor_name = rec.find('DescriptorName/String').text
    scope_path = 'ConceptList/Concept[@PreferredConceptYN="Y"]/ScopeNote'
    scope_elm = rec.find(scope_path)
    scope_note = scope_elm.text.strip() if scope_elm is not None else ''

    m = MeSH(descriptor_ui, descriptor_name, scope_note)
    tn_list = []
    try:
        for tn in rec.find('TreeNumberList'):
            tn_list.append(tn.text)
            t.add_child(m, tn.text)
    except TypeError:
        t.excluded_meshes[m.ui] = m
        continue
    finally:
        m.tree_numbers = tn_list

banner = """
MeSH tree is constructed using the TreeNumbers of all codes.

You can render a branch of the tree by
  a TreeNumber (e.g. `t.render_branch('D05.500.099')`) or
  a MeSH ui (e.g. `t.render_mesh('D007212')`).

Also, you can find a traversal path from a node of the tree
  (e.g. `path = []; t.inorder_traversal('D05.500', path); print(path)`).
"""
if __name__ == "__main__":
    code.interact(banner=banner, local=locals())
