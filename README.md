BMET is a Transformer model for fine-tuning biomedical word embedding that are jointly trained on the concept (MeSH, Medical Subject Heading) relatedness classification task. We provide our embedding for public use for any downstream application or research endeavors.

BMET-all

This contains all the MeSH descriptors and a subset of supplementary concepts that meets a frequency threshold. Vocabulary is divided into three sections: (1) BERT special tokens ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]') (2) MeSH codes (3) English words in descending frequency order. (vocabulary size is 333,301)

BMET-meshes

These files contain only MeSH codes. (vocabulary size is 45,015)

BMET-words

These files contain only English words. (vocabulary size is 288,281)

