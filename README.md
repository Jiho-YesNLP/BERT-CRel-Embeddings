**BERT-CRel** is a transformer model for fine-tuning biomedical word embeddings that are jointly learned along with concept embeddings using a pre-training phase with fastText and a fine-tuning phase with a transformer setup. The goal is to provide high quality pre-trained biomedical embeddings that can be used in any downstream task by the research community. This repository contains the code used to implement the BERT-CRel methods and generate the embeddings. The corpus used for BERT-CRel contains biomedical citations from PubMed and the concepts are from the Medical Subject Headings (MeSH codes) terminology used to index citations. 

All our fine-tuned embeddings can be obtained from Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4383195.svg)](https://doi.org/10.5281/zenodo.4383195)

**BERT-CRel-all**

This contains word embeddings and all the MeSH descriptors and a subset of supplementary concepts (each of which meet a frequency threshold). Vocabulary is divided into three sections: (1) BERT special tokens (2) MeSH codes (3) English words in descending frequency order. (vocabulary size is 333,301)

**BERT-CRel-MeSH**

These files contain only MeSH code embeddings. (vocabulary size is 45,015)

**BERT-CRel-words**

These files contain only English word embeddings. (vocabulary size is 288,281) 
