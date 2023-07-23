# Syntactic-Informed-Graph-Networks-for-Sentence-Matching (TOIS 2023)
## Xu Chen, Renming University of China, Gaoling School of Artificial intelligence
Any question, please mail to xc_chen@ruc.edu.cn
Implementation of Syntactic-Informed Graph Networks for Sentence Matching of TOIS 2023

# Note: 
We only provide the small dataset: Scitail, 
For QQP and SNLI dataset, please download from https://www.kaggle.com/c/quora-question-pairs and https://nlp.stanford.edu/projects/snli

please download the BERT/RoBERTa model to ~/modeling_bert/
url: https://github.com/huggingface/transformers

please download the stanford CoreNLP parser to any dir:
url: https://stanfordnlp.github.io/CoreNLP/

## 1. enter the stanford CoreNLP parser dir:
```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 1500000
```

## 2. change the data dir to your own at configs/ and set your parameters

## 3. Preprepare the syntactic structures: 
```bash
python prepare_graph_bert.py configs/scitail.json5
```

## 4. train the model:
```bash
python train.py configs/scitail.json5
```

##For citation, please cite the following bib
```
@article{10.1145/3609795,
author = {Xu, Chen and Xu, Jun and Dong, Zhenhua and Wen, Ji-Rong},
title = {Syntactic-Informed Graph Networks for Sentence Matching},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1046-8188},
url = {https://doi.org/10.1145/3609795},
doi = {10.1145/3609795},
abstract = {Matching two natural language sentences is a fundamental problem in both Nature Language Processing (NLP) and Information Retrieval (IR). Preliminary studies have shown that the syntactic structures help improve the matching accuracy and different syntactic structures in natural language are complementary to sentence semantic understanding. Ideally, a matching model would leverage all the syntactic information. Existing models, however, are only able to combine limited (usually one) types of syntactic information due to the complex and heterogeneous nature of the syntactic information. To deal with the problem, we propose a novel matching model, which formulates sentence matching as a representation learning task on a syntactic-informed heterogeneous graph. The model, referred to as Syntactic-Informed Graph Network (SIGN), firstly constructs a heterogeneous matching graph based on the multiple syntactic structures of two input sentences. Then graph attention network algorithm is applied to the matching graph to learn the high-level representations of the nodes. With the help of the graph learning framework, the multiple syntactic structures, as well as the word semantics, can be represented and interacted in the matching graph, and therefore collectively enhance the matching accuracy. We conducted comprehensive experiments on three public datasets. The results demonstrated that SIGN outperformed the state-of-the-art and it also can discriminate the sentences in an interpretable way.},
note = {Just Accepted},
journal = {ACM Trans. Inf. Syst.},
month = {jul},
keywords = {syntactic structures, sentence matching, graph learning}
}
```



