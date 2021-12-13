# Syntactic-Informed-Graph-Networks-for-Sentence-Matching
## Xu Chen, Renming University of China, Gaoling School of Artificial intelligence
Any question, please mail to xc_chen@ruc.edu.cn
Implementation of Syntactic-Informed Graph Networks for Sentence Matching

# Note: 
We only provide the small dataset: Scitail, 
For QQP and SNLI dataset, please download from https://www.kaggle.com/c/quora-question-pairs and https://nlp.stanford.edu/projects/snli

please download the BERT/RoBERTa model to ~/modeling_bert/
url: https://github.com/huggingface/transformers

please download the stanford CoreNLP parser to any dir:
url: https://stanfordnlp.github.io/CoreNLP/

## 1. enter the stanford CoreNLP parser dir:
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 1500000

## 2. change the data dir to your own at configs/ and set your parameters

## 3. Preprepare the syntactic structures: 
python prepare_graph_bert.py configs/scitail.json5

## 4. train the model:
python train.py configs/scitail.json5
