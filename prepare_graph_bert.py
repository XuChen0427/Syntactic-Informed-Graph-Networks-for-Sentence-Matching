import nltk
import numpy as np
import os
import sys
import json5
from pprint import pprint
from hetesrc.utils import params
# from src.utils import vocab
from pytorch_transformers import BertTokenizer
import copy
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag import StanfordPOSTagger
from nltk.internals import find_jars_within_path
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse import CoreNLPParser
from tqdm import tqdm,trange
import json
#import scipy.sparse as sparse


#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 1500000
class TextGraphBuilder():
    def __init__(self,args):

        self.grammer = r"""
        JJRB: {<RB><JJ>}
        NSP: {<DT|PR.*| JJ.*|NN.*>+}          # Chunk sequences of DT, JJ, NN
        
        NJP: {<JJ.*><NSP>}
        PPP: {<IN><NS.*>}               # Chunk prepositions followed by NP
        
        VRTO: {<VB.*>+<TO>+<VB.*>+}
        VRTON:{<VB.*><TO><NSP>}
        VRING:{<VB.*>+<VB.*>+}
        VRBf:{<RB.*>+<VB.*>+}
        VRBS:{<VB.*>+<RB.*>+}
        VBP: {<V.*><NSP|CLAUSE>+} # Chunk verbs and their arguments
        CLAUSE: {<NSP><VBP>}           # Chunk NP, VP
        """


        #self.dependency_parser = StanfordDependencyParser(path_to_jar=myp2, path_to_models_jar=myp)
        self.dependency_parser = CoreNLPDependencyParser(url='http://localhost:9000')

        #self.postager = StanfordPOSTagger(model_filename=poster,path_to_jar=tagger_path)
        self.postager =  CoreNLPParser(url='http://localhost:9000', tagtype='pos')
        self.nertager =  CoreNLPParser(url='http://localhost:9000', tagtype='ner')


        self.data_dir = args.data_dir

        #print(args.graph_file)
        self.cp = nltk.RegexpParser(self.grammer)

        self.args = args
        self.data = self.load_data()
        self.tokenizer = BertTokenizer.from_pretrained ('modeling_bert/bert-base-uncased-vocab.txt')
       
        self.word2tags = {}
        self.phrase2tags = {}
        self.wordtags_count = 0
        self.phrasetags_count = 0

        self.edges_type = ["syntactic_edge","pos_edge","ner_edge","chunk_edge","wordagg_edge","chkagg_edge"]


    def get_nltktree_w(self,t):
        if str(type(t)) == "<class 'tuple'>":
            #return {filterpos(t[1]):t[0]}
            return [t[0]]
        else:

            w_list = []
            for w in t:
                if str(type(w)) == "<class 'tuple'>":
                    w_list.extend([w[0]])
                else:
                    w_list.extend(self.get_nltktree_w(w))

            #return {label:w_list}
            return w_list

    def get_nltktree_label(self,t):
        if str(type(t)) == "<class 'tuple'>":
            return self.filterpos(t[1])
        else:
            #return self.filterpos(t.label())
            return t.label()

    def DependencyParse(self,s):

        result = self.dependency_parser.parse(s)
        dependency_list = []
        for unit in list(result.__next__().triples()):
            dependency_list.append((unit[0][0],unit[2][0]))

        return dependency_list

    def Parse(self,s):
        pos_tag = self.postager.tag(s)
        ner_tag = self.nertager.tag(s)

        filtered_tag = list(copy.copy(pos_tag))
        for i in range(len(filtered_tag)):
            filtered_tag[i] = list(filtered_tag[i])
            filtered_tag[i][1] = self.filterpos(filtered_tag[i][1])

        ner_tag_list = list(ner_tag)

        #print(filtered_tag)
        parse_tree = self.cp.parse(pos_tag)
        phrase_list = []
        for sub_tree in parse_tree:
            phrase_list.append([self.get_nltktree_w(sub_tree),self.get_nltktree_label(sub_tree)])

        return filtered_tag,phrase_list,ner_tag_list

    def WriteGraphIntoFile(self):
        #graph_file = open(os.path.join(self.args.graph_dir,self.args.graph_file),"w")
        graph_files = {k:open(os.path.join(self.args.graph_dir,self.args.graph_file+"_"+k+".out"),'w') for k in self.edges_type}
        node_file = open(os.path.join(self.args.graph_dir,self.args.node_file),"w")
        #label_file =  open(os.path.join(self.args.graph_dir,"QQP_labels_bert_DAG2.out"),"w")
        #index_file = open(os.path.join(self.args.graph_dir, "QQP_index_DAG.out"), "w")

        graphs = []
        index_list = []
        for id in trange(0,len(self.data)):
        #or id in trange(0, 2):
            adjmatrix,nodes,labels,indexes = self.get_adjmatrix(self.data[id])
            # label_file.write(str(self.data[id]['sentence_id']) + "  ")
            # for label_id in range(len(labels)):
            #     label_file.write(str(labels[label_id]))
            #     if label_id != len(labels) -1:
            #         label_file.write(",")
            #
            # label_file.write("\n")

            index_list.append(indexes)

            #graphs.append(adjmatrix)
            node_file.write(str(self.data[id]['sentence_id'])+"  ")
            for node in nodes:
                for index,element in enumerate(node):
                    node_file.write(str(element))
                    if index != len(node)-1:
                        node_file.write(",")
                node_file.write(" ")
            node_file.write("\n")

            for edge_type in self.edges_type:
                edges = len(adjmatrix[edge_type][0])

                graph_files[edge_type].write(str(self.data[id]['sentence_id'])+"  ")
                for i in range(edges):
                    graph_files[edge_type].write(str(adjmatrix[edge_type][0][i]))
                    if i != edges - 1:
                        graph_files[edge_type].write(",")
                graph_files[edge_type].write(" ")
                for i in range(edges):
                    graph_files[edge_type].write(str(adjmatrix[edge_type][1][i]))
                    if i != edges - 1:
                        graph_files[edge_type].write(",")

                graph_files[edge_type].write("\n")

        for edge_type in self.edges_type:
            graph_files[edge_type].close()
        print("complete!")

        #index_list = np.array(index_list)
        #with open(os.path.join(self.args.graph_dir, "QQP_index_DAG.out"), "w") as f:
        # np.save(os.path.join(self.args.graph_dir, "QQP_bert_index_DAG2"),index_list)
        # with open(os.path.join(self.args.graph_dir, "QQP_bert_word2tag2.json"), "w") as f:
        #     json.dump(self.word2tags,f)
        # with open(os.path.join(self.args.graph_dir, "QQP_bert_phrase2tags2.out"), "w") as f:
        #     json.dump(self.phrase2tags, f)

        #graph_file.close()
        #label_file.close()
        node_file.close()

    def get_adjmatrix(self,sample):

        s1,s2  = self.process_sample(sample)


        tag_s1,phrase_s1,ner_s1 = self.Parse(s1)

        tag_s2,phrase_s2,ner_s2 = self.Parse(s2)
        # print(ner_s1)
        # print(ner_s2)
        # exit(0)


        lables = []
        indexes = [len(tag_s1),len(phrase_s1),len(tag_s2),len(phrase_s2)]
        dependency_s1 = self.DependencyParse(s1)

        dependency_s2 = self.DependencyParse(s2)



        Adjmatrix = self.BuildAdjMatrix(tag_s1,tag_s2,ner_s1,ner_s2,phrase_s1,phrase_s2,dependency_s1,dependency_s2)


        node = []

        s1_list = []
        s2_list = []
        for w in tag_s1:
            node.append([self.tokenizer.convert_tokens_to_ids(w[0])])
            s1_list.append(self.tokenizer.convert_tokens_to_ids(w[0]))
            tag = w[1]
            if tag not in self.word2tags.keys():
                self.word2tags[tag] = self.wordtags_count
                self.wordtags_count += 1
            lables.append(self.word2tags[tag])



        for p in phrase_s1:
            p_list = []
            for ps in p[0]:
                p_list.append(self.tokenizer.convert_tokens_to_ids(ps))

            tag = p[1][:1]
            if tag not in self.phrase2tags.keys():
                self.phrase2tags[tag] = self.phrasetags_count
                self.phrasetags_count += 1
            lables.append(self.phrase2tags[tag])

            node.append(p_list)

        for w in tag_s2:
            node.append([self.tokenizer.convert_tokens_to_ids(w[0])])
            s2_list.append(self.tokenizer.convert_tokens_to_ids(w[0]))
            tag = w[1]
            if tag not in self.word2tags.keys():
                self.word2tags[tag] = self.wordtags_count
                self.wordtags_count += 1
            lables.append(self.word2tags[tag])

        for p in phrase_s2:
            p_list = []
            for ps in p[0]:
                p_list.append(self.tokenizer.convert_tokens_to_ids(ps))

            tag = p[1][:1]
            if tag not in self.phrase2tags.keys():
                self.phrase2tags[tag] = self.phrasetags_count
                self.phrasetags_count += 1
            lables.append(self.phrase2tags[tag])
            node.append(p_list)

        node.append(s1_list)
        node.append(s2_list)

        return Adjmatrix,node,lables,indexes

    def BuildAdjMatrix(self,tag_s1,tag_s2,ner_s1,ner_s2,phrase_s1,phrase_s2,dependency_s1,dependency_s2):
        '''

        :param tag_s1: [(wordi,pos_tagi)]
        :param tag_s2: [(wordi,pos_tagi)]
        :param phrase_s1: [(phrase_i,labeli)]
        :param phrase_s2: [(phrase_i,labeli)]
        :param dependency_s1: [(wordi,wordj)]
        :param dependency_s2: [(wordi,wordj)]
        :return: AdjMatrix

        the graph build on this:
        In single sentence words' vex based on the dependency parser
        In two match sentence, the words' vex based on the same pos_tag

        In single sentence phrases's vex based on the sentence order
        in two match sentence, the phrases' vex based on the same label

        the whole sentence is fully connected with the its own words and phrases
        '''

        word_num = len(tag_s1) + len(tag_s2)
        phrase_num = len(phrase_s1) + len(phrase_s2)
        N = word_num + phrase_num + 2
       # N = word_num + 2

        #AdjMatrix = np.zeros((N,N))
        #the Matrix order is s1_wordi s1_phrasei s1 s2_wordi s2_phrasei s2

        s1_pos = N-2
        s2_pos = N-1

        s1_w_pos = 0
        s1_p_pos = len(tag_s1)

        s2_w_pos = s1_p_pos + len(phrase_s1)
        #s2_w_pos = len(tag_s1)
        s2_p_pos = s2_w_pos + len(tag_s2)

        row_list = []
        col_list = []

        #edges_type = ["syntactic_edge","pos_edge","ner_edge","chunk_edge","wordagg_edge","chkagg_edge"]
        Adjmatrix_type = {k:[[],[]] for k in self.edges_type}
        #########################################
            #start to build syntactic edge#
        ########################################
        for i in range(s1_w_pos,s1_p_pos):
            for j in range(s1_w_pos,s1_p_pos):
                if (tag_s1[i-s1_w_pos][0],tag_s1[j-s1_w_pos][0]) in dependency_s1:

                    Adjmatrix_type["syntactic_edge"][0].append(i)
                    Adjmatrix_type["syntactic_edge"][0].append(j)
                    Adjmatrix_type["syntactic_edge"][1].append(j)
                    Adjmatrix_type["syntactic_edge"][1].append(i)

                if i == j:
                    Adjmatrix_type["syntactic_edge"][0].append(i)
                    Adjmatrix_type["syntactic_edge"][1].append(j)


        for i in range(s2_w_pos,s2_p_pos):
            for j in range(s2_w_pos,s2_p_pos):
                if (tag_s2[i-s2_w_pos][0],tag_s2[j-s2_w_pos][0]) in dependency_s2:
                    Adjmatrix_type["syntactic_edge"][0].append(i)
                    Adjmatrix_type["syntactic_edge"][0].append(j)
                    Adjmatrix_type["syntactic_edge"][1].append(j)
                    Adjmatrix_type["syntactic_edge"][1].append(i)

                if i == j:
                    Adjmatrix_type["syntactic_edge"][0].append(i)
                    Adjmatrix_type["syntactic_edge"][1].append(j)


        #########################################
        # start to build two sentence pos edge#
        ########################################
        for i in range(s1_w_pos,s1_p_pos):
            for j in range(s2_w_pos,s2_p_pos):
                # for i in range(s1_w_pos, s2_w_pos):
                #     for j in range(s2_w_pos, s1_pos):
                if tag_s1[i-s1_w_pos][1] == tag_s2[j-s2_w_pos][1]:
                    Adjmatrix_type["pos_edge"][0].append(i)
                    Adjmatrix_type["pos_edge"][0].append(j)
                    Adjmatrix_type["pos_edge"][1].append(j)
                    Adjmatrix_type["pos_edge"][1].append(i)

        for i in range(s1_p_pos, s2_w_pos):
            for j in range(s2_p_pos, s1_pos):
                if phrase_s1[i-s1_p_pos][1][:1] == phrase_s2[j-s2_p_pos][1][:1]:
                    Adjmatrix_type["pos_edge"][0].append(i)
                    Adjmatrix_type["pos_edge"][0].append(j)
                    Adjmatrix_type["pos_edge"][1].append(j)
                    Adjmatrix_type["pos_edge"][1].append(i)

        #########################################
        #start to build ner edge#
        ########################################
        ner_invalid_type = ['O','NUMBER']

        for i in range(s1_w_pos,s1_p_pos):
            for j in range(s2_w_pos,s2_p_pos):
                # for i in range(s2_w_pos, s1_pos):
                #         #     for j in range(s2_w_pos, s1_pos):
                if ner_s1[i-s1_w_pos][1] != 'O' and (ner_s1[i-s1_w_pos][1] == ner_s2[j-s2_w_pos][1]):
                    Adjmatrix_type["ner_edge"][0].append(i)
                    Adjmatrix_type["ner_edge"][0].append(j)
                    Adjmatrix_type["ner_edge"][1].append(j)
                    Adjmatrix_type["ner_edge"][1].append(i)

                if i == j:
                    Adjmatrix_type["ner_edge"][0].append(i)
                    Adjmatrix_type["ner_edge"][1].append(j)

        # print(ner_s1)
        # print(ner_s2)
        # print(Adjmatrix_type['ner_edge'])
        # exit(0)

        #########################################
        # start to build chunk edge#
        ########################################
        for i in range(s1_w_pos, s1_p_pos):
            for j in range(s1_p_pos, s2_w_pos):
                if tag_s1[i-s1_w_pos][0] in phrase_s1[j-s1_p_pos][0]:
                    Adjmatrix_type["chunk_edge"][0].append(i)
                    #row_list.append(j)
                    Adjmatrix_type["chunk_edge"][1].append(j)
                    #col_list.append(i)

        for i in range(s2_w_pos, s2_p_pos):
            for j in range(s2_p_pos, s1_pos):
                if tag_s2[i - s2_w_pos][0] in phrase_s2[j - s2_p_pos][0]:
                    Adjmatrix_type["chunk_edge"][0].append(i)
                    #row_list.append(j)
                    Adjmatrix_type["chunk_edge"][1].append(j)
                    #col_list.append(i)



        #########################################
        # start to build two wordagg edge#
        ########################################
        for i in range(s1_w_pos,s1_p_pos):
            # AdjMatrix[i,s1_pos] = 1
            # AdjMatrix[s1_pos,i] = 1

            Adjmatrix_type["wordagg_edge"][0].append(i)
            #row_list.append(s1_pos)
            Adjmatrix_type["wordagg_edge"][1].append(s1_pos)
            #col_list.append(i)

        Adjmatrix_type["wordagg_edge"][0].append(s1_pos)
        Adjmatrix_type["wordagg_edge"][1].append(s1_pos)

        for i in range(s2_w_pos,s2_p_pos):
            # AdjMatrix[i,s1_pos] = 1
            # AdjMatrix[s1_pos,i] = 1

            Adjmatrix_type["wordagg_edge"][0].append(i)
            #row_list.append(s1_pos)
            Adjmatrix_type["wordagg_edge"][1].append(s2_pos)
            #col_list.append(i)

        Adjmatrix_type["wordagg_edge"][0].append(s2_pos)
        Adjmatrix_type["wordagg_edge"][1].append(s2_pos)


        #########################################
        # start to build two chkagg_edge#
        ########################################
        for i in range(s1_p_pos,s2_w_pos):
            Adjmatrix_type["chkagg_edge"][0].append(i)
            #row_list.append(s2_pos)
            Adjmatrix_type["chkagg_edge"][1].append(s1_pos)
            #col_list.append(i)

        Adjmatrix_type["chkagg_edge"][0].append(s1_pos)
        Adjmatrix_type["chkagg_edge"][1].append(s1_pos)

        for i in range(s2_p_pos,s1_pos):
            Adjmatrix_type["chkagg_edge"][0].append(i)
            #row_list.append(s2_pos)
            Adjmatrix_type["chkagg_edge"][1].append(s2_pos)
            #col_list.append(i)

        Adjmatrix_type["chkagg_edge"][0].append(s2_pos)
        Adjmatrix_type["chkagg_edge"][1].append(s2_pos)

        ###check Matrix######
        # for i in range(s1_w_pos,s1_p_pos):
        #     print(tag_s1[i-s1_w_pos][0],end='')
        #     print(" ", end='')
        #
        # for i in range(s1_p_pos,s2_w_pos):
        #     print(phrase_s1[i-s1_p_pos][0],end='')
        #     print(" ", end='')
        #
        #
        #
        # for i in range(s2_w_pos, s2_p_pos):
        #     print(tag_s2[i-s2_w_pos][0], end='')
        #     print(" ", end='')
        #
        # for i in range(s2_p_pos, s1_pos):
        #     print(phrase_s2[i-s2_p_pos][0], end='')
        #     print(" ", end='')
        #
        # print("s1 ", end='')
        # print("s2 ")
        #
        # for i in range(s1_w_pos, s1_p_pos):
        #     print(tag_s1[i-s1_w_pos][1], end='')
        #     print(" ", end='')
        #
        # for i in range(s1_p_pos, s2_w_pos):
        #     print(phrase_s1[i-s1_p_pos][1], end='')
        #     print(" ", end='')
        #
        #
        #
        # for i in range(s2_w_pos, s2_p_pos):
        #     print(tag_s2[i-s2_w_pos][1], end='')
        #     print(" ", end='')
        #
        # for i in range(s2_p_pos, s1_pos):
        #     print(phrase_s2[i-s2_p_pos][1], end='')
        #     print(" ", end='')
        #
        # print("sentence ", end='')
        # print("sentence ")
        #

        return Adjmatrix_type




    def filterpos(self,pos):
        return pos[:2]

    def filterword(self,sentence):

            filtered_s = []
            for word in sentence:
                if word != "\\":
                    filtered_s.append(word)
                # else:
                #     print("filter:",word)

            return filtered_s


    def load_data (self, split=None):
        sentence_id = 0
        data_dir = self.data_dir
        data = []
        if split is None:
            files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.txt')]
        else:
            if not split.endswith('.txt'):
                split += '.txt'
            files = [os.path.join(data_dir, f'{split}')]
        print(files)
        # exit(0)
        #files = [os.path.join(data_dir,'test.txt')]
        for file in files:
            print(file)
            with open(file) as f:
                for line in f:
                    datas = line.rstrip().split('\t')
                    if len(datas) == self.args.file_length:
                        text1, text2, label = datas[0],datas[1],datas[2]
#text2:parisflatlist
#text2:AOSDHIADSOIHADSO DASODASHDASOH
                        if text2 != "parisflatlist" and text2 != "AOSDHIADSOIHADSO DASODASHDASOH":
                            data.append({
                                'text1': text1,
                                'text2': text2,
                                'target': label,
                                'sentence_id': sentence_id
                            })
                            sentence_id += 1

                        else:
                            print("filter:",text2)

        return data

    def process_sample (self, sample):
        text1 = sample['text1']
        text2 = sample['text2']
        if self.args.lower_case:
            text1 = text1.lower()
            text2 = text2.lower()

        text1 = self.postager.tokenize(text1)
        text2 = self.postager.tokenize(text2)

        return self.filterword(text1),self.filterword(text2)


def main():
    argv = sys.argv

    if len(argv) == 2:
        arg_groups = params.parse(sys.argv[1])
        for args, config in arg_groups:
            graphbuilder = TextGraphBuilder(args)
            #print(graphbuilder.data[0])

            graphbuilder.WriteGraphIntoFile()


            exit(0)
    elif len(argv) == 3 and '--dry' in argv:
        argv.remove('--dry')
        arg_groups = params.parse(sys.argv[1])
        pprint([args.__dict__ for args, _ in arg_groups])
    else:
        print('Usage: "python train.py configs/xxx.json5"')

if __name__ == "__main__":
    main()