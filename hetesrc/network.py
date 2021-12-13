
import torch
import torch.nn.functional as F
from .modules import Module, ModuleList, ModuleDict
from pytorch_transformers import  BertModel, BertConfig,BertTokenizer
from .modules.prediction import registry as prediction
from .modules.prediction import Prediction_Bert,Prediction_Bert_GAT
from .modules.GCNS import *
import torch.nn as nn
class PhraseGating(nn.Module):
    def __init__ (self, args):  # code_length为fc映射到的维度大小
        super(PhraseGating, self).__init__()

        self.gate_fc =  nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(768, 768 * 2),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(768 * 2, 768),
            nn.Sigmoid()
        )


    def forward(self,x):
        return self.gate_fc(x)


class TextNet(nn.Module):
    def __init__(self,args): #code_length为fc映射到的维度大小
        super(TextNet, self).__init__()
        code_length = args.hidden_size * 4
        modelConfig = BertConfig.from_pretrained(args.bert_config_dir)
        self.textExtractor = BertModel.from_pretrained(
            args.bert_model_dir, config=modelConfig)

        self.textExtractor.train()
        embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)

        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features, output[0]

    def do_eval(self):
        self.textExtractor.eval()

    def do_train(self):
        self.textExtractor.train()



class Network(Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.bert_feature = TextNet(args)


        #SIGN-HGAT
        self.HeteGAT = HeteGAT(input_dim=768, hidden_dim=args.hidden_size, output_dim=args.hidden_size,n_type = args.n_type,
                      dropout_rate=self.dropout,layer_num=args.gat_layernum)
        #As for SIGN-GAT use:
        #self.HeteGAT = GAT(input_dim=768, hidden_dim=args.hidden_size, output_dim=args.hidden_size,n_type = args.n_type,
        #dropout_rate=self.dropout,layer_num=args.gat_layernum)

        self.fusions_node = nn.Linear(args.hidden_size*3,args.hidden_size)
        self.prediction = Prediction_Bert_GAT(args)
        self.bert_predict = Prediction_Bert(args)


    def forward(self, inputs):
        bert_feature,output = self.bert_feature(inputs['text_batch_tensor'],inputs['segment_batch_tensor'],inputs['mask_batch_tensor'])

        feature_matrix = []
        #
        # #output_gating = self.phrase_gate(output)
        for id in range(inputs['batch_size']):
            # print(id)

            text_a = output[id, 1:1+inputs['text1_length'][id], :]
            text_b = output[id, 2+inputs['text1_length'][id]:2+inputs['text1_length'][id]+inputs['text2_length'][id], :]

            # text_a_gate = output_gating[id, 1:1 + inputs['text1_length'][id], :]
            # text_b_gate = output_gating[id,
            #          2 + inputs['text1_length'][id]:2 + inputs['text1_length'][id] + inputs['text2_length'][id], :]
            # node_list = inputs['Nodelist']

            feature_matrix.append(text_a)
            # print(text_a.shape)

            for i in range(len(inputs['text1_phrase'][id])):
                # print(inputs['text1_phrase'][id][i])
                if len(inputs['text1_phrase'][id][i]) == 1:
                    feature_matrix.append(text_a[i, :].unsqueeze(0))
                    #feature_matrix.append((text_a[i, :]*text_a_gate[i,:]).unsqueeze(0))

                else:
                    feature_matrix.append(self.poolings(text_a[inputs['text1_phrase'][id][i][0]:inputs['text1_phrase'][id][i][-1]+1,:]))
                    #feature_matrix.append(torch.mean(text_a[inputs['text1_phrase'][id][i][0]:inputs['text1_phrase'][id][i][-1]+1,:],dim=0,keepdim=True))
                    # feature_matrix.append(torch.sum(
                    #     text_a[inputs['text1_phrase'][id][i][0]:inputs['text1_phrase'][id][i][-1] + 1, :] *
                    #     text_a_gate[inputs['text1_phrase'][id][i][0]:inputs['text1_phrase'][id][i][-1] + 1, :],dim=0,keepdim=True))

            feature_matrix.append(text_b)

            for i in range(len(inputs['text2_phrase'][id])):
                if len(inputs['text2_phrase'][id][i]) == 1:
                    feature_matrix.append(text_b[i, :].unsqueeze(0))
                    #feature_matrix.append((text_b[i, :] * text_b_gate[i, :]).unsqueeze(0))

                else:
                    feature_matrix.append(self.poolings(text_b[inputs['text2_phrase'][id][i][0]:inputs['text2_phrase'][id][i][-1]+1,:]))
                    # feature_matrix.append(
                    #     torch.mean(text_b[inputs['text2_phrase'][id][i][0]:inputs['text2_phrase'][id][i][-1] + 1, :], dim=0,
                    #                keepdim=True))
                    # feature_matrix.append(torch.sum(
                    #     text_b[inputs['text2_phrase'][id][i][0]:inputs['text2_phrase'][id][i][-1] + 1, :] *
                    #     text_b_gate[inputs['text2_phrase'][id][i][0]:inputs['text2_phrase'][id][i][-1] + 1, :],dim=0,keepdim=True))

            feature_matrix.append(self.poolings(text_a))

            feature_matrix.append(self.poolings(text_b))

        feature_matrix = torch.cat(feature_matrix, dim=0)


        GCN_matrix = self.HeteGAT(x=feature_matrix, edge_index=inputs['Coomatrix'])

        gat_a = []
        gat_b = []

        word_a = []
        phrase_a = []

        word_b = []
        phrase_b = []
        last_node_num = 0
        for id in range(inputs['batch_size']):
            gat_a.append(GCN_matrix[inputs['node_num'][id] - 2, :].unsqueeze(0))
            gat_b.append(GCN_matrix[inputs['node_num'][id] - 1, :].unsqueeze(0))

            word_a_pos = last_node_num  # 0
            phrase_a_pos = last_node_num + inputs['text1_length'][id]  # 0 + 3 = 3
            word_b_pos = last_node_num + inputs['text1_length'][id] + len(inputs['text1_phrase'][id])  # 0 + 3 + 2 = 5
            phrase_b_pos = last_node_num + inputs['text1_length'][id] + len(inputs['text1_phrase'][id]) + \
                           inputs['text2_length'][id]  # 0 + 3 + 2 + 3 = 8

            # node num : 12
            word_a.append(self.poolings(GCN_matrix[word_a_pos:phrase_a_pos]))
            phrase_a.append(self.poolings(GCN_matrix[phrase_a_pos:word_b_pos]))
            word_b.append(self.poolings(GCN_matrix[word_b_pos:phrase_b_pos]))
            phrase_b.append(self.poolings(GCN_matrix[phrase_b_pos:inputs['node_num'][id] - 2]))

            last_node_num = inputs['node_num'][id]

        a = torch.cat(gat_a, dim=0)
        b = torch.cat(gat_b, dim=0)

        word_a = torch.cat(word_a, dim=0)
        phrase_a = torch.cat(phrase_a, dim=0)
        word_b = torch.cat(word_b, dim=0)
        phrase_b = torch.cat(phrase_b, dim=0)

        a_s = self.fusions_node(torch.cat([a, word_a, phrase_a], dim=-1))
        b_s = self.fusions_node(torch.cat([b, word_b, phrase_b], dim=-1))

        a_s = F.dropout(a_s, p=self.dropout, training=self.training)
        b_s = F.dropout(b_s, p=self.dropout, training=self.training)
        a_s = F.relu(a_s)
        b_s = F.relu(b_s)
        result = self.prediction(a_s,b_s,bert_feature)

        return result
        #return self.prediction(a_s,b_s)

    def poolings(self,x):
        return x.max(dim=0)[0].unsqueeze(0)





