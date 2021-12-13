

import os
import numpy as np

def load_CooMatrix(graphdir,matrix_type):
    #CooMatrixFile = open(matrix_dir,'r')
    CooMatrix = {}
    test_index = 0
    for type in matrix_type:

        with open(os.path.join(graphdir,type+".out"),'r') as CooMatrixFile:
            for line in CooMatrixFile:
                if line != "\\n":
                    line = line.strip()
                    line_list = line.split("  ")
                    if len(line_list) == 2:
                        sentense_id,row_cols = line.split("  ")
                        row_col = row_cols.split(" ")
                        row = row_col[0].split(",")
                        col = row_col[1].split(",")
                        lenths = len(row)
                        for indexs in range(lenths):
                            row[indexs] = int(row[indexs])
                            col[indexs] = int(col[indexs])

                        coomatrixs = [row,col]
                        if int(sentense_id) not in CooMatrix.keys():
                            CooMatrix[int(sentense_id)] = []

                        CooMatrix[int(sentense_id)].append(coomatrixs)
                        test_index += 1
                    else:
                        print(type + " miss text: ", line)
    print("CooMatrix length:",test_index)
    print("Coomatrix format",CooMatrix[0])
    #CooMatrixFile.close()
    return CooMatrix

def load_Nodelist(Node_dir):
    NodeFile = open(Node_dir, 'r')
    NodeMatrix ={}
    test_index = 0
    for line in NodeFile:
        if line != "\\n":
            line = line.strip()
            sentense_id, row_coomatrix = line.split("  ")
            nodes = row_coomatrix.split(" ")
            node_list = []
            for node in nodes:
                node_str = node.split(",")

                for index in range(len(node_str)):
                    node_str[index] = int(node_str[index])
                node_list.append(node_str)
            NodeMatrix[int(sentense_id)] = node_list
            test_index += 1
    print("Nodelist length:", test_index)
    print("Nodelist format", NodeMatrix[0])
    NodeFile.close()
    return NodeMatrix


def load_data(data_dir, Coomatrix,Nodelist,filelength,n_type,max_len=50):
    sentence_id = 0
    data = []
    files = [os.path.join(data_dir,"test.txt"),os.path.join(data_dir,"dev.txt"),os.path.join(data_dir,"train.txt")]

    files_index = {"init":0}

    valid_sentence = 0

    for file in files:
        print(file)
        with open(file) as f:
            for line in f:
                datas = line.rstrip().split('\t')

                if len(datas) == filelength:

                    text1, text2, label = datas[0] , datas[1] , datas[2]

                    if text2 != "parisflatlist" and text2 != "AOSDHIADSOIHADSO DASODASHDASOH":
                        if len(Coomatrix[sentence_id]) == n_type:
                            if len(text1.split()) < max_len or len(text2.split()) < max_len:
                                node = Nodelist[sentence_id]
                                coomatirx = Coomatrix[sentence_id]

                                if len(node) == coomatirx[-1][0][-1] + 1:
                                    valid_sentence += 1
                                    data.append({
                                        'text1':text1,
                                        'text2':text2,
                                        'text1_id': Nodelist[sentence_id][-2],
                                        'text2_id': Nodelist[sentence_id][-1],
                                        'target': int(label),
                                        'Coomatrix':coomatirx,
                                        'Nodelist':node,
                                    })

                                else:
                                    print("GRAPH ERROR")
                                    print(coomatirx[0][0][-1])
                                    print(len(node))
                                    print(coomatirx)
                                    print(sentence_id)
                                    exit(0)

                            else:
                                print("exceed the maxlen",len(text1.split()),len(text2.split()))


                        else:
                            print("GRAPH MISSING: ",sentence_id)
                            #print(len(Coomatrix[sentence_id]))
                            #exit(0)



                        sentence_id += 1


                    else:
                        print("fileter:",text2)

        print("file length:",valid_sentence)
        files_index[file] = valid_sentence

    return data,files_index

