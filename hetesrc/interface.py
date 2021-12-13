

import os
import random
from .utils.loader import load_data,load_CooMatrix,load_Nodelist


class Interface:
    def __init__(self, args):
        self.args = args
        self.matrix_type = ["syntactic_edge","pos_edge","chunk_edge","wordagg_edge","chkagg_edge"]
        for i in range(self.args.n_type):
            self.matrix_type[i] = args.data_name+"_"+self.matrix_type[i]
        #self.Coomatrix = []
        self.Coomatrix = load_CooMatrix(args.graph_dir,self.matrix_type)
        self.Nodelist = load_Nodelist(os.path.join(args.graph_dir,args.node_dir))
        #self.data, self.file_index = load_data(self.args.data_dir, self.Coomatrix, self.Nodelist,self.args.filelength)


    def shuffle_data(self,data):
        random.shuffle(data)
        

