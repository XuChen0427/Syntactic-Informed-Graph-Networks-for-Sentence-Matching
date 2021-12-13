

import os
from pprint import pprint
from .model import Model
from .interface import Interface
from .utils.loader import load_data
from .MatrixVis import matrix_visualization

class Evaluator:
    def __init__(self, model_path, data_file):
        self.model_path = model_path
        self.data_file = data_file
        data = load_data(*os.path.split(self.data_file))
        self.model, checkpoint = Model.load(self.model_path)

        self.args = checkpoint['args']


        self.interface = Interface(self.args)
        self.batches = self.interface.pre_process(data, training=False)
    def evaluate(self):



        _, stats = self.model.evaluate(self.batches)
        pprint(stats)
