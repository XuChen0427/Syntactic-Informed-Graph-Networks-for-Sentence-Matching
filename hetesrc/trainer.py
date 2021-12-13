import os
import random
import json5
import torch
from datetime import datetime
from pprint import pformat
from .utils.loader import load_data
from .utils.logger import Logger
from .model import Model
from .interface import Interface
import numpy as np


class Trainer:
    def __init__(self, args):
        self.args = args
        self.log = Logger(self.args)

    def train(self):
        start_time = datetime.now()
        model, interface, states = self.build_model()
        Coomatrix,Nodelist = interface.Coomatrix,interface.Nodelist

        train = {}
        dev = {}
        all_data,file_index = load_data(self.args.data_dir,Coomatrix,Nodelist,self.args.filelength,self.args.n_type)

        last_num = 0
        for key in file_index.keys():

            if "train" in key:
                train = all_data[last_num:file_index[key]]

            elif self.args.eval_file in key:
                dev = all_data[last_num:file_index[key]]

            last_num = file_index[key]

        if len(train) * len(dev) == 0:
            raise KeyError("the graph has some ERROR please check")

        self.log(f'train ({len(train)}) | {self.args.eval_file} ({len(dev)})')
        self.log('setup complete: {}s.'.format(str(datetime.now() - start_time).split(".")[0]))

        try:
            for epoch in range(states['start_epoch'], self.args.epochs + 1):
                states['epoch'] = epoch
                self.log.set_epoch(epoch)

                interface.shuffle_data(train)
                for batch_id in range(int(np.floor(len(train)/self.args.batch_size))):
                    min_index = batch_id * self.args.batch_size
                    max_index = min(len(train),(batch_id+1)*self.args.batch_size)
                    stats = model.update(train[min_index:max_index])
                    self.log.update(stats)

                    if batch_id % 1000 == 0:
                        self.log.newline()
                        score, dev_stats = model.evaluate(dev)
                        if score > states['best_eval']:
                            states['best_eval'], states['best_epoch'], states['best_step'] = score, epoch, model.updates
                            if self.args.save:
                                model.save(states, name=model.best_model_name)
                        self.log.log_eval(dev_stats)
                        if self.args.save_all:
                            model.save(states)
                            model.save(states, name='last')

                model.save(states, name="epoch_"+str(epoch)+"_"+model.best_model_name)


                self.log.newline()
            self.log('Training complete.')
        except KeyboardInterrupt:
            self.log.newline()
            self.log(f'Training interrupted. Stopped early.')
        except EarlyStop as e:
            self.log.newline()
            self.log(str(e))
        self.log(f'best dev score {states["best_eval"]} at step {states["best_step"]} '
                 f'(epoch {states["best_epoch"]}).')
        self.log(f'best eval stats [{self.log.best_eval_str}]')
        training_time = str(datetime.now() - start_time).split('.')[0]
        self.log(f'Training time: {training_time}.')
        states['start_time'] = str(start_time).split('.')[0]
        states['training_time'] = training_time
        return states

    def build_model(self):
        states = {}
        interface = Interface(self.args)

        #self.log(f'#classes: {self.args.num_classes}; #vocab: {self.args.num_vocab}')
        if self.args.seed:
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            if self.args.cuda:
                torch.cuda.manual_seed(self.args.seed)
            if self.args.deterministic:
                torch.backends.cudnn.deterministic = True

        model = Model(self.args)

        states['start_epoch'] = 1
        states['best_eval'] = 0.
        states['best_epoch'] = 0
        states['best_step'] = 0

        self.log(f'trainable params: {model.num_parameters():,d}')
       # self.log(f'trainable params (exclude embeddings): {model.num_parameters(exclude_embed=True):,d}')
        #validate_params(self.args)
        with open(os.path.join(self.args.summary_dir, 'args.json5'), 'w') as f:
            json5.dump(self.args.__dict__, f, indent=2)
        self.log(pformat(vars(self.args), indent=2, width=120))
        return model, interface, states


class EarlyStop(Exception):
    pass
