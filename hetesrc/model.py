
import os
import math
import random
import torch
import torch.nn.functional as f
from tqdm import tqdm
from .network import Network
from .utils.metrics import registry as metrics
import numpy as np
import torch.nn as nn
from pytorch_transformers import AdamW, WarmupLinearSchedule, BertTokenizer,WarmupCosineSchedule

from torch.autograd import Variable


class Model:
    prefix = 'checkpoint'
    best_model_name = 'best.pt'

    def __init__ (self, args, state_dict=None):
        self.args = args

        # network
        self.network = Network(args)
        # print(self.network.blocks[0]['alignment'].alignparsing.ParserNet.parameters())
        # exit(0)
        self.device = torch.cuda.current_device() if args.cuda else torch.device('cpu')
        # print(torch.cuda.current_device())

        self.network.to(self.device)
        # optimizer
        self.params = list(filter(lambda x: x.requires_grad, self.network.parameters()))



        #self.opt = torch.optim.Adam(self.params, lr=args.lr, betas=(args.beta1, args.beta2))

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_vocal_dir)
        self.opt = AdamW (self.params, args.lr, correct_bias=False)
        num_total_steps = args.epochs * int(np.ceil(args.total_data / args.batch_size))
        num_warmup_steps = num_total_steps * args.warmup_rate
        self.scheduler = WarmupLinearSchedule(self.opt,warmup_steps=num_warmup_steps, t_total=num_total_steps)

        self.losses = nn.CrossEntropyLoss()
        # updates
        self.updates = state_dict['updates'] if state_dict else 0

        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['model'].keys()):
                if k not in new_state:
                    del state_dict['model'][k]
            self.network.load_state_dict(state_dict['model'])
            self.opt.load_state_dict(state_dict['opt'])

    def _update_schedule (self):
        if self.args.lr_decay_rate < 1.:
            args = self.args
            t = self.updates
            base_ratio = args.min_lr / args.lr
            if t < args.lr_warmup_steps:
                ratio = base_ratio + (1. - base_ratio) / max(1., args.lr_warmup_steps) * t
            else:
                ratio = max(base_ratio, args.lr_decay_rate ** math.floor((t - args.lr_warmup_steps) /
                                                                         args.lr_decay_steps))
            self.opt.param_groups[0]['lr'] = args.lr * ratio

            base_ratio = args.gcn_min_lr / args.gcn_lr
            if t < args.lr_warmup_steps:
                ratio = base_ratio + (1. - base_ratio) / max(1., args.lr_warmup_steps) * t
            else:
                ratio = max(base_ratio, args.lr_decay_rate ** math.floor((t - args.lr_warmup_steps) /
                                                                         args.lr_decay_steps))
            self.opt.param_groups[1]['lr'] = args.gcn_lr * ratio

    def update (self, batches):
        self.network.train()

        self.opt.zero_grad()
        inputs, target = self.process_data(batches)
        output = self.network(inputs)
        summary = self.network.get_summary()
        loss = self.get_loss(output, target)
        loss.backward()
        # grad_norm = (torch.nn.utils.clip_grad_norm_(self.params[0]["params"],
        #                                            self.args.grad_clipping) + torch.nn.utils.clip_grad_norm_(
        #     self.params[1]["params"], self.args.grad_clipping)).item()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params,
                                                                 self.args.grad_clipping)
        # assert grad_norm >= 0, 'encounter nan in gradients.'

        self.opt.step()
        self.scheduler.step()
        #self._update_schedule()


        self.updates += 1
        stats = {
            'updates': self.updates,
            'loss': loss.item(),
            'lr': self.opt.param_groups[0]['lr'],
            #'gcn_lr': self.opt.param_groups[1]['lr'],
            'gnorm': grad_norm,
            'summary': summary,
        }
        return stats

    def evaluate (self, data):
        self.network.eval()
        targets = []
        probabilities = []
        predictions = []
        losses = []

        for dev_id in tqdm(range(int(np.floor(len(data) / self.args.batch_size)))):
            min_index = dev_id * self.args.batch_size
            max_index = min(len(data), (dev_id + 1) * self.args.batch_size)
            inputs, target = self.process_data(data[min_index:max_index])
            with torch.no_grad():
                output = self.network(inputs)
                loss = self.get_loss(output, target)
                pred = torch.argmax(output, dim=1)
                prob = torch.nn.functional.softmax(output, dim=1)
                losses.append(loss.item())
                targets.extend(target.tolist())
                probabilities.extend(prob.tolist())
                predictions.extend(pred.tolist())

        outputs = {
            'target': targets,
            'prob': probabilities,
            'pred': predictions,
            'args': self.args,
        }

        stats = {
            'updates': self.updates,
            'loss': sum(losses[:-1]) / (len(losses) - 1) if len(losses) > 1 else sum(losses),
        }
        for metric in self.args.watch_metrics:
            if metric not in stats:  # multiple metrics could be computed by the same function
                stats.update(metrics[metric](outputs))
        assert 'score' not in stats, 'metric name collides with "score"'
        eval_score = stats[self.args.metric]
        stats['score'] = eval_score

        return eval_score, stats  # first value is for early stopping

    def predict (self, batch):
        self.network.eval()
        inputs, _ = self.process_data(batch)
        with torch.no_grad():
            output = self.network(inputs)
            output = torch.nn.functional.softmax(output, dim=1)
        return output.tolist()

    def process_data (self, batch):


        text_batch = []
        segment_batch = []
        mask_batch = []

        #coomatrix_batch = []
        coomatrix_batch = [[] for i in range(self.args.n_type)]
        phrase1_batch = []
        phrase2_batch = []

        node_num_batch = []
        text1_length = []
        text2_length = []

        batch_size = len(batch)
        last_graph_length = 0

        text_maxlen = 0
        # text2_maxlen = 0
        target = []
        for ids in range(batch_size):
            target.append(batch[ids]['target'])

            if len(node_num_batch) == 0:
                last_length = 0
                node_num_batch.append(len(batch[ids]['Nodelist']))
            else:
                last_length = node_num_batch[-1]
                node_num_batch.append(len(batch[ids]['Nodelist']) + last_length)

            length_text1 = len(batch[ids]['text1_id'])
            length_text2 = len(batch[ids]['text2_id'])

            text_batch.append([self.tokenizer.convert_tokens_to_ids("[CLS]")] + list(batch[ids]['text1_id']) + [
                self.tokenizer.convert_tokens_to_ids("[SEP]")]
                              + list(batch[ids]['text2_id']) + [self.tokenizer.convert_tokens_to_ids("[SEP]")])

            segment_batch.append([0] * (length_text1 + 2) + [1] * (length_text2 + 1))

            mask_batch.append([1] * len(text_batch[ids]))

            text_maxlen = max(text_maxlen, len(text_batch[ids]))

            text1_length.append(length_text1)
            text2_length.append(length_text2)

            text1_phrase = []
            phrase_num = 0
            for i in range(length_text1):
                phrases = batch[ids]['Nodelist'][i + length_text1]
                phrase_id = []
                for node in phrases:
                    phrase_id.append(phrase_num)
                    phrase_num += 1

                text1_phrase.append(phrase_id)
                if phrase_num == length_text1:
                    break

            text2_phrase = []
            phrase_num = 0
            for i in range(length_text2):
                phrases = batch[ids]['Nodelist'][i + length_text1 + len(text1_phrase) + length_text2]
                phrase_id = []
                for node in phrases:
                    phrase_id.append(phrase_num)
                    phrase_num += 1

                text2_phrase.append(phrase_id)
                if phrase_num == length_text2:
                    break

            phrase1_batch.append(text1_phrase)
            phrase2_batch.append(text2_phrase)
            for type in range(self.args.n_type):
                coomatrix_np = np.array(batch[ids]['Coomatrix'][type])
                coomatrix_batch[type].append(torch.LongTensor(coomatrix_np + last_length).to(self.device))  # [2,E]



        for ids in range(batch_size):
            padding = [0] * (text_maxlen - len(text_batch[ids]))
            text_batch[ids] += padding
            segment_batch[ids] += padding
            mask_batch[ids] += padding

        text_batch_tensor = torch.LongTensor(text_batch).to(self.device)
        segment_batch_tensor = torch.LongTensor(segment_batch).to(self.device)
        mask_batch_tensor = torch.LongTensor(mask_batch).to(self.device)

        coomatrix = []
        for type in range(self.args.n_type):
            coomatrix.append(torch.cat(coomatrix_batch[type], dim=-1))

        inputs = {
            # 'Nodelist':nodelist,
            'text1_phrase': phrase1_batch,
            'text2_phrase': phrase2_batch,
            'Coomatrix': coomatrix,
            'text_batch_tensor': text_batch_tensor,
            'segment_batch_tensor': segment_batch_tensor,
            'mask_batch_tensor': mask_batch_tensor,
            'node_num': node_num_batch,
            'batch_size': batch_size,
            'text1_length': text1_length,
            'text2_length': text2_length,

        }

        # if 'target' in batch:

        target = torch.LongTensor(target).to(self.device)

        # return inputs, target
        return inputs, target

    @staticmethod
    def get_loss (logits, target):
        return f.cross_entropy(logits, target)

    def save (self, states, name=None):
        if name:
            filename = os.path.join(self.args.summary_dir, name)
        else:
            filename = os.path.join(self.args.summary_dir, f'{self.prefix}_{self.updates}.pt')
        params = {
            'state_dict': {
                'model': self.network.state_dict(),
                'opt': self.opt.state_dict(),
                'updates': self.updates,
            },
            'args': self.args,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state()
        }
        params.update(states)
        if self.args.cuda:
            params['torch_cuda_state'] = torch.cuda.get_rng_state()
        torch.save(params, filename)

    @classmethod
    def load (cls, file):
        checkpoint = torch.load(file, map_location=(
            lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
        ))
        prev_args = checkpoint['args']

        # update args
        prev_args.output_dir = os.path.dirname(os.path.dirname(file))
        prev_args.summary_dir = os.path.join(prev_args.output_dir, prev_args.name)
        prev_args.cuda = prev_args.cuda and torch.cuda.is_available()
        return cls(prev_args, state_dict=checkpoint['state_dict']), checkpoint

    def num_parameters (self):
        num_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        return num_params