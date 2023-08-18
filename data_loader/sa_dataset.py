import os
from torch.utils.data.dataset import Dataset
import json
import pickle
import torch
import random
import numpy as np


class AmazonMultiProcessor(Dataset):
    def __init__(self):
        super().__init__()
        self.labels = [["negative", 'bad'], ["positive", 'good']]

    def get_examples(self, data_dir, split, domains):
        examples = []
        for domain_idx, domain in enumerate(domains):
            tmp = []
            with open(os.path.join(data_dir, domain, '{}.txt'.format(split)), 'r', encoding='utf-8') as fin:
                for idx, line in enumerate(fin.readlines()):
                    line = line.strip().split(' ||| ')
                    if len(line) == 2:  ## sentence and label
                        text_a = line[0]
                        label = line[-1]
                    elif len(line) == 1:  ## only sentence for pos/neg
                        text_a = line[0]
                        label = '0' if split == 'neg' else '1'
                    example = dict(text_a=text_a, label=int(label), domain_idx=domain_idx)
                    tmp.append(example)
            examples += tmp
        return examples
    def get_fewshot_examples(self, data_dir, split, domains, seed, shots):
        examples = []
        random.seed(seed)
        sample_idxs = random.sample(range(800), shots)
        for domain_idx, domain in enumerate(domains):
            tmp = []
            with open(os.path.join(data_dir, domain, '{}.txt'.format(split)), 'r', encoding='utf-8') as fin:
                for idx, line in enumerate(fin.readlines()):
                    line = line.strip().split(' ||| ')
                    if len(line) == 2:  ## sentence and label
                        text_a = line[0]
                        label = line[-1]
                    elif len(line) == 1:  ## only sentence for pos/neg
                        text_a = line[0]
                        label = '0' if split == 'neg' else '1'
                    example = dict(text_a=text_a, label=int(label), domain_idx=domain_idx)
                    tmp.append(example)
            for sample_idx in sample_idxs:
                examples.append(tmp[sample_idx])
        return examples

    def get_train_examples(self, data_dir, domains):
        pos_ex = self.get_examples(data_dir, 'pos', domains)
        neg_ex = self.get_examples(data_dir, 'neg', domains)
        return pos_ex + neg_ex
    def get_train_fewshot_examples(self, data_dir, domains, seed, shots):
        pos_ex = self.get_fewshot_examples(data_dir, 'pos', domains, seed, shots)
        neg_ex = self.get_fewshot_examples(data_dir, 'neg', domains, seed, shots)
        return pos_ex + neg_ex

    def get_dev_examples(self, data_dir, domains):
        return self.get_examples(data_dir, 'dev', domains)

    def get_test_examples(self, data_dir, target_domain):
        return self.get_examples(data_dir, 'dev', [target_domain])


class IMDBProcessor(Dataset):
    def __init__(self):
        super().__init__()
        self.labels = [["negative", 'bad'], ["positive", 'good']]

    def get_examples(self, data_dir, split):
        examples = []
        label_file = open(os.path.join(data_dir, "{}_labels.txt".format(split)), 'r', encoding='utf-8')
        labels = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir, '{}.txt'.format(split)), 'r', encoding='utf-8') as fin:
            for idx, line in enumerate(fin):
                text_a = line.strip()
                example = dict(text_a=text_a, label=int(labels[idx]))
                examples.append(example)
        return examples

    def get_train_examples(self, data_dir):
        return self.get_examples(data_dir, 'train')

    def get_test_examples(self, data_dir):
        return self.get_examples(data_dir, 'test')


import csv


class SST2Processor(Dataset):
    def __init__(self):
        super().__init__()
        self.labels = [["negative", 'bad'], ["positive", 'good']]

    def get_examples(self, data_dir, split):
        examples = []
        path = os.path.join(data_dir, "{}.tsv".format(split))
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for idx, example_json in enumerate(reader):
                text_a = example_json['sentence'].strip()
                example = dict(text_a=text_a, label=int(example_json['label']))
                examples.append(example)
        return examples


class AmazonDataset(Dataset):
    def __init__(self, data_dir, source_domain, target_domain, template, tokenizer, type='train', seed=1, shots=1):
        super().__init__()
        self.token_ids = []
        self.processor = AmazonMultiProcessor()
        self.labels = self.processor.labels
        for verbs in self.labels:
            ids = tokenizer(verbs)['input_ids']
            self.token_ids.append([y[1:-1] for y in ids])
        source_domain.sort()
        if type == 'train':
            self.examples = self.processor.get_train_examples(data_dir, source_domain)
        elif type =='train_fewshot':
            self.examples = self.processor.get_train_fewshot_examples(data_dir, source_domain, seed, shots)
        elif type == 'dev':
            self.examples = self.processor.get_dev_examples(data_dir, source_domain)
        else:
            self.examples = self.processor.get_test_examples(data_dir, target_domain)


        if template == 'type0':
            self.prompt = self.template_wrapper0()
            self.tokens = tokenizer(self.prompt),
        elif template == 'type1':
            self.prompt = self.template_wrapper1()
            self.tokens = tokenizer(self.prompt),
        elif template == 'type2':
            self.prompt = self.template_wrapper2()
            self.tokens = tokenizer(self.prompt),
    def template_wrapper0(self):
        # "<s><text_a><eos>"
        return [x['text_a'] for x in self.examples]

    def template_wrapper1(self):
        # "It was <mask>. <text_a>"
        return ["It was <mask>. " + x['text_a'] for x in self.examples]
    def template_wrapper2(self):
        # "<mask>. <text_a>"
        return ["<mask>. " + x['text_a'] for x in self.examples]
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[0]['input_ids'][idx]), torch.tensor(self.tokens[0]['attention_mask'][idx]), \
            self.examples[idx]['label'], self.examples[idx]['domain_idx']


class IMDBDataset(Dataset):
    def __init__(self, data_dir, template, tokenizer, type='test'):
        super().__init__()
        self.token_ids = []

        self.processor = IMDBProcessor()
        self.labels = self.processor.labels
        for verbs in self.labels:
            ids = tokenizer(verbs)['input_ids']
            self.token_ids.append([y[1:-1] for y in ids])

        self.examples = self.processor.get_test_examples(data_dir)

        if template == 'type0':
            self.prompt = self.template_wrapper0()
            self.tokens = tokenizer(self.prompt),
        elif template == 'type1':
            self.prompt = self.template_wrapper1()
            self.tokens = tokenizer(self.prompt),
        elif template == 'type2':
            self.prompt = self.template_wrapper2()
            self.tokens = tokenizer(self.prompt),

    def template_wrapper0(self):
        # "<s><text_a><eos>"
        return [x['text_a'] for x in self.examples]

    def template_wrapper1(self):
        # "It was <mask>. <text_a>"
        return ["It was <mask>. " + x['text_a'] for x in self.examples]

    def template_wrapper2(self):
        # "<mask>. <text_a>"
        return ["<mask>. " + x['text_a'] for x in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[0]['input_ids'][idx]), torch.tensor(self.tokens[0]['attention_mask'][idx]), \
            self.examples[idx]['label'], 0


class SST2Dataset(Dataset):
    def __init__(self, data_dir, template, tokenizer, type='test'):
        super().__init__()
        self.token_ids = []

        self.processor = SST2Processor()
        self.labels = self.processor.labels
        for verbs in self.labels:
            ids = tokenizer(verbs)['input_ids']
            self.token_ids.append([y[1:-1] for y in ids])
        self.examples = self.processor.get_examples(data_dir, 'test')

        if template == 'type0':
            self.prompt = self.template_wrapper0()
            self.tokens = tokenizer(self.prompt),
        elif template == 'type1':
            self.prompt = self.template_wrapper1()
            self.tokens = tokenizer(self.prompt),
        elif template == 'type2':
            self.prompt = self.template_wrapper2()
            self.tokens = tokenizer(self.prompt),

    def template_wrapper0(self):
        # "<s><text_a><eos>"
        return [x['text_a'] for x in self.examples]

    def template_wrapper1(self):
        # "It was <mask>. <text_a>"
        return ["It was <mask>. " + x['text_a'] for x in self.examples]

    def template_wrapper2(self):
        # "<mask>. <text_a>"
        return ["<mask>. " + x['text_a'] for x in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[0]['input_ids'][idx]), torch.tensor(self.tokens[0]['attention_mask'][idx]), \
            self.examples[idx]['label'], 0

