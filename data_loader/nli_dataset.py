import os
from torch.utils.data.dataset import Dataset
import json
import pickle
import torch
import random

class MNLIProcessor(Dataset):
    def __init__(self):
        super().__init__()
        self.labels = ["neutral", "contradiction", "entailment"]
        '''
        0-2-("'Less loud.", 'Please be quiet.')-entailment
        1-1-("Turned out, I wasn't completely wrong.", 'I was 100 percent wrong. ')-contradiction
        2-2-('Mrs. Vandemeyer, she read, "20 South Audley Mansions.', '"20 South Audley Mansions" is a book that Mrs. Vandemeyer read.')-entailment
        3-2-("Warn't no Apaches, that's for certain.", "It wasn't the Apaches, certainly.")-entailment
        4-0-('It was dead quiet.', "There wasn't much noise in the air.")-neutral
        [neutral, contradiction, entailment]
        '''

    def get_examples(self, data_dir, split, domains):
        examples = []

        for domain_idx, domain in enumerate(domains):
            tmp = []
            with open(os.path.join(data_dir, domain, split), 'rb') as f:
                (text, labels) = pickle.load(f)
            assert len(text) == len(labels)
            for i in range(len(text)):
                tmp.append({'text_a': text[i][0], 'text_b': text[i][1], 'label': labels[i], 'domain_idx': domain_idx})
            examples += tmp
        return examples
    #
    # def get_fewshot_examples(self, data_dir, split, domains, seed, shots):
    #     examples = []
    #     random.seed(seed)
    #     sample_idxs = random.sample(range(2500), shots)
    #     for domain_idx, domain in enumerate(domains):
    #         tmp = []
    #         with open(os.path.join(data_dir, domain, split), 'rb') as f:
    #             (text, labels) = pickle.load(f)
    #         assert len(text) == len(labels)
    #         for i in range(len(text)):
    #             tmp.append({'text_a': text[i][0], 'text_b': text[i][1], 'label': labels[i], 'domain_idx': domain_idx})
    #
    #         for sample_idx in sample_idxs:
    #             examples.append(tmp[sample_idx])
    #     return examples
    def get_fewshot_examples(self, data_dir, split, domains, seed, shots):
        examples = []
        random.seed(seed)
        for domain_idx, domain in enumerate(domains):
            label_group = {}

            with open(os.path.join(data_dir, domain, split), 'rb') as f:
                (text, labels) = pickle.load(f)
            assert len(text) == len(labels)

            for i in range(len(text)):
                label = labels[i]
                if label not in label_group:
                    label_group[label] = []
                label_group[label].append({'text_a': text[i][0],
                                           'text_b': text[i][1],
                                           'label': labels[i],
                                           'domain_idx': domain_idx})

            for label_group, label_examples in label_group.items():
                sample_idxs = random.sample(range(len(label_examples)), shots)
                examples.extend(label_examples[idx] for idx in sample_idxs)

        return examples
    def get_train_examples(self, data_dir, domains):
        '''{'travel', 'slate', 'government', 'telephone', 'fiction'}'''
        return self.get_examples(data_dir, 'train', domains)

    def get_dev_examples(self, data_dir, domains):
        '''{'travel', 'slate', 'government', 'telephone', 'fiction'}'''
        return self.get_examples(data_dir, 'dev', domains)

    def get_test_examples(self, data_dir, target_domain):
        return self.get_examples(data_dir, 'test', [target_domain])


class SNLIProcessor(Dataset):
    def __init__(self):
        super().__init__()
        self.labels = ["neutral", "contradiction", "entailment"]

    def get_examples(self, data_dir, split):
        gold_label_to_label_idx = {
            'neutral': 0,
            'contradiction': 1,
            'entailment': 2
        }
        examples = []
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                if example_json['gold_label'] == '-':
                    continue
                label = gold_label_to_label_idx[example_json["gold_label"]]
                text_a = example_json['sentence1']
                text_b = example_json['sentence2']
                example = dict(text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
        return examples

    def get_train_examples(self, data_dir):
        '''{'travel', 'slate', 'government', 'telephone', 'fiction'}'''
        return self.get_examples(data_dir, 'train')

    def get_dev_examples(self, data_dir):
        '''{'travel', 'slate', 'government', 'telephone', 'fiction'}'''
        return self.get_examples(data_dir, 'dev')

    def get_test_examples(self, data_dir):
        return self.get_examples(data_dir, 'test')


class SICKProcessor(Dataset):
    def __init__(self):
        super().__init__()
        self.labels = ["neutral", "contradiction", "entailment"]

    def get_examples(self, data_dir):
        gold_label_to_label_idx = {
            'neutral': 0,
            'contradicts': 1,
            'entails': 2
        }
        examples = dict(
            train=[],
            trial=[],
            test=[]
        )
        path = os.path.join(data_dir, "SICK.txt")
        with open(path, encoding='utf8') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                items = line.strip().split('\t')
                text_a = items[1]
                text_b = items[2]
                label_ab = items[5][2:-2].lower()
                label1 = gold_label_to_label_idx[label_ab]
                label_ba = items[6][2:-2].lower()
                label2 = gold_label_to_label_idx[label_ba]
                split = items[-1].lower()
                example1 = dict(text_a=text_a, text_b=text_b, label=label1)
                example2 = dict(text_a=text_b, text_b=text_a, label=label2)

                examples[split].append(example1)
                examples[split].append(example2)

        return examples

    def get_train_examples(self, data_dir):
        '''{'travel', 'slate', 'government', 'telephone', 'fiction'}'''
        return self.get_examples(data_dir)['train']

    def get_dev_examples(self, data_dir):
        '''{'travel', 'slate', 'government', 'telephone', 'fiction'}'''
        return self.get_examples(data_dir)['trial']

    def get_test_examples(self, data_dir):
        return self.get_examples(data_dir)['test']


class MNLIDataset(Dataset):
    def __init__(self, data_dir, source_domain, target_domain, template, tokenizer, type='train', seed=1, shots=1):
        super().__init__()

        self.labels = [["neutral", 'uncertain'], ['no', 'false'], ['yes', 'true']]
        self.token_ids = []
        for verbs in self.labels:
            ids = tokenizer(verbs)['input_ids']
            self.token_ids.append([y[1:-1] for y in ids])
        self.processor = MNLIProcessor()
        source_domain.sort()
        if type == 'train':
            self.examples = self.processor.get_train_examples(data_dir, source_domain)
        elif type == 'train_fewshot':
            self.examples = self.processor.get_fewshot_examples(data_dir, 'train', source_domain, seed, shots)
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
        # "<s><text_a>, <text_b><eos>"
        return [x['text_a'] + ' ' + x['text_b'] for x in self.examples]

    def template_wrapper1(self):
        # "Does the first sentence entails the second? <mask>. <text_a> <text_b>"
        return ["Does the first sentence entails the second? <mask>." + x['text_a'] + x['text_b'] for x in
                self.examples]

    def template_wrapper2(self):
        return ['<mask>, ' + x['text_a'] + ' ' + x['text_b'] for x in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[0]['input_ids'][idx]), torch.tensor(self.tokens[0]['attention_mask'][idx]), \
            self.examples[idx]['label'], self.examples[idx]['domain_idx']


class SNLIDataset(Dataset):
    def __init__(self, data_dir, template, tokenizer, type='train'):
        super().__init__()

        self.labels = [["neutral", 'uncertain'], ['no', 'false'], ['yes', 'true']]
        self.token_ids = []
        for verbs in self.labels:
            ids = tokenizer(verbs)['input_ids']
            self.token_ids.append([y[1:-1] for y in ids])
        self.processor = SNLIProcessor()
        if type == 'train':
            self.examples = self.processor.get_train_examples(data_dir)
        elif type == 'dev':
            self.examples = self.processor.get_dev_examples(data_dir)
        else:
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
        # "<s><text_a>, <text_b><eos>"
        return [x['text_a'] + ' ' + x['text_b'] for x in self.examples]

    def template_wrapper1(self):
        # "Does the first sentence entails the second? <mask>. <text_a> <text_b>"
        return ["Does the first sentence entails the second? <mask>." + x['text_a'] + x['text_b'] for x in
                self.examples]

    def template_wrapper2(self):
        return ['<mask>, ' + x['text_a'] + ' ' + x['text_b'] for x in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[0]['input_ids'][idx]), torch.tensor(self.tokens[0]['attention_mask'][idx]), \
            self.examples[idx]['label'], 0


class SICKDataset(Dataset):
    def __init__(self, data_dir, template, tokenizer, type='train'):
        super().__init__()

        self.labels = [["neutral", 'uncertain'], ['no', 'false'], ['yes', 'true']]
        self.token_ids = []
        for verbs in self.labels:
            ids = tokenizer(verbs)['input_ids']
            self.token_ids.append([y[1:-1] for y in ids])
        self.processor = SICKProcessor()
        if type == 'train':
            self.examples = self.processor.get_train_examples(data_dir)
        elif type == 'dev':
            self.examples = self.processor.get_dev_examples(data_dir)
        else:
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
        # "<s><text_a>, <text_b><eos>"
        return [x['text_a'] + '. ' + x['text_b'] for x in self.examples]

    def template_wrapper1(self):
        # "Does the first sentence entails the second? <mask>. <text_a> <text_b>"
        return ["Does the first sentence entails the second? <mask>." + x['text_a'] + x['text_b'] for x in
                self.examples]

    def template_wrapper2(self):
        return ['<mask>, ' + x['text_a'] + ' ' + x['text_b'] for x in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[0]['input_ids'][idx]), torch.tensor(self.tokens[0]['attention_mask'][idx]), \
            self.examples[idx]['label'], 0
