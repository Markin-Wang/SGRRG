import json
import re
from collections import Counter
import os
import torch
import pickle
from tqdm import tqdm

class Tokenizer(object):
    def __init__(self, config, all_text):
        #self.ann_path = os.path.join(config['data_dir'], config['dataset_name'], 'annotation.json')
        self.threshold = config['threshold']
        self.dataset_name = config['dataset_name']
        self.data_dir = config['data_dir']
        self.all_text = all_text
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        #self.ann = json.loads(open(self.ann_path, 'r').read())
        # voc_path = os.path.join(config['data_dir'], config['dataset_name'], 'vocabulary.pkl')
        # if os.path.exists(voc_path):
        #     with open(voc_path,'rb') as f:
        #         self.token2idx, self.idx2token, threshold = pickle.load(f)
        #     if threshold != self.threshold:
        #         self.token2idx, self.idx2token = self.create_vocabulary()
        # else:
        self.token2idx, self.idx2token = self.create_vocabulary()
        #torch.save(self.token2idx,'mimic_token2idx100nltk.pth')

    def create_vocabulary(self):
        print('Generating vocabulary...')
        total_tokens = []
        # for example in tqdm(self.ann['train']):
        for example in tqdm(self.all_text):
            # tokens = self.clean_report(example['report']).split()
            tokens = example[0].split() # only one caption
            #tokens = nltk.word_tokenize(self.clean_report(example['report']))
            #print(tokens)
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        #vocab = [k for k, v in counter.items() if v >= self.threshold] + ['unk']
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        # with open(os.path.join(self.data_dir, self.dataset_name, 'vocabulary.pkl'), 'wb') as f:
        #      pickle.dump([token2idx, idx2token,self.threshold], f)
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            #return self.token2idx['unk']
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx) + 1 # for special token like bos, eos, pad

    def __call__(self, report):
        # tokens = self.clean_report(report).split()
        tokens = report.split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
