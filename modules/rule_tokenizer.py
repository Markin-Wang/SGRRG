from collections import Counter
import os
import pickle
from tqdm import tqdm
import spacy
import torch
from transformers.tokenization_utils_base import BatchEncoding
from collections.abc import Mapping


def tokenize(text):
    return text.split(' ')


class RuleTokenizer(object):
    def __init__(self, config, all_texts=None, tokenizer=None):
        self.threshold = config['min_frequency']
        self.dataset_name = config['dataset_name']
        self.voc_path = os.path.join(config['data_root'], tokenizer, str(self.threshold))
        self.special_tokens = ['<CLS>', '<PAD>', '<MASK>', '<SEP>']
        self.cls_token, self.pad_token, self.mask_token, self.sep_token = \
            '<CLS>', '<PAD>', '<MASK>', '<SEP>'
        self.special_tokens_ids = [0, 1, 2, 3]
        self.cls_token_id = 0
        if tokenizer == 'rule-spacy':
            self.tokenizer = spacy.load("en_core_web_sm")
        elif tokenizer == 'rule-split':
            self.tokenizer = tokenize
        else:
            raise NotImplementedError
        self.pad_token_id = 1
        self.mask_token_id = 2
        self.sep_token_id = 3
        if all_texts is None:
            with open(self.voc_path+'/vocabulary.pkl', 'rb') as f:
                self.token2idx, self.idx2token, threshold = pickle.load(f)
            if threshold != self.threshold:
                self.token2idx, self.idx2token = self.create_vocabulary()
        else:
            self.token2idx, self.idx2token = self.create_vocabulary(all_texts)
        self.vocab_size = len(self.token2idx)
        self.model_input_names = ['input_ids']
        #torch.save(self.token2idx,'mimic_token2idx100nltk.pth')

    def create_vocabulary(self, all_texts):
        print('Generating vocabulary...')
        total_tokens = []
        for example in tqdm(all_texts):
            tokens = self.tokenizer(example)
            for token in tokens:
                if not isinstance(token, str): token = token.text
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<UNK>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        token2idx['<CLS>'], token2idx['<PAD>'], token2idx['<MASK>'], token2idx['<SEP>'] = \
            self.cls_token_id, self.pad_token_id, self.mask_token_id, self.sep_token_id
        idx2token[self.cls_token_id], idx2token[self.pad_token_id],\
        idx2token[self.mask_token_id], idx2token[self.sep_token_id] = '<CLS>', '<PAD>', '<MASK>', '<SEP>'
        start = len(token2idx)
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + start
            idx2token[idx + start] = token
        os.makedirs(self.voc_path, exist_ok=True)
        with open(self.voc_path+'/vocabulary.pkl', 'wb') as f:
             pickle.dump([token2idx, idx2token, self.threshold], f)
        return token2idx, idx2token

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            #return self.token2idx['unk']
            return self.token2idx['<UNK>']
        return self.token2idx[token]

    def convert_tokens_to_ids(self, token):
        if token not in self.token2idx:
            #return self.token2idx['unk']
            return self.token2idx['<UNK>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report, padding="max_length", truncation=False,
                 max_length=100, return_special_tokens_mask=False):
        tokens = self.tokenizer(report)
        ids = [self.pad_token_id] * max_length
        attention_mask = [0] * max_length
        attention_mask[0] = 1
        ids[0] = self.cls_token_id
        special_token_masks = [1] * max_length
        for i, token in enumerate(tokens):
            if i >= max_length-1: break
            if not isinstance(token, str): token = token.text
            if token in self.token2idx:
                id = self.get_id_by_token(token)
            else:
                id = self.get_id_by_token('<UNK>')
            ids[i+1] = id
            special_token_masks[i+1] = int(id in self.special_tokens_ids)
            attention_mask[i+1] = 1
        ret = {'input_ids': ids, 'special_tokens_mask': special_token_masks, 'attention_mask': attention_mask}
        return BatchEncoding(ret)

    def decode(self, ids, skip_special_tokens=False):
        txt = ''
        for i, idx in enumerate(ids):
            if isinstance(idx,torch.Tensor):
                idx = idx.item()
            if idx in self.special_tokens_ids:
                if skip_special_tokens: continue
            else:
                if len(txt) > 0: txt += ' '
                txt += self.idx2token[idx]
        return txt

    def batch_decode(self, ids_batch, skip_special_tokens=False):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids, skip_special_tokens=skip_special_tokens))
        return out

    def pad(self, encoded_inputs, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=None):
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

            # The model's main input name, usually `input_ids`, has be passed for padding

        return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

    def __len__(self):
        return self.vocab_size