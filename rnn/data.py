import collections
import os

import torch

import rnn.env as env

from tools.tool import load_array


# data_dir = '/Users/hogan/Desktop/AICode/traindata/aclImdb'

def read_imdb(datadir, is_train=True):

    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(datadir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)

    return data, labels


## 预处理，过滤出现次数不到5次的单词

def tokenize(lines):
    return [line.split() for line in lines]

def truncate_pad(seq, seq_len, pad_token):
    if len(seq) > seq_len:
        seq = seq[:seq_len]
    return seq + [pad_token] * (seq_len - len(seq))

class Vocab:

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = self.count_corpus(tokens)
        self._tokens_freqs = sorted(counter.items(), key=lambda x: x[1],
                                    reverse=True)
        self.idx_to_tokens = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_tokens)}
        
        for token, freq in self._tokens_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_tokens.append(token)
                self.token_to_idx[token] = len(self.idx_to_tokens)-1

    def __len__(self):
        return len(self.idx_to_tokens)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_tokens[indices]
        return [self.idx_to_tokens[idx] for idx in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs

    def count_corpus(self, tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)


class TokenEmbedding:
    def __init__(self, emb_dir) -> None:
        self.idx_to_token, self.idx_to_vec = self._load_embedding(emb_dir)
        self.unkown_idx = 0
        self.token_to_idx = {token:idx
                             for idx, token in enumerate(self.idx_to_token)}
    
    def _load_embedding(self, emb_dir):
        print(f'start to read embedding files: {emb_dir}')
        idx_to_token = ['<unk>']
        idx_to_vec = []
        with open(os.path.join(emb_dir,'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                if len(elems) < 2 :
                    continue
                token, emb = elems[0], [float(ele) for ele  in elems[1:]]
                idx_to_token.append(token)
                idx_to_vec.append(emb)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)
    
    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unkown_idx)
                   for token in tokens]
        return self.idx_to_vec[torch.tensor(indices)]
    
    def __len__(self):
        return len(self.idx_to_token)

def load_data_imdb(data_dir, batch_size, seq_len=512):
    print('start to read imdb data files....')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens_lines = tokenize(train_data[0])
    test_tokens_lines = tokenize(test_data[0])
    vocab = Vocab(train_tokens_lines, min_freq=5)
    train_features = torch.tensor([truncate_pad(vocab[line], seq_len=seq_len, pad_token=vocab['<pad>'])
                                   for line in train_tokens_lines])
    test_features = torch.tensor([truncate_pad(vocab[line], seq_len=seq_len, pad_token=vocab['<pad>'])
                                   for line in test_tokens_lines])
    train_iter = load_array((train_features, torch.tensor(train_data[1])), batch_size)
    test_iter = load_array((test_features, torch.tensor(train_data[1])), batch_size)
    return train_iter, test_iter, vocab




# train_data = read_imdb(datadir=data_dir, is_train=True)
# train_tokens = tokenize(train_data[0])
# vocab = Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])