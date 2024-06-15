import torch

import sys
sys.path.append('..')

import tools.tool as tool

import data
from data import read_imdb


def predict_sentiment(net, sentences):
    seq = torch.tensor(vocab[sentences.split()], device=tools.try_gpu)
    label = torch.argmax(net(seq.reshape(1,-1)), dim=1)
    return 'positive' if label == 1 else 'negtive'










if __name__ == "__main__":
    # predict_sentiment(net, 'this movie is bad')

    # data_dir = '/Users/hogan/Desktop/AICode/traindata/aclImdb'
    # train_data = read_imdb(datadir=data_dir, is_train=True)
    # print(train_data)
    data.test()