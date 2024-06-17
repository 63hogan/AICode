import torch

import tools.tool as tools
import rnn.data as data
from rnn.data import read_imdb
import rnn.train as train
import rnn.env as env


def predict_sentiment(net, vocab, sentences):
    seq = torch.tensor(vocab[sentences.split()], device=tools.dev())
    label = torch.argmax(net(seq.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negtive'


def train_rnn():
    env.sample = True
    net, vocab = train.train_rnn()
    return net, vocab


if __name__ == "__main__":

    net, vocab = train_rnn()
    print(predict_sentiment(net, vocab, "I hate this movie"))
    print(predict_sentiment(net, vocab, "I love this movie"))
