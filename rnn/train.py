
from rnn.data import load_data_imdb
import tools.tool as tools
from rnn.BiRNN import BiRNN
from rnn.data import TokenEmbedding
import torch
from torch import nn
import rnn.env as env


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

def acc(preds, gt):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == gt)
    acc = correct.float().mean().item()
    return acc



def train_batch(net, features, labels, loss, trainer, devices):
    features = features.to(devices)
    labels = labels.to(devices)
    net.train()
    trainer.zero_grad()
    pred = net(features)
    l = loss(pred, labels)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = acc(pred, labels)
    return train_loss_sum, train_acc_sum

def train_rnn():
    batch_size = 64
    imdb_data_dir = env.data_dir_with('aclImdb')
        

    train_iter, test_iter, vocab = load_data_imdb(imdb_data_dir, batch_size, 512)

    emb_size, num_hiddens, num_layers = 100, 100, 2
    devs = tools.dev()

    net = BiRNN(len(vocab), emb_size, num_hiddens, num_layers)
    net.apply(init_weight)

    emb_dir = env.data_dir_with('glove.6B.100d')
    glove_emb = TokenEmbedding(emb_dir)
    embs = glove_emb[vocab.idx_to_tokens]

    net.embedding.weight.data.copy_(embs)
    net.embedding.weight.requires_grad = False

    lr, num_epochs = 0.01, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(num_epochs):
        l, acc = 0, 0
        for i, (features, labels) in enumerate(train_iter):
            l, acc = train_batch(net,features, labels,loss, trainer, devs)
        print(f'loss:{l}, acc:{acc}')
    return net, vocab