import torch
import tools
import tools.tool as tools


def predict_sentiment(net, sentences):
    seq = torch.tensor(vocab[sentences.split()], device=tools.try_gpu)
    label = torch.argmax(net(seq.reshape(1,-1)), dim=1)
    return 'positive' if label == 1 else 'negtive'










if __name__ == "__main__":
    predict_sentiment(net, 'this movie is bad')