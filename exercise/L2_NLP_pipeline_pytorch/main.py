# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn

import data
import model
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch NLP Pipeline')
parser.add_argument('--data', type=str, default='./data/glue-sst2/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=10,  # 你可能需要调整它
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='verify the code and the model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

device = torch.device("cuda" if args.cuda else "cpu")

# 和word_language_model不同, sst2的corpus既要包含输入也要包含输出(label)
corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

###############################################################################
# sst2改动batchify功能: n行句子分成nbatch批, shape从输入的data的[n, seq_len]变为[nbatch, bsz, seq_len]，并且要包含标签信息
# 即输出是一个数组，每个元素代表一个batch，每个batch是一个tuple，包含data和label，data的shape为[bsz, seq_len]，label是一维0/1数组
###############################################################################
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // bsz
    new_data = []
    for i in range(nbatch):
        tensors = []
        new_label = []
        for j in range(bsz):
            tensors.append(data[i*bsz+j][0])
            new_tensor = torch.stack(tensors, dim=0)
            new_label.append(data[i*bsz+j][1])
            new_tuple = (new_tensor, new_label)
        new_data.append(new_tuple)
    return new_data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)
if args.model == 'LSTM':
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

# 这个函数可能不再需要了，因为经过batchify后，我们的数据已经处理成一个batch一个batch的数组了，直接取就可以了
"""
def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
"""

def evaluate(data_source):  # 参考下面的train函数更改
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for batch in range(len(data_source)):
            data = data_source[batch][0].to(device).permute(1, 0)
            targets = torch.Tensor(data_source[batch][1]).type(torch.LongTensor).to(device)
            output = model(data)
            total_loss += criterion(output, targets).item()
            ###############################################################################
            # sst2不应只得到loss, 还要评测label正确的概率, 即acc
            ###############################################################################
            # 怎么得到预测结果呢？
            # 提示：pred = output.argmax(dim=1)
            pred = output.argmax(dim=1)
            correct += pred.eq(targets).sum().item() / eval_batch_size
    return total_loss / (len(data_source) - 1), correct / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch in range(len(train_data)):  # 由于我们已经把train_data整理成batch构成的数组了，这里直接进行迭代即可。
        data = train_data[batch][0].to(device).permute(1, 0)
        targets = torch.Tensor(train_data[batch][1]).type(torch.LongTensor).to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        ###############################################################################
        # sst2的output应为0/1二分类的概率（经过softmax之前）
        ###############################################################################
        output = model(data)  # 作出相应更改
        loss = criterion(output, targets)  # 这里需要改吗？softmax在哪里执行的？
        loss.backward()
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        # 这里展示的是手动更新参数的方式，而不是使用optimizer。你可以试一试改成optimizer。
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)
        
        total_loss += loss.item()
        
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            with open ('log.txt', 'a') as f:
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.3f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} |'.format(
                    epoch, batch, len(train_data), lr,
                    elapsed * 1000 / args.log_interval, cur_loss), file = f)  # 去掉perplexity
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss, val_acc = evaluate(val_data)
        with open ('log.txt', 'a') as f:
            print('-' * 89, file = f)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid acc {:8.3f} |'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, val_acc), file = f)
            print('-' * 89, file = f)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss, test_acc = evaluate(test_data)
with open ('log.txt', 'a') as f:
    print('=' * 89, file = f)
    print('| End of training | test loss {:5.2f} | test acc {:8.3f} |'.format(
        test_loss, test_acc), file = f)
    print('=' * 89, file = f)