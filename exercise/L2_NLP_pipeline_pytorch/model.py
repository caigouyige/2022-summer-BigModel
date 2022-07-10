import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        ###############################################################################
        # sst2的decoder不应在整个词表上预测，而是预测0/1的概率，因此下面ntoken可能需要改成2，即2分类
        ###############################################################################
        self.decoder = nn.Linear(nhid, 2)
        self.drop = nn.Dropout(dropout)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input):  # hidden在这里面没用了，我们不需要上一个状态的hidden，每个样本都是独立的
        emb = self.drop(self.encoder(input))
        list_input = torch.Tensor.tolist(input)
        real_seq_lens = []
        for i in range(input.shape[1]):
            zero_num = 0
            for j in range(len(list_input)):
                if list_input[j][i] == 0:
                    zero_num += 1
            real_seq_lens.append(len(list_input) - zero_num - 1)
        output = self.rnn(emb)  # 这里面也不需要hidden作为输入了，输出也不用要hidden
        output = self.drop(output[0])  # shape是seq_len, bsz, nhid
        output = output[real_seq_lens, torch.arange(output.shape[1]), :]
        # 补充代码，从上面的output中，抽取最后一个词的输出作为最终输出。要注意考虑到序列的真实长度。最后得到一个shape是bsz, nhid的tensor
        # 提示：output = output[real_seq_lens - 1, torch.arange(output.shape[1]), :]

        decoded = self.decoder(output)
        decoded = decoded.view(-1, 2)
        return decoded # 不再需要输出hidden；最终输出的shape是bsz, 2