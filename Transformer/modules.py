##模块
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)      ##xavier初始化
        if bias:
            init.zeros_(self.linear.bias)           ##全0分布

    def forward(self, inputs):
        return self.linear(inputs)


class ScaledDotProductAttention(nn.Module):  ##点乘注意力
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = (torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor).cuda()   ##matmul函数详解https://blog.csdn.net/qsmx666/article/details/105783610/
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)   ##masked_fill方法有两个参数，maske和value，mask是一个pytorch张量（Tensor），元素是布尔值，value是要填充的值，填充规则是mask中取值为True位置对应于self的相应位置用value填充
        attn = self.dropout(self.softmax(scores))

        # outputs: [b_size x n_heads x len_q x d_v]
        context = torch.matmul(attn, v)

        return context, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))  ##将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.beta = nn.Parameter(torch.zeros(d_hid))  ##ones返回一个全为1 的张量
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)  ##标准差
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out


class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]  ##power 乘方
            for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)   ##concat多数组拼接 astype类型转换，

        # additional single row for PAD idx
        self.pos_enc = nn.Embedding(max_seq_len + 1, d_word_vec)
        # fix positional encoding: exclude weight from grad computation
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)##功能：torch.from_numpy(ndarray) → Tensor，即 从numpy.ndarray创建一个张量。

    def forward(self, input_len):
        max_len = torch.max(input_len)
        # tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # input_pos = tensor([list(range(1, len+1)) + [0]*(max_len-len) for len in input_len])
        input_pos = torch.zeros((input_len.size(0),max_len)).long().cuda()
        for i,len in enumerate(input_len):
            input_pos[i,:len] = torch.arange(1,len+1)   ##torch.arange(start=1.0,end=6.0)的结果不包括end
                                                        #torch.range(start=1.0, end=6.0)的结果包括end
        return self.pos_enc(input_pos)