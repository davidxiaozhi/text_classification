#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# author bjlizhipeng

"""

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_batch_tanh_linear(out_state, linear_weight, linear_bias):
    """
    attetion 当中的 线性变化 tanh(W(w)*H(it)+B(w))
    :param out_state: 双向 gru 的输出 output_word:(batch, seq_len, hidden_size * num_directions) batch_first=True 默认
            注意如果是批量处理的话 batch = document_batch *  contain_sent_for_document
    :param linear_weight: 线性变化的权重 [word_gru_hidden_num,word_gru_hidden_num]
    :param linear_bias: [word_gru_hidden_num]
    :return:
    """

    # seq : (batch, seq_len, hidden_size * num_directions)
    batch_sent = out_state.shape[0]
    sent_length = out_state.shape[1]
    bias_dim = linear_bias.shape[0]
    s = list()
    for i in range(batch_sent):
        # sentences=[seq_len, hidden_size * num_directions
        sentences = out_state[i]
        bias = linear_bias.expand(sent_length, bias_dim)
        _s = torch.mm(sentences,linear_weight)
        _s_add_bias = _s + bias
        _s_add_bias = torch.tanh(_s_add_bias)
        _s_add_bias = _s_add_bias.unsqueeze(0)
        s.append(_s_add_bias)
    b_u_it = torch.cat(s,dim = 0)
    del s
    return b_u_it

def attention_batch_context_matmul(b_U_it, context_weight):
    """
     b_U_it 行列变化后 与 上下文向量 context_weight 相乘
    :param b_U_it: 经过 线性变化以后的 out_state,然后在 tanh 处理,就是 U_it
               b_U_it: (batch, seq_len, hidden_size * num_directions)
    :param context_weight: [hidden_size * num_directions,1]
    :return:
    """
    # seq : (batch, seq_len, hidden_size * num_directions)
    batch_sent = b_U_it.shape[0]
    sent_length = b_U_it.shape[1]
    #s = torch.from_numpy(np.zeros(batch_sent))
    s = list()
    for i in range(batch_sent):
        # seq_len, hidden_size * num_directions
        U_it = b_U_it[i]
        _s = torch.mm(U_it, context_weight)
        #context_weight.detach().numpy()
        #idx = np.where(np.isnan(A))
        # [ seq_len ,1 ] 使用transpose(0,1) 可以达到同样的效果
        #_s = _s.unsqueeze(0)
        _s = _s.transpose(0,1)
        s.append(_s)
    # [ batch_size,seq_len]
    r = torch.cat(s, dim=0)
    del s

    return r



def attention_mul(rnn_outputs, att_weights):
    """

    :param rnn_outputs: [sen_num,token_num,hidden_num]
    :param att_weights: [sen_num,token_nu]
    :return:
    """
    # [batch_size, token_num, hidden_num ]
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        #print("{}-{}".format(a_i.shape, h_i.shape))
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    # [batch_size, hidden_num ]
    sent_vectors = torch.sum(attn_vectors, 1)
    return sent_vectors


# ## Word attention word_model with bias

class AttentionWordRNN(nn.Module):

    def __init__(self, vocab_size, embed_size, word_gru_hidden_num, max_words, bidirectional=True):
        """
        单词级别的注意力模型 我们使用的 gru 默认都配置 batch_first = True
        :param vocab_size: 词汇表大小
        :param embed_size: embedding 的维度大小
        :param word_gru_hidden_num: gru hidden unit
        :param max_words 一句文本里面最大包含的 词 的 数目
        :param bidirectional:  是否使用双向 gru
        """
        super(AttentionWordRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_words =  max_words
        self.word_gru_hidden = word_gru_hidden_num
        self.bidirectional = bidirectional
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.word_gru = nn.GRU(embed_size, word_gru_hidden_num, bidirectional=True, batch_first=True)
        if bidirectional == True:
            #定义参数 W [2 * word_gru_hidden_num, 2 * word_gru_hidden_num] B [2 * word_gru_hidden_num, 1]
            self.weight_W_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden_num, 2 * word_gru_hidden_num))
            self.bias_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden_num))
            self.weight_proj_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden_num, 1))
        else:
            # attention linear compute the  W b
            self.weight_W_word = nn.Parameter(torch.Tensor(word_gru_hidden_num, word_gru_hidden_num))
            self.bias_word = nn.Parameter(torch.Tensor(word_gru_hidden_num, 1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(word_gru_hidden_num, 1))

        self.softmax_word = nn.Softmax(dim=1)  # batch_first=True dim = 1 为 序列
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1, 0.1)
        self.bias_word.data.uniform_(-0.1, 0.1)

    def forward(self, input, state_word):
        input = torch.tensor(input)
        # embeddings
        embedded = self.lookup(input)
        # word level gru
        # output_word:(batch, seq_len, hidden_size * num_directions)
        # state_word: (batch, num_layers * num_directions, hidden_size)
        output_word, state_word = self.word_gru(embedded, state_word)
        # attention compute
        # 针对隐藏状态 进行 MLP word_squish及相当于 U(it) = tanh(W(w)H(it)+B(w))
        word_squish = attention_batch_tanh_linear(output_word, self.weight_W_word, self.bias_word)
        # 使用上下文向量 weight_proj_word 与整个矩阵 U(it) [sent_num,token_num]
        word_attn = attention_batch_context_matmul(word_squish, self.weight_proj_word)
        # transpose(1, 0)交换是为了使用 softmax 单词的注意力,注意可以使用 softmax 的维度参数,而不用交换行 列
        # 这里使用 softmax维度信息
        #使用 softmax 计算 得到每个词 的重要性 A(it) [sent_num,token_num]
        word_attn_norm = self.softmax_word(word_attn)
        #校验
        sum = 0
        # for i in word_attn_norm[0]:
        #     sum+= i
        # print("sum:{}".format(sum))
        #每一个词的重要性与 隐藏状态相乘 求和得到句向量 [sent_num ,word_gru_hidden_num]
        sent_vectors = attention_mul(output_word, word_attn_norm)
        return sent_vectors, state_word, word_attn_norm

    def init_hidden(self, batch_sentence_size):
        """
        :param batch_sentence_size: 批次大小
        :return:
        """
        num_directions = 2 if self.bidirectional else 1
        # batch_first:   default is True
        # h_0 (batch, num_layers * num_directions, , hidden_size)
        return torch.randn(num_directions, batch_sentence_size, self.word_gru_hidden)





class AttentionSentRNN(nn.Module):

    def __init__(self, sent_gru_hidden, word_result_hidden_num, bidirectional=True):

        super(AttentionSentRNN, self).__init__()
        self.sent_gru_hidden = sent_gru_hidden
        self.word_gru_hidden = word_result_hidden_num
        self.bidirectional = bidirectional
        self.sent_gru = nn.GRU(word_result_hidden_num, sent_gru_hidden, bidirectional=True, batch_first=True)
        if bidirectional == True:
            self.weight_W_sent = nn.Parameter(torch.Tensor(2 * sent_gru_hidden, 2 * sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(2 * sent_gru_hidden))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(2 * sent_gru_hidden, 1))
        else:
            self.weight_W_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(sent_gru_hidden))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, 1))
        self.softmax_sent = nn.Softmax(dim=1)
        self.weight_W_sent.data.uniform_(-0.1, 0.1)
        self.weight_proj_sent.data.uniform_(-0.1, 0.1)

    def forward(self, sent_vectors, state_sent):
        output_sent, state_sent = self.sent_gru(sent_vectors, state_sent)
        # attention compute
        # 针对隐藏状态 进行 MLP word_squish及相当于 U(it) = tanh(W(w)H(it)+B(w))
        sent_squish = attention_batch_tanh_linear(output_sent, self.weight_W_sent, self.bias_sent)
        sent_attn = attention_batch_context_matmul(sent_squish, self.weight_proj_sent)
        sent_attn_norm = self.softmax_sent(sent_attn)
        # 校验
        sum = 0
        for i in sent_attn_norm[0]:
            sum += i
        print("sum:{}".format(sum))
        # 每一个词的重要性与 隐藏状态相乘 求和得到句向量 [sent_num ,word_gru_hidden_num]
        document_vectors = attention_mul(output_sent, sent_attn_norm)
        return document_vectors,state_sent, sent_attn_norm

    def init_hidden(self,batch_size):
        """
        :param batch_sentence_size: 批次大小
        :return:
        """
        num_directions = 2 if self.bidirectional else 1
        # batch_first:   default is True
        # h_0 (batch, num_layers * num_directions, , hidden_size)
        return torch.randn(num_directions, batch_size, self.sent_gru_hidden)


class AttentionFC(nn.Module):
    def __init__(self,attention_type, gru_hidden, n_classes, bidirectional=True):
        super(AttentionFC, self).__init__()
        self.attention_type = attention_type
        self.n_classes = n_classes
        r_gpu_hidden = 2*gru_hidden if bidirectional else gru_hidden
        self.final_linear = nn.Linear(r_gpu_hidden, n_classes)
        self.final_softmax = nn.Softmax()

    def forward(self, attention_vectors, output_state):
        # final classifier
        final_map = self.final_linear(attention_vectors.squeeze(0))
        final_softmax = F.log_softmax(final_map, dim=1)
        return final_softmax



if __name__ == '__main__':
    vocab_size = 100
    embed_size = 200
    word_gru_hidden_num = 100
    max_words = 24
    bidirectional = True
    model = AttentionWordRNN(vocab_size=vocab_size, embed_size=embed_size,
                             word_gru_hidden_num=word_gru_hidden_num, max_words=max_words, bidirectional=bidirectional)
    batch1 = torch.zeros((64, 24), dtype=torch.long)
    for i in range(batch1.shape[0]):
        a = batch1[i]
        for k in range(a.shape[0]):
            a[k] = k
    hidden_state = model.init_hidden(batch_sentence_size=batch1.shape[0])
    sent_vectors, state_word, word_attn_norm = model(batch1,hidden_state)
    print("======sent_vectors======")
    print(sent_vectors)
    # 句子直接进入全连接
    fc_model = AttentionFC("sentence", gru_hidden=100, n_classes=5)
    s_fc = fc_model(sent_vectors,state_word)
    print(s_fc)
    print("=======document==========")
    print(sent_vectors.view(-1,16,200))
    #默认bidirectional = True 所以 word_gru_hidden_num=200
    sent_vectors = sent_vectors.view(-1, 16, 200)
    batch_size = sent_vectors.shape[0]
    sent_gru_hidden = 50
    word_result_hidden_num = 2* word_gru_hidden_num if bidirectional else word_gru_hidden_num
    d_model = AttentionSentRNN(word_result_hidden_num=word_result_hidden_num,
                               sent_gru_hidden=sent_gru_hidden, bidirectional=bidirectional)
    d_hidden_state = d_model.init_hidden(batch_size=sent_vectors.shape[0])
    document_vectors, state_sent, sent_attn_norm = d_model(sent_vectors, d_hidden_state)
    d_fc_model = AttentionFC("document", gru_hidden=50, n_classes=5, bidirectional=True)
    d_fc = d_fc_model(document_vectors,state_sent)
    print(d_fc)
    print("======document_vectors======")
    print(document_vectors)
