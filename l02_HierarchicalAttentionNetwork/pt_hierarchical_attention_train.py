#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# author bjlizhipeng

"""

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pt_hierarchical_attention_model import AttentionWordRNN ,AttentionFC
import pt_data_utils as data_utils
torch.manual_seed(1)


if __name__ == '__main__':
    embed_size = 200
    word_gru_hidden_num = 100
    max_words = 24
    batch_sentence_size=50
    bidirectional = True
    vocab_size = 100
    epoll_num = 200
    dev_sample_percentage =0.1

    sent_text_list, label_text_list = data_utils.load_text_data("../data/corpus_0.txt",
                              content_label_split=";;;",
                              load_label=True)
    corpus_size = len(sent_text_list)
    # 检测样本数量大小
    if corpus_size == 0:
        print("corpus_size:{} ".format(corpus_size))
        exit(-1)
    vocabulary_word2index, vocabulary_index2word, label_word2index, label_index2word=data_utils.vacab_from_text(
        sent_text_list,label_text_list)
    vocab_size = len(vocabulary_word2index)
    n_classes = len(label_word2index)
    # 检测 词汇数量 和 分类数量
    if vocab_size == 0 or n_classes == 0:
        print("vocab_size:{},n_classes:{}".format(vocab_size,n_classes))
        exit(-1)

    # 训练集 测试集 拆分
    data_list = list(zip(sent_text_list, label_text_list))
    data_list = data_utils.shuffle(data_list)
    x, y = zip(*data_list)
    del data_list
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(x)))
    x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
    y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
    del x
    del y
    train_length = len(x_train)
    dev_length = len(x_dev)
    print("train/dev = {}/{} ".format(train_length, dev_length))
    #构建模型
    word_model = AttentionWordRNN(vocab_size=vocab_size, embed_size=embed_size,
                                  word_gru_hidden_num=word_gru_hidden_num,
                                  max_words=max_words, bidirectional=bidirectional)
    fc_model = AttentionFC("sentence", gru_hidden=word_gru_hidden_num, n_classes=n_classes)
    # 选择损失函数
    #loss_function = nn.NLLLoss()
    loss_function = nn.CrossEntropyLoss()
    #构建两个模型的学习器
    w_optimizer = optim.Adam(word_model.parameters(), lr=0.1)
    fc_optimizer = optim.Adam(fc_model.parameters(), lr=0.1)
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()

    # with torch.no_grad():
    #     #inputs = []
    #     #tag_scores = model(inputs)
    #     print("befor train the model see the result")
    # 开始训练
    global_step = 1
    for epoch in range(epoll_num):  # again, normally you would NOT do 300 epochs, it is toy data

        # 每一次训练都打乱一下训练数据
        train_data_list = list(zip(x_train,y_train))
        if global_step == 1:
            train_data_list = data_utils.shuffle(train_data_list)
        x_tra, y_tra = zip(*train_data_list)
        x_t_length = len(x_tra)
        x_dev_length = len(x_dev)
        train_batch_num = int(x_t_length) // batch_sentence_size
        train_batch_num = train_batch_num if x_t_length % batch_sentence_size == 0 else train_batch_num + 1
        dev_batch_num = int(x_dev_length) // batch_sentence_size
        dev_batch_num = dev_batch_num if x_dev_length % batch_sentence_size ==0 else train_batch_num + 1

        for train_batch  in range(train_batch_num):
            if global_step >1 and global_step % 50 == 0:
                # 满足测试条件 验证测试集
                test_loss=0.0
                test_acc=0.0
                for test_batch in range(dev_batch_num):
                    start_index = test_batch * batch_sentence_size
                    end_index = start_index + batch_sentence_size
                    # list [:] endindex 自动 -1
                    end_index = end_index if end_index <= x_dev_length else x_dev_length
                    x_dev_batch = x_dev[start_index:end_index]
                    y_dev_batch = y_dev[start_index:end_index]
                    x_dev_id, y_dev_id,_ = data_utils.prepare_sequence(vocabulary_word2index, label_word2index,
                                                                         x_dev_batch, y_dev_batch,
                                                                         max_word_length=30)
                    torch.set_grad_enabled(False)
                    #with torch.no_grad:
                    word_hidden = word_model.init_hidden(batch_sentence_size=x_dev_id.shape[0])
                    sent_vectors, state_word, word_attn_norm = word_model(x_dev_id, word_hidden)
                    # print("======sent_vectors======")
                    # print(sent_vectors)

                    # Step 4. Compute the loss, gradients, and update the parameters by
                    #  calling optimizer.step()
                    y_dev_id = torch.tensor(y_dev_id)
                    # 句子直接进入全连接
                    #以下两个 argmax 为了保证同纬度比较
                    y_score = fc_model(sent_vectors, state_word)
                    y_dev_id = torch.argmax(y_dev_id, dim=1, keepdim=True)

                    y_pre = torch.argmax(y_score, dim=1, keepdim=True)
                    compare = torch.eq(y_dev_id, y_pre).view(-1)
                    sum = torch.sum(compare, dim=0)
                    acc = float(sum) / y_pre.shape[0]
                    # 满足 loss 计算需要
                    # 由 [batch,1] 变成 [batch]
                    y_dev_id=y_dev_id.squeeze(1)
                    test_loss += loss_function(y_score, y_dev_id)
                    test_acc += acc
                print("the -Evaluation- train_batch_num:{}/{} of epoll:{}  the test acc:{}  loss:{} ,the global_step:{}".format(
                    train_batch, train_batch_num, epoch, test_acc/dev_batch_num, test_loss/dev_batch_num, global_step))
                torch.set_grad_enabled(True)

            start_index = train_batch * batch_sentence_size
            end_index = start_index+batch_sentence_size
            # list [:] endindex 自动 -1
            end_index = end_index if end_index <= x_t_length else x_t_length

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            word_model.zero_grad()
            fc_model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            x_train_batch = x_tra[start_index:end_index]
            y_train_batch = y_tra[start_index:end_index]
            x_train_id,y_train_id,_ = data_utils.prepare_sequence(vocabulary_word2index, label_word2index,
                                        x_train_batch,y_train_batch, max_word_length=30)

            # Step 3. Run our forward pass.
            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            word_hidden = word_model.init_hidden(batch_sentence_size=x_train_id.shape[0])
            sent_vectors, state_word, word_attn_norm = word_model(x_train_id, word_hidden)
            #print("======sent_vectors======")
            #print(sent_vectors)
            # 句子直接进入全连接
            y_score = fc_model(sent_vectors, state_word)
            y_train_id = torch.tensor(y_train_id, dtype=torch.double)
            # 以下两个 argmax 为了保证同纬度比较
            y_pre = torch.argmax(y_score, dim=1, keepdim=True)
            y_target_id = torch.argmax(y_train_id, dim=1, keepdim=True)
            # compute acc
            compare = torch.eq(y_pre, y_target_id).view(-1)
            sum = torch.sum(compare, dim=0)
            acc = float(sum) / y_pre.shape[0]

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            # 满足 loss 计算需要
            y_target_id=y_target_id.squeeze(1)
            #(torch.where(y_target_id>y_target_id.shape[0], y_target_id,y_target_id))
            loss = loss_function(y_score, y_target_id,)
            loss.backward()
            w_optimizer.step()
            fc_optimizer.step()
            print("the train train_batch_num:{}/{} of epoll:{}  the acc:{}  loss:{} ,the global_step:{}".format(
                train_batch, train_batch_num, epoch, acc, loss, global_step))
            global_step+=1

    # See what the scores are after training
    # with torch.no_grad():
    #     inputs = prepare_sequence(training_data[0][0], word_to_ix, 24)
    #     word_hidden = word_model.init_hidden(batch_sentence_size=x.shape[0])
    #     sent_vectors, state_word, word_attn_norm = word_model(inputs, word_hidden)
    #     print("======sent_vectors======")
    #     print(sent_vectors)
    #     # 句子直接进入全连接
    #     y_score = fc_model(sent_vectors, state_word)
    #     y_pre = torch.argmax(y_score, dim=1)
    #     print(y_pre)

