#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import re
import pickle

#字典只保留中文,以及情感标点符号
TOKENIZER_RE = re.compile(r"[\u4e00-\u9fa5|!？?!]+", re.UNICODE)


def tokenizer(iterator):
  """Tokenizer generator.

  Args:
    iterator: Input iterator with strings.

  Yields:
    array of tokens per each value in the input.
  """
  for value in iterator:
    yield TOKENIZER_RE.findall(value)




def padding(words,aim_length=200):
    sentence_length = len(words)
    if sentence_length >= aim_length:
        return words[0:aim_length]
    else:
        matrix = [ 0 for i in range(aim_length)]
        for i in range(sentence_length):
            matrix[i] = words[i]        
        return  matrix      


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
        batch_size:每批次的数据量
        num_epochs:数据集被循环多少遍
        
    """
    data = np.array(data)
    data_size = len(data)
    #每次 epoch 多少个批次,这个如果数据不变,num_batches_per_epoch 不会变
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    #epoch 数据提取
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        # 每一次 epoll 都shuffer整个数据集,数据集过大的时候建议通过其他方式打乱
        if shuffle: 
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            
            
def batch_iter_new(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
        batch_size:每批次的数据量
        epochs_no: 第几个 epochs
        
    """
    data = np.array(data)
    data_size = len(data)
    #每次 epoch 多少个批次,这个如果数据不变,num_batches_per_epoch 不会变
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    # 每一次 epoll 都shuffer整个数据集,数据集过大的时候建议通过其他方式打乱
    if shuffle: 
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def vacab_from_text(sent_text_list,label_text_list,
                    words_split = " ",
                    use_stop_word =False,
                    stop_word_path = "stop_word_path",
                    cache_path = "cache_path",
                    mini_count = 0,
                    label_pre=True,
                    name_scope = 'train'):
    cache_path = os.path.join(os.path.curdir, name_scope + "_word_voabulary.pik")
    #if False:
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):#如果缓存文件存在，则直接读取
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index, vocabulary_index2word, label_word2index, label_index2word=pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word, label_word2index, label_index2word
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        label_word2index = {}
        label_index2word = {}
        stop_word={} #停用词
        vocabulary_freq={}
        label_dic = {}
        t_length = len(sent_text_list)
        print("the length of corpus was {}".format(t_length))
        # 文本处理
        for index in range(t_length):
            content =  sent_text_list[index]
            words = content.split(words_split)
            for i in  range(len(words)):
                word = words[i]
                if word in vocabulary_freq:
                    freq = vocabulary_freq[word]
                    vocabulary_freq[word]=freq+1
                else:
                    vocabulary_freq[word] = 1

        # label 处理
        l_length = len(label_text_list)
        for index in range(l_length):
            label = label_text_list[index]
            # 处理 标签
            label_dic[label] = 1

        # if use_stop_word:
        #     with open(stop_word_path,'r') as f:
        #         for line in f.readlines():
        #             sw = line.strip()
        #             stop_word[sw] = 1
        #             if sw in vocabulary_freq.keys():
        #                 vocabulary_freq.pop(sw)
        # 如果不翻转 踢出调低频词后,由于迭代浪费索引,会导致 train_batch 超过 vocab_size 的 bug
        vocabulary_key_sort = sorted(vocabulary_freq.keys(), key=lambda x: vocabulary_freq[x], reverse=True)
        label_dic_sort = sorted(label_dic.keys(), key=lambda x: label_dic[x], reverse=True)
        # word
        for index,value  in enumerate(vocabulary_key_sort):
            if vocabulary_freq[value] > mini_count:
                vocabulary_index2word[index+1] = value
                vocabulary_word2index[value] = index +1
        # label 从 0开始 pytorch loss 问题
        for index,value  in enumerate(label_dic_sort):
            label_index2word[index] = value
            label_word2index[value] = index
        del vocabulary_freq
        del label_dic
        # 特殊补齐填充单词
        vocabulary_word2index['PAD_ID']=0
        vocabulary_index2word[0]='PAD_ID'
        special_index=0

        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path): #如果不存在写到缓存文件中
            with open(cache_path, 'wb') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word,label_word2index,label_index2word), data_f)
                print("pickle dump the vocab")
    return vocabulary_word2index,vocabulary_index2word,label_word2index,label_index2word


def create_voabulary(training_data_path,
                    content_label_split = ";;;",
                    words_split = " ",
                    use_stop_word =False,
                    stop_word_path = "stop_word_path",
                    cache_path = "cache_path",
                    mini_count = 0,
                    label_pre=True,
                    name_scope = 'train'):
    cache_path = os.path.join(os.path.curdir, name_scope + "_word_voabulary.pik")
    #if False:
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):#如果缓存文件存在，则直接读取
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index, vocabulary_index2word, label_word2index, label_index2word=pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word, label_word2index, label_index2word
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        label_word2index = {}
        label_index2word = {}
        stop_word={} #停用词
        vocabulary_freq={}
        label_dic = {}
        with open(training_data_path,"r",encoding="UTF-8") as f:
            for line  in f.readlines():
                data_t = line.split(content_label_split)
                if len(data_t) == 2:
                    content=""
                    label = ""
                    if label_pre:
                        label = data_t[0].strip()
                        content = data_t[1].strip()
                    else:
                        content = data_t[0].strip()
                        label = data_t[1].strip()
                    # 处理内容
                    words = content.split(words_split)
                    for i in  range(len(words)):
                        word = words[i]
                        if word in vocabulary_freq:
                            freq = vocabulary_freq[word]
                            vocabulary_freq[word]=freq+1
                        else:
                            vocabulary_freq[word] = 1
                    # 处理 标签
                    label_dic[label] = 1
                else:
                    continue
        # if use_stop_word:
        #     with open(stop_word_path,'r') as f:
        #         for line in f.readlines():
        #             sw = line.strip()
        #             stop_word[sw] = 1
        #             if sw in vocabulary_freq.keys():
        #                 vocabulary_freq.pop(sw)
        # 如果不翻转 踢出调低频词后,由于迭代浪费索引,会导致 train_batch 超过 vocab_size 的 bug
        vocabulary_key_sort = sorted(vocabulary_freq.keys(), key=lambda x: vocabulary_freq[x], reverse=True)
        label_dic_sort = sorted(label_dic.keys(), key=lambda x: label_dic[x], reverse=True)
        # word
        for index,value  in enumerate(vocabulary_key_sort):
            if vocabulary_freq[value] > mini_count:
                vocabulary_index2word[index+1] = value
                vocabulary_word2index[value] = index +1
        # label
        for index,value  in enumerate(label_dic_sort):
            label_index2word[index+1] = value
            label_word2index[value] = index +1
        del vocabulary_freq
        del label_dic
        # 特殊补齐填充单词
        vocabulary_word2index['PAD_ID']=0
        vocabulary_index2word[0]='PAD_ID'
        special_index=0

        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path): #如果不存在写到缓存文件中
            with open(cache_path, 'wb') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word,label_word2index,label_index2word), data_f)
                print("pickle dump the vocab")
    return vocabulary_word2index,vocabulary_index2word,label_word2index,label_index2word


def load_voabulary(name_scope = 'train'):
    cache_path = os.path.join(os.path.curdir, name_scope + "_word_voabulary.pik")
    if os.path.exists(cache_path):#如果缓存文件存在，则直接读取
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index, vocabulary_index2word, label_word2index, label_index2word=pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word, label_word2index, label_index2word
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        label_word2index = {}
        label_index2word = {}
    return vocabulary_word2index,vocabulary_index2word,label_word2index,label_index2word




def load_data(vocabulary_word2index, vocabulary_word2index_label,
              data_file='train-zhihu4-only-title-all.txt',
              content_label_split=";;;",
              words_split=" ",
              max_word_length=30,
              shuffle = True,
              label_pre = True
              ):
    sent_list = []
    label_list = []
    with open(data_file, "r", encoding="UTF-8") as f:
        for line in f.readlines():
            data_t = line.split(content_label_split)
            if len(data_t) == 2:
                content = ""
                label = ""
                if label_pre:
                    label = data_t[0].strip()
                    content = data_t[1].strip()
                else:
                    content = data_t[0].strip()
                    label = data_t[1].strip()
                # 处理 标签
                if label in vocabulary_word2index_label.keys():
                    label_list.append(vocabulary_word2index_label[label])
                else:
                    # 如果 lable 不满足 词条样本无效
                    continue
                # 处理内容
                words = content.split(words_split)
                for word in words:
                    word_list = list()
                    if word in vocabulary_word2index.keys():
                        word_list.append(vocabulary_word2index.get(word.strip()))
                    else:
                        word_list.append(0)
                sent_list.append(word_list)

            else:
                continue
    x = np.zeros((len(sent_list), max_word_length), dtype=np.int)
    y = np.zeros((len(sent_list), len(vocabulary_word2index_label)), dtype=np.int)
    for index,value in enumerate(sent_list):
        sents = sent_list[index]
        for i in range(max_word_length):
            if i < len(sents):
                x[index][i] = sents[i]
                if sents[i] > len(vocabulary_word2index):
                    print("{}".format(sents[i]))
    # label 已经被替换成 id 这里转矩阵
    for index, label_id in enumerate(label_list):
        # id 从 1开始的所以要 -1
        y[index][label_id - 1] = 1

    del sent_list
    del label_list
    #shuffle_indices = np.random.permutation(np.arange(len(y)))
    if shuffle:
        shuffle_indices = list(np.arange(len(x)))
        np.random.seed(10)
        np.random.shuffle(shuffle_indices)
        x_shuffle = x[shuffle_indices]
        y_shuffle = y[shuffle_indices]
        del x
        del y
        return x_shuffle,y_shuffle
    else:
        return x,y


def load_text_data(data_file,content_label_split=";;;",load_label=False,label_pre =True):
    sent_text_list = []
    label_text_list = []
    with open(data_file, "r", encoding="UTF-8") as f:
        for line in f.readlines():
            data_t = line.strip().split(content_label_split)
            if len(data_t) ==2:
                content = ""
                label = ""
                if label_pre:
                    label = data_t[0].strip()
                    content = data_t[1].strip()
                else:
                    content = data_t[0].strip()
                    label = data_t[1].strip()
                sent_text_list.append(content)
                if load_label:
                    label_text_list.append(label)
            else:
                #如果长度不为2取 train_batch =0 即认为整条内容都是文本 content
                sent_text_list.append(data_t[0])

    return sent_text_list,label_text_list


def prepare_sequence(vocabulary_word2index, vocabulary_word2index_label,
              x_text,
              y_text,
              words_split=" ",
              max_word_length=30
              ):
    sent_list = []
    label_list = []

    for index,content in enumerate(x_text):
        if len(x_text) == len(y_text):
            # 处理 标签
            label = y_text[index]
            if label in vocabulary_word2index_label.keys():
                label_list.append(vocabulary_word2index_label[label])
            else:
                # 如果 lable 不满足 词条样本无效
                continue
        # 处理内容
        words = content.split(words_split)
        for word in words:
            word_list = list()
            if word in vocabulary_word2index.keys():
                word_list.append(vocabulary_word2index.get(word.strip()))
            else:
                word_list.append(0)
        sent_list.append(word_list)

    x = np.zeros((len(sent_list), max_word_length), dtype=np.int)
    y = np.zeros((len(sent_list), len(vocabulary_word2index_label)), dtype=np.int)
    for index, value in enumerate(sent_list):
        sents = sent_list[index]
        for i in range(max_word_length):
            if i < len(sents):
                x[index][i] = sents[i]
                # 词汇 id 不应该短语词汇的长度
                if sents[i] > len(vocabulary_word2index):
                    print(" error-vacab-size:{}".format(sents[i]))
    #有选项决定是否加载处理label
    if len(x_text) == len(y_text):
        # label 已经被替换成 id 这里转矩阵
        for index, label_id in enumerate(label_list):
            # id 从 1开始的所以要 -1
            y[index][label_id - 1] = 1

    del sent_list
    # shuffle_indices = np.random.permutation(np.arange(len(y)))
    #print("load_data.ended...")
    return x,y,label_list


def shuffle(data_list,seed=10):
    shuffle_indices = list(np.arange(len(data_list)))
    if seed == 0:
        np.random.seed()
    else:
        np.random.seed(seed)
    np.random.shuffle(shuffle_indices)
    #data_shuffle = data_list[shuffle_indices]
    data_shuffle = [data_list[indices] for indices in  shuffle_indices]
    return data_shuffle
