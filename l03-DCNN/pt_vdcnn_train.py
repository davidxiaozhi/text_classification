# -*- coding:UTF-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import pt_data_utils_dcnn as data_utils
from pt_vdcnn_model import VDCNN
import lib
import argparse

def str2bool(s):
    if s == "True":
        return True
    else:
        return False

def get_args():
    parser = argparse.ArgumentParser("""
    Very Deep CNN with optional residual connections (https://arxiv.org/abs/1606.01781)
    """)
    #rdc-catalog-train.tsv
    parser.add_argument("--train_file", type=str, default='../data/rdc-catalog-train-new.tsv')
    parser.add_argument("--train_add_rate", type=int, default=1)
    parser.add_argument("--tag", type=str, default='tag')
    parser.add_argument("--content_label_split", type=str, default='\t', help=" the label and the text split str")
    parser.add_argument("--model_folder", type=str, default="../data/DCNN/")
    parser.add_argument("--depth", type=int, choices=[9, 17, 29, 49], default=9, help="Depth of the network tested in the paper (9, 17, 29, 49)")
    parser.add_argument("--maxlen", type=int, default=30)
    # parser.add_argument('--shortcut', action='store_true', default=False)
    parser.add_argument('--shortcut', type=str2bool, default=False)
    parser.add_argument('--label_pre', type=str2bool, default=False)
    # parser.add_argument('--shuffle', action='store_true', default=False, help="shuffle train and test sets")
    parser.add_argument('--shuffle', type=str2bool, default=True, help="shuffle train and test sets")
    parser.add_argument("--batch_size", type=int, default=100, help="number of example read by the gpu")
    parser.add_argument("--epoch_num", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_halve_interval", type=float, default=5, help="Number of iterations before halving learning rate")
    parser.add_argument("--class_weights", nargs='+', type=float, default=None)
    parser.add_argument("--evaluation_step", type=int, default=500, help="Number of iterations between testing phases")
    parser.add_argument('--gpu', type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--last_pooling_layer", type=str, choices=['k-max-pooling', 'max-pooling'], default='k-max-pooling', help="type of last pooling layer")

    args = parser.parse_args()
    return args



if __name__ == "__main__":

    opt = get_args()

    if not os.path.exists(opt.model_folder):
        os.makedirs(opt.model_folder)
    use_gpu =(opt.gpu and torch.cuda.is_available())
    batch_size = opt.batch_size
    evaluation_step = opt.evaluation_step
    seed = opt.seed
    model_path = opt.model_folder
    tag = opt.tag
    content_label_split = opt.content_label_split


    logger = lib.get_logger(logdir=opt.model_folder, logname="logs.txt")
    logger.info("parameters: {}".format(vars(opt)))
    #训练集加载
    sent_text_list, label_text_list = data_utils.load_text_data(opt.train_file,
                                                                content_label_split=content_label_split,
                                                                load_label=True,label_pre=opt.label_pre)
    corpus_size = len(sent_text_list)
    # 检测样本数量大小
    if corpus_size == 0:
        print("corpus_size:{} ".format(corpus_size))
        exit(-1)
    vocabulary_word2index, vocabulary_index2word, label_word2index, label_index2word = data_utils.vacab_from_text(
        sent_text_list, label_text_list, name_scope=opt.tag)
    vocab_size = len(vocabulary_word2index)
    n_classes = len(label_word2index)
    # 检测 词汇数量 和 分类数量
    if vocab_size == 0 or n_classes == 0:
        logger.info("vocab_size:{},n_classes:{}".format(vocab_size, n_classes))
        exit(-1)
    logger.info("vocab_size:{},n_classes:{}".format(vocab_size, n_classes))
    # 训练集 测试集 拆分
    data_list = list(zip(sent_text_list, label_text_list))
    data_list = data_utils.shuffle(data_list, seed=0)
    x, y = zip(*data_list)
    del data_list
    dev_sample_index = -1 * int(0.1 * float(len(x)))
    x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
    y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
    del x
    del y
    # 训练集增加的倍数
    x_train = list(x_train)
    y_train = list(y_train)
    for i in range(opt.train_add_rate):
        x_train += x_train
        y_train += y_train
    train_length = len(x_train)
    dev_length = len(x_dev)
    logger.info("train/dev = {}/{} ".format(train_length, dev_length))

    torch.manual_seed(opt.seed)
    print("Seed for random numbers: ", torch.initial_seed())

    model = VDCNN(n_classes=n_classes, num_embedding=vocab_size, embedding_dim=32, depth=opt.depth, n_fc_neurons=128,
                 shortcut=False)
    # model = VDCNN(n_classes=n_classes, vocab_size=vocab_size, embedding_dim=32, n_gram=3,
    #               n_fc_neurons=512)
    print(model)
    if use_gpu:
        model.cuda()
        #model=nn.DataParallel(model)
    if opt.class_weights:
        criterion = nn.CrossEntropyLoss(torch.cuda.FloatTensor(opt.class_weights))
    else:
        criterion = nn.CrossEntropyLoss()

    #optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, amsgrad=True)
    model.train()
    global_step = 1

    if use_gpu:
        print("=========使用 gpu 进行训练===========")
        torch.cuda.manual_seed(1)
    best_acc = 0.0
    epoch_num=opt.epoch_num
    for epoch in range(epoch_num):  # again, normally you would NOT do 300 epochs, it is toy data
        # 每一epoll次训练都打乱一下训练数据
        train_data_list = list(zip(x_train, y_train))
        if epoch > 1:
            if opt.shuffle:
                #每次打乱都雷同那就没必要了
                train_data_list = data_utils.shuffle(train_data_list, 0)
        x_tra, y_tra = zip(*train_data_list)
        x_t_length = len(x_tra)
        x_dev_length = len(x_dev)
        train_batch_num = int(x_t_length) // batch_size
        train_batch_num = train_batch_num if x_t_length % batch_size == 0 else train_batch_num + 1
        dev_batch_num = int(x_dev_length) // opt.batch_size
        dev_batch_num = dev_batch_num if x_dev_length % batch_size == 0 else dev_batch_num + 1

        for batch_num in range(train_batch_num):

            if global_step > 1 and global_step % evaluation_step == 0:
                # 满足测试条件 验证测试集
                # 开启验证模式
                model.eval()
                torch.set_grad_enabled(False)
                test_loss = 0.0
                test_acc = 0.0
                for test_batch in range(dev_batch_num):
                    start_index = test_batch * batch_size
                    end_index = start_index + batch_size
                    # list [:] endindex 自动 -1
                    end_index = end_index if end_index <= x_dev_length else x_dev_length
                    x_dev_batch = x_dev[start_index:end_index]
                    y_dev_batch = y_dev[start_index:end_index]
                    x_dev_id, y_dev_id, _ = data_utils.prepare_sequence(vocabulary_word2index, label_word2index,
                                                                        x_dev_batch, y_dev_batch,
                                                                        max_word_length=opt.maxlen)
                    # with torch.no_grad:
                    x_dev_id = torch.tensor(x_dev_id)
                    y_dev_id = torch.tensor(y_dev_id)
                    if use_gpu:
                        x_dev_id = x_dev_id.cuda()
                        y_dev_id = y_dev_id.cuda()

                    # Step 4. Compute the loss, gradients, and update the parameters by
                    #  calling optimizer.step()
                    # 句子直接进入全连接
                    # 以下两个 argmax 为了保证同纬度比较
                    y_score = model(x_dev_id)
                    y_dev_id = torch.argmax(y_dev_id, dim=1, keepdim=True)
                    y_pre = torch.argmax(y_score, dim=1, keepdim=True)
                    compare = torch.eq(y_dev_id, y_pre).view(-1)
                    sum = torch.sum(compare, dim=0)
                    acc = float(sum) / y_pre.shape[0]
                    # 满足 loss 计算需要
                    # 由 [batch,1] 变成 [batch]
                    y_dev_id = y_dev_id.squeeze(1)
                    test_loss += criterion(y_score, y_dev_id)
                    test_acc += acc
                avg_acc = test_acc / dev_batch_num
                avg_loss = test_loss / dev_batch_num

                if avg_acc > best_acc:
                    best_acc = avg_acc
                    save_model_path = os.path.join(model_path, tag)
                    # if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path, exist_ok=True)
                    torch.save(model, os.path.join(save_model_path, "vdcnn.model"))
                    print("save the model for the best_test_acc:{} global_step:{} epoch:{}/{} ".format(best_acc,
                                                                                                       global_step,
                                                                                                       epoch,
                                                                                                       epoch_num))
                else:
                    print(
                        "the -Evaluation- train_batch_num:{}/{} of epoch:{}/{}  the test acc:{}  loss:{} ,best_test_acc:{} the global_step:{} ".format(
                            batch_num, train_batch_num, epoch, epoch_num, avg_acc, avg_loss, best_acc, global_step))
                torch.set_grad_enabled(True)
                model.train()
            start_index = batch_num * batch_size
            end_index = start_index + batch_size
            # list [:] endindex 自动 -1
            end_index = end_index if end_index <= x_t_length else x_t_length

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            x_train_batch = x_tra[start_index:end_index]
            y_train_batch = y_tra[start_index:end_index]
            x_train_id, y_train_id, _ = data_utils.prepare_sequence(vocabulary_word2index, label_word2index,
                                                                    x_train_batch, y_train_batch, max_word_length=opt.maxlen)

            optimizer.zero_grad()
            x_train_id = torch.tensor(x_train_id)
            y_train_id = torch.tensor(y_train_id, dtype=torch.double)
            if use_gpu:
                x_train_id = x_train_id.cuda()
                y_train_id = y_train_id.cuda()
            # print("======sent_vectors======")
            # print(sent_vectors)
            # 句子直接进入全连接
            y_score = model(x_train_id)

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
            y_target_id = y_target_id.squeeze(1)
            # (torch.where(y_target_id>y_target_id.shape[0], y_target_id,y_target_id))
            loss = criterion(y_score, y_target_id)
            loss.backward()
            # for f in model.parameters():
            #     #print("befor:",f.data)
            #     if hasattr(f, "data") and hasattr(f.grad, "data"):
            #         f.data.sub_(f.grad.data * opt.lr)
            # w_optimizer.step()p
            optimizer.step()
            #print("x[0][1].embed-after:", model.embed.weight.data[x_train_id[0][1]])
            print(
                "the train train_batch_num:{}/{} of epoch:{}/{}  the acc:{}  loss:{} ,best_test_acc:{} the global_step:{}".format(
                    batch_num, train_batch_num, epoch, epoch_num, acc, loss, best_acc, global_step))

            #if global_step % 2 == 1:
            #for m in model.modules():
            # for name, m in model.named_modules():
            #     if isinstance(m, nn.Embedding):
            #         print(name, "-:-", m.weight.grad)
            #     if isinstance(m, nn.Conv1d):
            #         print(name,"-:-",m.weight.grad)
            #     if isinstance(m, nn.Linear):
            #         print(name, "-:-", m.weight.grad)
            #print("y_pre:", torch.cat((y_target_id.unsqueeze(1),y_pre),dim = 1))
            global_step += 1
        if epoch % opt.lr_halve_interval == 0 and epoch > 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            lr /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logger.info("new lr: {}".format(lr))

