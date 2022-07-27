# -*- coding: utf-8 -*-
from torchtext import data
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
from torchtext.data import BucketIterator
import numpy as np
import random
from Main_model_lastest.model.example_new import Example
import torch
import torch.nn as nn
import torch.utils.data
import json
import os

global config

def tensor2str(his, itos):
    return ' '.join([itos[i] for i in his.tolist()])


def get_index(vocab_dict, tokens):
    indexed = []
    for t in tokens:
        try:
            num = vocab_dict[t]
        except:
            num = vocab_dict["<unk>"]
        indexed.append(num)
    return indexed


class MyDataset(data.Dataset):
    name = 'grand dataset'
    # 配置信息
    def __init__(self, config, path, fields, vocab):
        examples = []
        csv_data = pd.read_csv(path)
        print('read data from {}'.format(path))
        kg_path = '/'.join(path.split('/')[:-1]) + '/kg.csv'
        if config.hop2:
            if os.path.exists(kg_path):
                kgframe = pd.read_csv(kg_path)
                print('read kg from {}'.format(kg_path))
            else:
                kg = []
                with open('/'.join(path.split('/')[:-1]) + '/1hops_100_directed_triple_filter.json', 'r') as f:
                    for line in f.readlines():
                        line_data = json.loads(line)
                        kg.append(line_data)
                kgframe = pd.DataFrame(columns=('t11', 't22', 't33', 't44'), index=None)
                for idx, line in tqdm(enumerate(kg), total=len(kg)//4):
                    try:
                        idx0 = 0 + idx * 4
                        idx1 = 1 + idx * 4
                        idx2 = 2 + idx * 4
                        idx3 = 3 + idx * 4

                        kg[idx0]['relations'] = [x[0] for x in kg[idx0]['relations']]
                        kg[idx1]['relations'] = [x[0] for x in kg[idx1]['relations']]
                        kg[idx2]['relations'] = [x[0] for x in kg[idx2]['relations']]
                        kg[idx3]['relations'] = [x[0] for x in kg[idx3]['relations']]

                        kgframe = kgframe.append(
                            [{'t11': list(kg[idx0].values()), 't22': list(kg[idx1].values()),
                              't33': list(kg[idx2].values()), 't44': list(kg[idx3].values())}],
                        ignore_index=True)
                    except:
                        break
                kgframe.to_csv(kg_path)
            tmp = zip(csv_data['t1'], csv_data['t2'], csv_data['t3'], csv_data['t4'], kgframe['t11'], kgframe['t22'],
                      kgframe['t33'], kgframe['t44'])
            for i, l in tqdm(enumerate(tmp), total=len(csv_data)):
                if config.debug:
                    if i > 50:
                        break
                # t1_hop1 = get_index(vocab, eval(l[0])[3][0])
                # t1_hop2 = get_index(vocab, eval(l[0])[3][1])
                # t1_hop3 = get_index(vocab, eval(l[0])[3][2])
                # t1_hop4 = get_index(vocab, eval(l[0])[3][3])
                # t1_hop5 = get_index(vocab, eval(l[0])[3][4])
                #
                # t2_hop1 = get_index(vocab, eval(l[1])[3][0])
                # t2_hop2 = get_index(vocab, eval(l[1])[3][1])
                # t2_hop3 = get_index(vocab, eval(l[1])[3][2])
                # t2_hop4 = get_index(vocab, eval(l[1])[3][3])
                # t2_hop5 = get_index(vocab, eval(l[1])[3][4])
                #
                # t3_hop1 = get_index(vocab, eval(l[2])[3][0])
                # t3_hop2 = get_index(vocab, eval(l[2])[3][1])
                # t3_hop3 = get_index(vocab, eval(l[2])[3][2])
                # t3_hop4 = get_index(vocab, eval(l[2])[3][3])
                # t3_hop5 = get_index(vocab, eval(l[2])[3][4])
                #
                # t4_hop1 = get_index(vocab, eval(l[3])[3][0])
                # t4_hop2 = get_index(vocab, eval(l[3])[3][1])
                # t4_hop3 = get_index(vocab, eval(l[3])[3][2])
                # t4_hop4 = get_index(vocab, eval(l[3])[3][3])
                # t4_hop5 = get_index(vocab, eval(l[3])[3][4])

                concept0 = get_index(vocab, eval(l[4])[0])
                concept1 = get_index(vocab, eval(l[5])[0])
                concept2 = get_index(vocab, eval(l[6])[0])
                concept3 = get_index(vocab, eval(l[7])[0])

                try:
                    if config.his_mode == 'last_one_his':
                        examples.append(Example.fromlist_my([
                            (
                                eval(l[0])[0], eval(l[0])[1], eval(l[0])[2],
                                eval(l[0])[3][0], eval(l[0])[3][1], eval(l[0])[3][2], eval(l[0])[3][3],
                                eval(l[0])[3][4],
                                concept0, eval(l[4])[1], eval(l[4])[2],
                                eval(l[4])[3], eval(l[4])[4], eval(l[4])[5], eval(l[4])[6],
                            ),  # t1
                            (
                                eval(l[1])[0], eval(l[1])[1], eval(l[1])[2],
                                eval(l[1])[3][0], eval(l[1])[3][1], eval(l[1])[3][2], eval(l[1])[3][3],
                                eval(l[1])[3][4],
                                concept1, eval(l[5])[1], eval(l[5])[2],
                                eval(l[5])[3], eval(l[5])[4], eval(l[5])[5], eval(l[5])[6],
                            ),  # t2
                            (
                                eval(l[1])[1] + eval(l[1])[2], eval(l[2])[1], eval(l[2])[2],
                                eval(l[2])[3][0], eval(l[2])[3][1], eval(l[2])[3][2], eval(l[2])[3][3],
                                eval(l[2])[3][4],
                                concept2, eval(l[6])[1], eval(l[6])[2],
                                eval(l[6])[3], eval(l[6])[4], eval(l[6])[5], eval(l[6])[6],
                            ),  # t3
                            (
                                eval(l[2])[1] + eval(l[2])[2], eval(l[3])[1], eval(l[3])[2],  # t4
                                eval(l[3])[3][0], eval(l[3])[3][1], eval(l[3])[3][2], eval(l[3])[3][3],
                                eval(l[3])[3][4],
                                concept3, eval(l[7])[1], eval(l[7])[2],
                                eval(l[7])[3], eval(l[7])[4], eval(l[7])[5], eval(l[7])[6],
                            )
                        ],
                            fields))
                    else:  # all_his
                        examples.append(Example.fromlist_my([
                            (
                                eval(l[0])[0], eval(l[0])[1], eval(l[0])[2],
                                eval(l[0])[3][0], eval(l[0])[3][1], eval(l[0])[3][2], eval(l[0])[3][3],
                                eval(l[0])[3][4],
                                concept0, eval(l[4])[1], eval(l[4])[2],
                                eval(l[4])[3], eval(l[4])[4], eval(l[4])[5], eval(l[4])[6],
                            ),  # t1
                            (
                                eval(l[1])[0], eval(l[1])[1], eval(l[1])[2],
                                eval(l[1])[3][0], eval(l[1])[3][1], eval(l[1])[3][2], eval(l[1])[3][3],
                                eval(l[1])[3][4],
                                concept1, eval(l[5])[1], eval(l[5])[2],
                                eval(l[5])[3], eval(l[5])[4], eval(l[5])[5], eval(l[5])[6],
                            ),  # t2
                            (
                                eval(l[2])[0], eval(l[2])[1], eval(l[2])[2],
                                eval(l[2])[3][0], eval(l[2])[3][1], eval(l[2])[3][2], eval(l[2])[3][3],
                                eval(l[2])[3][4],
                                concept2, eval(l[6])[1], eval(l[6])[2],
                                eval(l[6])[3], eval(l[6])[4], eval(l[6])[5], eval(l[6])[6],
                            ),  # t3
                            (
                                eval(l[3])[0], eval(l[3])[1], eval(l[3])[2],  # t4
                                eval(l[3])[3][0], eval(l[3])[3][1], eval(l[3])[3][2], eval(l[3])[3][3],
                                eval(l[3])[3][4],
                                concept3, eval(l[7])[1], eval(l[7])[2],
                                eval(l[7])[3], eval(l[7])[4], eval(l[7])[5], eval(l[7])[6],
                            )
                        ],
                            fields))
                except:
                    print('IndexError')
                    continue
        else:
            tmp = zip(csv_data['t1'], csv_data['t2'], csv_data['t3'], csv_data['t4'])
            for i, l in tqdm(enumerate(tmp), total=len(csv_data)):
                if config.debug:
                    if i > 50:
                        break
                if config.his_mode == 'last_one_his':
                    try:
                        examples.append(Example.fromlist_my([
                            (
                                eval(l[0])[0], eval(l[0])[1], eval(l[0])[2],
                                eval(l[0])[3][0], eval(l[0])[3][1], eval(l[0])[3][2], eval(l[0])[3][3], eval(l[0])[3][4],
                            ),  # t1
                            (
                                eval(l[1])[0], eval(l[1])[1], eval(l[1])[2],
                                eval(l[1])[3][0], eval(l[1])[3][1], eval(l[1])[3][2], eval(l[1])[3][3], eval(l[1])[3][4],
                            ),  # t2
                            (
                                eval(l[1])[1] + eval(l[1])[2], eval(l[2])[1], eval(l[2])[2],
                                # eval(l[2])[0][len(eval(l[1])[0]):], eval(l[2])[1], eval(l[2])[2],
                                eval(l[2])[3][0], eval(l[2])[3][1], eval(l[2])[3][2], eval(l[2])[3][3], eval(l[2])[3][4],
                            ),  # t3
                            (
                                eval(l[2])[1] + eval(l[2])[2], eval(l[3])[1], eval(l[3])[2],  # t4
                                # eval(l[3])[0][len(eval(l[2])[0]):], eval(l[3])[1], eval(l[3])[2],  # t4
                                eval(l[3])[3][0], eval(l[3])[3][1], eval(l[3])[3][2], eval(l[3])[3][3], eval(l[3])[3][4],
                            )
                        ],
                            fields))
                    except:
                        print('index_{} has no 1hop'.format(i))
                        continue
                else:  # all_his
                    examples.append(Example.fromlist_my([
                        (
                            eval(l[0])[0], eval(l[0])[1], eval(l[0])[2],
                            eval(l[0])[3][0], eval(l[0])[3][1], eval(l[0])[3][2], eval(l[0])[3][3], eval(l[0])[3][4],
                        ),  # t1
                        (
                            eval(l[1])[0], eval(l[1])[1], eval(l[1])[2],
                            eval(l[1])[3][0], eval(l[1])[3][1], eval(l[1])[3][2], eval(l[1])[3][3], eval(l[1])[3][4],
                        ),  # t2
                        (
                            # eval(l[2])[0][len(eval(l[1])[0]):], eval(l[2])[1], eval(l[2])[2],
                            eval(l[2])[0], eval(l[2])[1], eval(l[2])[2],
                            eval(l[2])[3][0], eval(l[2])[3][1], eval(l[2])[3][2], eval(l[2])[3][3], eval(l[2])[3][4],
                        ),  # t3
                        (
                            # eval(l[3])[0][len(eval(l[2])[0]):], eval(l[3])[1], eval(l[3])[2],  # t4
                            eval(l[3])[0], eval(l[3])[1], eval(l[3])[2],  # t4
                            eval(l[3])[3][0], eval(l[3])[3][1], eval(l[3])[3][2], eval(l[3])[3][3], eval(l[3])[3][4],
                        )
                    ],
                        fields))

        super(MyDataset, self).__init__(examples, fields)

def DataLoader(config, device, POST, HOP1, CONCEPTS, LABELS, DISTENCES, H_T_R, TRIPLE_LABELS):
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    if config.hop2:
        fields = [[('h1', 'p1', 'r1',
                    'one_hop1_1', 'one_hop1_2', 'one_hop1_3', 'one_hop1_4', 'one_hop1_5',
                    'concepts1', 'labels1', 'distances1', 'relations1', 'head_ids1', 'tail_ids1', 'triple_labels1'),
                   (POST, POST, POST, POST, POST, POST, POST, POST, CONCEPTS, LABELS, DISTENCES, H_T_R, H_T_R, H_T_R,
                    TRIPLE_LABELS)],
                  [('h2', 'p2', 'r2',
                    'one_hop2_1', 'one_hop2_2', 'one_hop2_3', 'one_hop2_4', 'one_hop2_5',
                    'concepts2', 'labels2', 'distances2', 'relations2', 'head_ids2', 'tail_ids2', 'triple_labels2'),
                   (POST, POST, POST, POST, POST, POST, POST, POST, CONCEPTS, LABELS, DISTENCES, H_T_R, H_T_R, H_T_R,
                    TRIPLE_LABELS)],
                  [('h3', 'p3', 'r3', 'one_hop3_1', 'one_hop3_2', 'one_hop3_3', 'one_hop3_4', 'one_hop3_5',
                    'concepts3', 'labels3', 'distances3', 'relations3', 'head_ids3', 'tail_ids3', 'triple_labels3'),
                   (POST, POST, POST, POST, POST, POST, POST, POST, CONCEPTS, LABELS, DISTENCES, H_T_R, H_T_R, H_T_R,
                    TRIPLE_LABELS)],
                  [('h4', 'p4', 'r4', 'one_hop4_1', 'one_hop4_2', 'one_hop4_3', 'one_hop4_4', 'one_hop4_5',
                    'concepts4', 'labels4', 'distances4', 'relations4', 'head_ids4', 'tail_ids4', 'triple_labels4'),
                   (POST, POST, POST, POST, POST, POST, POST, POST, CONCEPTS, LABELS, DISTENCES, H_T_R, H_T_R, H_T_R,
                    TRIPLE_LABELS)]]
    else:
        fields = [[('h1', 'p1', 'r1', 'one_hop1_1', 'one_hop1_2', 'one_hop1_3', 'one_hop1_4', 'one_hop1_5'),
                   (POST, POST, POST, POST, POST, POST, POST, POST)],
                  [('h2', 'p2', 'r2', 'one_hop2_1', 'one_hop2_2', 'one_hop2_3', 'one_hop2_4', 'one_hop2_5'),
                   (POST, POST, POST, POST, POST, POST, POST, POST)],
                  [('h3', 'p3', 'r3', 'one_hop3_1', 'one_hop3_2', 'one_hop3_3', 'one_hop3_4', 'one_hop3_5'),
                   (POST, POST, POST, POST, POST, POST, POST, POST)],
                  [('h4', 'p4', 'r4', 'one_hop4_1', 'one_hop4_2', 'one_hop4_3', 'one_hop4_4', 'one_hop4_5'),
                   (POST, POST, POST, POST, POST, POST, POST, POST)]]

    # vocab = None
    if config.write_vocab_dict:
        vocab = None
    else:
        vocab = json.load(open(config.word_dict_path))
    train_data = MyDataset(config, config.train_data_path, fields=fields, vocab=vocab)
    valid_data = MyDataset(config, config.val_data_path, fields=fields, vocab=vocab)
    # train_data, valid_data = dataset.split([0.9, 0.1], random_state=random.seed(SEED))

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    # print(f"Number of testing examples: {len(test_data.examples)}")
    # print(vars(train_data.examples[0]))

    # build vocab  min_freq:最小出现频次是2次;unk_init:初始化train_data中不存在预训练词向量词表中的单词

    # POST.build_vocab(train_data, vectors="glove.6B.300d", min_freq=2, unk_init=torch.Tensor.normal_, specials=['<his>', '<src>'])
    POST.build_vocab(train_data, vectors="glove.840B.300d", min_freq=2, unk_init=torch.Tensor.normal_, specials=['<his>', '<src>'])
    print(f"POST Unique tokens in source (de) vocabulary: {len(POST.vocab)}")

    # embed = build_vocab(config, vocab)
    embed = None

    # wirte vocab for predict
    if config.write_vocab_dict:
        word_dict = dict(POST.vocab.stoi.items())
        word_dict_path = config.word_dict_path
        with open(word_dict_path, 'w') as f:
            json_str = json.dumps(word_dict)
            f.write(json_str)

    BATCH_SIZE = config.BATCH_SIZE

    # BucketIterator = BucketIterator(sort=False)
    train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data),
                                                           batch_size=BATCH_SIZE,
                                                           sort_key=lambda x: len(x.h1),
                                                           device=device, shuffle=True)

    return POST, train_iterator, valid_iterator, device, embed


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def create_masks(device, question, reply_input, reply_target):
    # def subsequent_mask(size):
    #     mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
    #     return mask.unsqueeze(0)
    #
    # question_mask = (question != 1).to(device)
    # question_mask = question_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, max_words)
    #
    # reply_input_mask = reply_input != 1
    # reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words)
    # reply_input_mask = reply_input_mask & subsequent_mask(reply_input.size(-1)).type_as(reply_input_mask.data)
    # reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words, max_words)
    # reply_target_mask = reply_target != 1  # (batch_size, max_words)
    #
    # return question_mask, reply_input_mask, reply_target_mask
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    src_seq_len = question.shape[0]
    tgt_seq_len = reply_input.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (question == 1).transpose(0, 1)
    tgt_padding_mask = (reply_input == 1).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

class LossWithLS(nn.Module):

    def __init__(self, size, smooth):
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size

    def forward(self, prediction, target, mask):
        """
        prediction of shape: (batch_size, max_words, vocab_size)
        target and mask of shape: (batch_size, max_words)
        """
        prediction = prediction.view(-1, prediction.size(-1))  # (batch_size * max_words, vocab_size)
        target = target.contiguous().view(-1)  # (batch_size * max_words)
        mask = mask.float()
        # mask = mask.view(-1)  # (batch_size * max_words)
        mask = mask.reshape(-1)  # (batch_size * max_words)
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = self.criterion(prediction, labels)  # (batch_size * max_words, vocab_size)
        loss = (loss.sum(1) * mask).sum() / mask.sum()
        return loss