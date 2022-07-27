# -*- coding: utf-8 -*-
import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))

import torch
import nltk
from Main_model_lastest.model.dmkcm import DMKCM
from Main_model_lastest.configs_manage import DVKCM_config
import torch.nn as nn
import json
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from Main_model_lastest.pre_process.utils_for_preprocess import PreProcessingHelper

import spacy
import networkx as nx

# import nltk
# nltk.download('punkt')


import heapq


def getListMaxNumIndex(num_list, topk=5):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    max_num_index = list(map(num_list.index, heapq.nlargest(topk, num_list)))
    # print('max_num_index:', list(max_num_index))
    max_num = []
    for idx in max_num_index:
        max_num.append(num_list[idx])
    return max_num_index, max_num


def list2str(li):
    st = ''
    for i in range(len(li)):
        if i == 0:
            st = li[i]
        else:
            st = st + ' ' + li[i]
    return st


def get_vocab_map(concepts_tensor, vocab):
    new_vocab_map_tensor = []
    new_vocab_mask_tensor = []
    for batch_con in concepts_tensor.transpose(0, 1):
        concepts = batch_con[batch_con != 1].tolist()
        vocab_map = []
        map_mask = []
        for idx in vocab.keys():  # 遍历词表的索引值
            try:
                pos = concepts.index(idx)  # _concept_ids中词的位置
                vocab_map.append(pos)
                map_mask.append(1)
            except:
                vocab_map.append(0)
                map_mask.append(0)
        new_vocab_map_tensor.append(torch.tensor(vocab_map).unsqueeze(1))
        new_vocab_mask_tensor.append(torch.tensor(map_mask).unsqueeze(1))

    return torch.cat(new_vocab_map_tensor, 1).long().to(concepts_tensor.device), \
           torch.cat(new_vocab_mask_tensor, 1).long().to(concepts_tensor.device)


def pad(tensor, pad_len=50):
    batch = tensor.size(1)
    tensor_len = tensor.size(0)
    pad_tensor = torch.ones(pad_len - tensor_len, batch).to(tensor.device).long()
    padded_tensor = torch.cat([tensor, pad_tensor], 0)
    return padded_tensor


class Predict():
    def __init__(self, data, itos, dmkcm_config, params_path, vocab, device):
        self.config = dmkcm_config
        self.vocab_dict = vocab
        self.itos = itos
        self.device = device
        self.init_model(data, itos, dmkcm_config, params_path)

        # self.fc = fc

    def get_indexs(self, vocab_dict, tokens):
        indexed = []
        for t in tokens:
            try:
                num = vocab_dict[t]
            except:
                num = vocab_dict["<unk>"]
            indexed.append(num)
        return indexed

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def greedy_decode(self, model, src, src_mask, inner, max_len, start_symbol,
                      concepts=None, labels=None, distances=None, relations=None, head_ids=None, tail_ids=None,
                      triple_labels=None, vocab_map=None, vocab_mask=None, memory=None, turn=None
                      ):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        encoded, memory, o_in_attn, o_m_attn = model.encode(src, src_mask, inner, memory, turn)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len - 1):
            encoded = encoded.to(self.device)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(self.device)
            out, triple_de_weight, gate = model.decode(ys, encoded, tgt_mask,
                                                       concepts, labels, distances, relations, head_ids, tail_ids,
                                                       triple_labels, vocab_map, vocab_mask)
            out = out.transpose(0, 1)
            if self.config.hop2:
                prob = out[:, -1]
            else:
                prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == 3:
                break
        return ys, memory, o_in_attn, o_m_attn, triple_de_weight, gate

    def pre_response(self, his_con, post, inner, model, device, vocab_dict, turn_num, memory, memory_mask, max_len=30,
                     concepts=None, labels=None, distances=None, relations=None, head_ids=None, tail_ids=None,
                     triple_labels=None, vocab_map=None, vocab_mask=None):
        model.eval()
        if his_con == '<NULL>':
            his_con = [his_con]
        if isinstance(post, str):
            pp = PreProcessingHelper()
            post = pp.repla(post)
            tokens = [token.lower() for token in post.split()]
        else:
            tokens = [token.lower() for token in post]
        if isinstance(his_con, str):
            pp = PreProcessingHelper()
            his_con = pp.repla(his_con)
            if isinstance(his_con, str):
                his_tokens = [token.lower() for token in his_con]
            else:
                his_tokens = his_con
        else:
            his_tokens = [token.lower() for token in his_con]

        init_token = '<sos>'
        eos_token = '<eos>'
        tokens = [init_token] + tokens + [eos_token]
        if his_con == '<null>':
            his_tokens = [init_token] + [his_con] + [eos_token]
        else:
            his_tokens = [init_token] + his_tokens + [eos_token]

        src_indexes = self.get_indexs(vocab_dict, tokens)
        his_indexes = self.get_indexs(vocab_dict, his_tokens)

        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
        his_tensor = torch.LongTensor(his_indexes).unsqueeze(1).to(device)

        his_len = torch.tensor([his_tensor.size(0)]).unsqueeze(1).to(device)
        src_len = torch.tensor([src_tensor.size(0)]).unsqueeze(1).to(device)

        # src_tensor = pad(src_tensor)  # batch=1不用pad
        # his_tensor = pad(his_tensor)

        inner_list = []
        for v in inner:
            # hop1_tokens = [init_token] + nltk.word_tokenize(v.lower()) + [eos_token]  # 没用
            try:
                hop1_tokens = [init_token] + v.split(' ') + [eos_token]  # predict
            except:
                hop1_tokens = [init_token] + v + [eos_token]  # evaluate
            hop1_indexes = self.get_indexs(vocab_dict, hop1_tokens)
            inner_list.append(torch.LongTensor(hop1_indexes).unsqueeze(1).to(device))

        inner_all = []
        inner_maxlen = 0
        for inn in inner_list:
            if len(inn) > inner_maxlen:
                inner_maxlen = len(inn)
        for inn in inner_list:
            inner_all.append(pad(inn, inner_maxlen))

        trg_indexes = [vocab_dict[init_token]]

        # '''
        print('===============================')
        # print('His: ', list2str(his_con))
        # print('Post: ', list2str(post))
        if self.config.encoding_mode != 'no_his':
            if isinstance(his_con, str):
                print('His: ', his_con)
            else:
                print('His: ', list2str(his_con))
        if self.config.only_hop1 or self.config.memory:
            for idx, inn in enumerate(inner):
                if isinstance(inn, str):
                    print('inner: ', inn)
                else:
                    print('1hop_s%s: ' % idx, list2str(inn))
        if self.config.hop2:
            concept_list = concepts
            print('2hop: ', list2str(concepts))
        if isinstance(post, str):
            print('post: ', post)
        else:
            print('post: ', list2str(post))
        # '''

        if self.config.hop2:
            concepts = self.get_indexs(vocab_dict, concepts)
            concepts = torch.LongTensor(concepts).unsqueeze(1).to(device)
            labels = torch.LongTensor(labels).unsqueeze(1).to(device)
            distances = torch.LongTensor(distances).unsqueeze(1).to(device)
            relations = torch.LongTensor(relations).unsqueeze(1).to(device)
            head_ids = torch.LongTensor(head_ids).unsqueeze(1).to(device)
            tail_ids = torch.LongTensor(tail_ids).unsqueeze(1).to(device)
            triple_labels = torch.LongTensor(triple_labels).unsqueeze(1).to(device)
            vocab_map, vocab_mask = get_vocab_map(concepts, {value: key for key, value in vocab_dict.items()})

        if turn_num == 0:
            memory = None
            memory_mask = None

        num_tokens = src_tensor.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        src = src_tensor
        his = his_tensor
        if self.config.encoding_mode == 'no_his':
            src = src
            src_mask = src_mask
        elif self.config.encoding_mode == 'inp_cat_2':
            src = torch.cat([his, src], 0)
            src_seq_len = src.shape[0]
            src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)
        elif self.config.encoding_mode == 'his_src_inp':
            his_src_max_len = his_len.max().item() + src_len.max().item() - 2
            his_src_inp = []
            for idx, b in enumerate(his.transpose(0, 1)):
                new_b = torch.cat([b[:his_len[idx].item() - 1], src.transpose(0, 1)[idx][1:src_len[idx].item()]], -1)
                if new_b.size(0) < his_src_max_len:
                    new_b = torch.cat([new_b, torch.ones(his_src_max_len - new_b.size(0)).long().to(his.device)], -1)
                his_src_inp.append(new_b.unsqueeze(0))
            src = torch.cat(his_src_inp).to(src.device).transpose(0, 1)
            src_seq_len = src.shape[0]
            src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

        temp_memory, temp_memory_mask = None, None

        tgt_tokens, temp_memory, o_in_attn, o_m_attn, triple_attn, gate = self.greedy_decode(
            model, src, src_mask, inner_all, max_len=num_tokens + 5, start_symbol=2,
            concepts=concepts, labels=labels, distances=distances, relations=relations, head_ids=head_ids,
            tail_ids=tail_ids,
            triple_labels=triple_labels, vocab_map=vocab_map, vocab_mask=vocab_mask, memory=memory, turn=turn_num
        )
        tgt_tokens = tgt_tokens.flatten()
        trg_tokens = [self.itos[i] for i in tgt_tokens.tolist()]
        res = ''
        for i in trg_tokens[1:-1]:
            # for i in trg_tokens:
            res += i + ' '

        print('bot: ', res)
        # '''
        # attn virualization
        if o_in_attn != None:
            print('8 heads inner attention scores: ', o_in_attn.max(-1)[0].max(-1)[0].tolist()[0])
            # display_attention(list2str(post).split(), res.split(), o_in_attn)
        if o_m_attn != None:
            # display_attention(list2str(post).split(), res.split(), o_m_attn)
            print('8 heads memory attention scores: ', o_m_attn.max(-1)[0].max(-1)[0].tolist()[0])
        if triple_attn != None:
            # display_attention(list2str(post).split(), res.split(), triple_attn)
            triple_attn_dict = {}
            for idx, attn in enumerate(triple_attn.tolist()[0]):
                max_idx, max_prob = getListMaxNumIndex(attn)
                triple_attn_dict[trg_tokens[1:][idx]] = list(zip([concept_list[m_i] if m_i < len(concept_list) else concept_list[0] for m_i in max_idx], max_prob))

            print('triple attention scores: ', triple_attn_dict)
            # print('triple attention scores: ', triple_attn.tolist())
            print('gate scores: ', gate.tolist())
        # '''

        return res, temp_memory, temp_memory_mask

    def init_model(self, data, itos, dmkcm_config, params_path):
        word_map = len(itos)
        SRC_VOCAB_SIZE = word_map
        TGT_VOCAB_SIZE = word_map
        EMB_SIZE = 512
        NHEAD = 8
        FFN_HID_DIM = 512
        BATCH_SIZE = 128
        NUM_ENCODER_LAYERS = 3
        NUM_DECODER_LAYERS = 3

        self.model = DMKCM(dmkcm_config, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                           NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).to(self.device)

        # load params
        print(params_path)
        save_model = torch.load(params_path)
        model_dict = self.model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)
        # self.model.load_state_dict(torch.load(params_path, map_location='cuda'))
        print("load params")
        return

    def get_res(self, his_con, post,
                inner,
                turn_num, memory, memory_mask=None,
                concepts=None, labels=None, distances=None, relations=None, head_ids=None, tail_ids=None,
                triple_labels=None, vocab_map=None, vocab_mask=None):
        response, memory, memory_mask = self.pre_response(his_con, post, inner, self.model, self.device,
                                                          self.vocab_dict,
                                                          turn_num, memory, memory_mask,
                                                          concepts=concepts, labels=labels, distances=distances,
                                                          relations=relations,
                                                          head_ids=head_ids, tail_ids=tail_ids,
                                                          triple_labels=triple_labels, vocab_map=vocab_map,
                                                          vocab_mask=vocab_mask
                                                          )
        return response, memory, memory_mask


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    # '''
    attention = attention[:, -1, :len(translation), :len(sentence)]
    fig = plt.figure(figsize=(15, 15))  # 图片大小
    ax = fig.add_subplot(111)

    attention = attention.squeeze(0).cpu().detach().numpy()[:len(translation)]
    # attention = attention.cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)  # 标签大小
    # ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'], rotation=45)
    ax.set_xticklabels([''] + [t.lower() for t in sentence], rotation=90)  # label的角度
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # '''

    '''
    assert n_rows * n_cols == n_heads

    attention = attention[:, :, :len(translation), :len(sentence)]
    
    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                           rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    '''
    plt.savefig('fig_cat.png')
    plt.show()
    plt.close()


def load_resources():
    global concept2id, relation2id, id2relation, id2concept, concept_embs, relation_embs
    concept2id = {}
    id2concept = {}
    with open('./data/knowledge/concept.txt', "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    with open('./data/knowledge/relation.txt', "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)

    print("relation2id done")


def load_cpnet():
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple
    print("loading cpnet....")
    cpnet = nx.read_gpickle('./data/knowledge/cpnet.graph')
    print("Done")

    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):  # u, v是否连通
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def load_total_concepts(data_path):
    global concept2id, total_concepts_id, config
    total_concepts = []
    total_concepts_id = []
    exs = []
    for path in [data_path + "/concepts_nv.json"]:
        with open(path, 'r') as f:
            for line in f.readlines():
                line = json.loads(line)
                total_concepts.extend(line['qc'] + line['ac'])

        total_concepts = list(set(total_concepts))  # 所有相关的concept

    for x in total_concepts:
        if concept2id.get(x, False):  # 检查是否在concept2id里
            total_concepts_id.append(concept2id[x])  # 所有concepts的index


def read_model_vocab(data_path):
    global model_vocab, vocab_dict
    vocab_dict = json.loads(open(data_path, 'r').readlines()[0])
    model_vocab = []
    for tok in vocab_dict.keys():
        model_vocab.append(tok)
    print(len(model_vocab))


def get_outer(post):
    def hard_ground(sent):
        # dataset与concept做匹配
        # 遍历句子中每个单词词干化后看是否属于cpnet_vocab和model_vocab，并且t.pos_词性是"NOUN" or "VERB"
        global cpnet_vocab, model_vocab
        sent = sent
        doc = nlp(sent)
        res = set()
        for t in doc:
            if t.lemma_ in cpnet_vocab and t.lemma_ not in blacklist and t.lemma_ in model_vocab:
                if t.pos_ == "NOUN" or t.pos_ == "VERB":
                    res.add(t.lemma_)
        return res

    def match_mentioned_concepts(sents):
        # global nlp
        # matcher = load_matcher(nlp)
        concepts = []
        question_concepts = hard_ground(sents)  # 来自输入的concepts
        return list(question_concepts)

    def get_edge(src_concept, tgt_concept):
        global cpnet, concept2id, relation2id, id2relation, id2concept
        try:
            rel_list = cpnet[src_concept][tgt_concept]
            return list(set([rel_list[item]["rel"] for item in rel_list]))  # 为什么有两个边，哪里来的
        except:
            return []

    def find_neighbours_frequency(source_concepts, T, max_B=100):
        global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple, total_concepts_id
        source = [concept2id[s_cpt] for s_cpt in source_concepts]  # source_concepts变成id
        start = source
        Vts = dict([(x, 0) for x in start])  # 开始节点，距离 {8742: 0, 11203: 0, 8107: 0}
        Ets = {}  # 存储T-hop的路径
        total_concepts_id_set = set(total_concepts_id)
        for t in range(T):
            V = {}  # 每一步的存储路径节点
            templates = []
            for s in start:  # 遍历source中每一个node的id
                if s in cpnet_simple:  # node的id在graph中
                    for n in cpnet_simple[s]:  # 遍历s在graph中的每个连接node
                        if n not in Vts and n in total_concepts_id_set:  # 如果连接的node不连接自己并且属于图中的node
                            if n not in Vts:  # 如果连接的node不连接自己
                                if n not in V:  # 如果连接的node不在V中
                                    V[n] = 1  # 连接node的频率为1
                                else:
                                    V[n] += 1  # 连接node的频率+1

                            if n not in Ets:  # 如果连接的node不在E中
                                rels = get_edge(s, n)  # 获得edge(rel)
                                if len(rels) > 0:  # 如果有边
                                    Ets[n] = {s: rels}  # 加入E
                            else:  # 如果连接的node在E中
                                rels = get_edge(s, n)  # 获得变
                                if len(rels) > 0:  # 如果有边
                                    Ets[n].update({s: rels})  # 更新E

            V = list(V.items())
            count_V = sorted(V, key=lambda x: x[1], reverse=True)[:max_B]  # 取前一百个最少跳的节点
            start = [x[0] for x in count_V if x[0] in total_concepts_id_set]  # 将连接到的node作为start节点

            Vts.update(dict([(x, t + 1) for x in start]))  # 更新开始节点

        _concepts = list(Vts.keys())
        _distances = list(Vts.values())  # 距离
        concepts = []  # 存储所有连接的concepts
        distances = []
        for c, d in zip(_concepts, _distances):
            concepts.append(c)
            distances.append(d)

        triples = []
        for v, N in Ets.items():
            if v in concepts:
                for u, rels in N.items():
                    if u in concepts:
                        triples.append((u, rels, v))  # 存储u到达v的三元组

        res = [id2concept[x].replace("_", " ") for x in concepts]  # 连接到的concepts
        triples = [(id2concept[x].replace("_", " "), y, id2concept[z].replace("_", " ")) for (x, y, z) in triples]

        return {"concepts": res, "distances": distances, "triples": triples}

    def bfs(start, triple_dict, source):
        paths = [[[start]]]
        stop = False
        shortest_paths = []
        count = 0
        while 1:
            last_paths = paths[-1]
            new_paths = []
            for path in last_paths:
                if triple_dict.get(path[-1], False):
                    triples = triple_dict[path[-1]]
                    for triple in triples:
                        new_paths.append(path + [triple[0]])

            # print(new_paths)
            for path in new_paths:
                if path[-1] in source:
                    stop = True
                    shortest_paths.append(path)

            if count == 2:
                break
            paths.append(new_paths)
            count += 1

        return shortest_paths

    def filter_directed_triple(e, max_concepts=200, max_triples=300):
        count = 0
        gt_triple_count = 0
        _data = []
        max_len = 0
        max_neighbors = 5
        triple_dict = {}
        triples = e['triples']
        concepts = e['concepts']
        labels = e['labels']
        distances = e['distances']

        for t in triples:
            head, tail = t[0], t[-1]
            head_id = concepts.index(head)
            tail_id = concepts.index(tail)
            if distances[head_id] <= distances[tail_id]:
                if t[-1] not in triple_dict:
                    triple_dict[t[-1]] = [t]
                else:
                    if len(triple_dict[t[-1]]) < max_neighbors:
                        triple_dict[t[-1]].append(t)

        starts = []
        for l, c in zip(labels, concepts):
            if l == 1:
                starts.append(c)
        # print(starts)

        sources = []
        for d, c in zip(distances, concepts):
            if d == 0:
                sources.append(c)

        shortest_paths = []
        for start in starts:
            shortest_paths.extend(bfs(start, triple_dict, sources))

        ground_truth_triples = []
        for path in shortest_paths:
            for i, n in enumerate(path[:-1]):
                ground_truth_triples.append((n, path[i + 1]))

        ground_truth_triples_set = set(ground_truth_triples)

        _triples = []
        triple_labels = []
        for k, v in triple_dict.items():
            for t in v:
                _triples.append(t)
                if (t[-1], t[0]) in ground_truth_triples_set:
                    triple_labels.append(1)
                else:
                    triple_labels.append(0)

        concepts = concepts[:max_concepts]
        _triples = _triples[:max_triples]
        triple_labels = triple_labels[:max_triples]

        heads = []
        tails = []
        for triple in _triples:
            heads.append(concepts.index(triple[0]))
            tails.append(concepts.index(triple[-1]))

        max_len = max(max_len, len(_triples))
        e['relations'] = [x[1][0] for x in _triples]
        e['head_ids'] = heads
        e['tail_ids'] = tails
        e['triple_labels'] = triple_labels
        e.pop('triples')

        return e

    def get_vocab_map(kg, vocab):
        vocab_map = []
        map_mask = []
        for idx in vocab.keys():  # 遍历词表的索引值
            try:
                # pos = kg['concepts'].index(idx)  # _concept_ids中词的位置
                pos = kg.index(idx)  # _concept_ids中词的位置
                vocab_map.append(pos)
                map_mask.append(1)
            except:
                vocab_map.append(0)
                map_mask.append(0)
        return vocab_map, map_mask

    concept = match_mentioned_concepts(post)
    e = find_neighbours_frequency(concept, 1, 100)
    e['labels'] = [0] * len(post.split(' '))
    e = filter_directed_triple(e)
    e['vocab_map'], e['vocab_mask'] = get_vocab_map(e['concepts'], vocab_dict)
    return e


if __name__ == '__main__':

    DATA_PATH = './data/dailyDialog'
    dmkcm_config = DVKCM_config()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    print(device)
    torch.cuda.set_device(2)
    print('正在使用显卡设备: ', torch.cuda.current_device())

    name = '*'

    dmkcm_config.dataChoice = 'dailyDialog'  # personaChat / dailyDialog
    dmkcm_config.only_hop1 = False
    dmkcm_config.memory = True
    dmkcm_config.hop2 = True
    dmkcm_config.paral = False
    dmkcm_config.his_mode = 'last_one_his'
    dmkcm_config.encoding_mode = 'inp_cat_2'

    if dmkcm_config.his_mode == 'all_his':
        vocab_name = '**'  # all_his vocab
    elif dmkcm_config.his_mode == 'last_one_his' and int(name[2:4]) < 25:
        vocab_name = '**_his'  # last_one_his vocab
    else:
        vocab_name = '**'  # last_one_his new vocab
    print('using vocab: %s' % vocab_name)

    cur_num = 70
    params_path = dmkcm_config.checkPoint_path + dmkcm_config.dataChoice + '/DMKCM_epoch_model_' + name + '/dmkcm_' + name + '_epoch' + str(
        cur_num)

    word_dict_path = dmkcm_config.base_path + 'data/' + dmkcm_config.dataChoice + '/vocab_data/vocab_' + vocab_name + '.json'
    vocab = json.load(open(word_dict_path))
    itos = {value: key for key, value in vocab.items()}

    # predict
    # post = input('please input post:')
    post = 'hi, how are you doing today ?'

    if dmkcm_config.only_hop1 or dmkcm_config.memory:
        from Main_model_lastest.model.utils_for_predict import PredictorHelper_DRKIT

        pre = PredictorHelper_DRKIT(dmkcm_config.dataChoice)

    if dmkcm_config.hop2:
        read_model_vocab(DATA_PATH + '/vocab_data/vocab_{}.json'.format(vocab_name))
        load_resources()  # 加载concept2id, relation2id, id2relation, id2concept
        load_cpnet()  # 加载cpnet graph
        load_total_concepts(DATA_PATH)  # 加载
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])  # spaCy加载预训练语言模型
        blacklist = set(
            ["from", "as", "more", "either", "in", "and", "on", "an", "when", "too", "to", "i", "do", "can", "be",
             "that",
             "or",
             "the", "a", "of", "for", "is", "was", "the", "-PRON-", "actually", "likely", "possibly", "want",
             "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
             "one", "something", "sometimes", "everybody", "somebody", "could", "could_be", "mine", "us", "em",
             "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above",
             "abst",
             "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae",
             "af",
             "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all",
             "allow",
             "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
             "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone",
             "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar",
             "are",
             "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available",
             "aw",
             "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became",
             "been",
             "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond",
             "bi",
             "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu",
             "but",
             "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce",
             "certain",
             "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come",
             "comes",
             "con",
             "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp",
             "cq",
             "cr",
             "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de",
             "definitely",
             "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does",
             "doesn",
             "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy",
             "e",
             "E",
             "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej",
             "el",
             "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er",
             "es",
             "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone",
             "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc",
             "few",
             "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo",
             "followed",
             "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from",
             "front",
             "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets",
             "getting",
             "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr",
             "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt",
             "have",
             "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres",
             "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr",
             "hs",
             "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie",
             "if",
             "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index",
             "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into",
             "inward",
             "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr",
             "js",
             "jt",
             "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la",
             "largely",
             "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let",
             "lets",
             "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks",
             "los",
             "lr",
             "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me",
             "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn",
             "mo",
             "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must",
             "mustn",
             "my",
             "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily",
             "neither",
             "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non",
             "none",
             "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns",
             "nt",
             "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og",
             "oh",
             "oi",
             "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op",
             "oq",
             "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow",
             "owing",
             "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular",
             "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl",
             "placed",
             "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably",
             "previously",
             "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj",
             "qu",
             "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily",
             "really",
             "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related",
             "relatively",
             "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj",
             "rl",
             "rm",
             "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say",
             "saying",
             "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems",
             "seen",
             "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns",
             "shows",
             "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some",
             "somehow",
             "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically",
             "specified",
             "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially",
             "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3",
             "take",
             "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks",
             "thanx",
             "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter",
             "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon",
             "these",
             "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those",
             "thou",
             "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til",
             "tip",
             "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr",
             "tried",
             "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U",
             "u201d",
             "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until",
             "unto",
             "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually",
             "ut",
             "v",
             "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs",
             "vt",
             "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went",
             "were",
             "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter",
             "whereas",
             "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither",
             "who",
             "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within",
             "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3",
             "xf",
             "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl",
             "you",
             "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"])
        with open('./data/knowledge/concept.txt', "r", encoding="utf8") as f:
            cpnet_vocab = [l.strip() for l in list(f.readlines())]
        cpnet_vocab = set([c.replace("_", " ") for c in cpnet_vocab])

    turn_num = 0
    memory = None
    memory_mask = None
    pos = "<NULL>"
    if len(post):
        predict = Predict('dailyDialog', itos, dmkcm_config, params_path, vocab, device)
    while len(post):
        # pos = ''  # 指定历史为空
        if dmkcm_config.only_hop1 or dmkcm_config.memory:
            inner = pre.run(post)
            print(inner)
        else:
            inner = []
        if dmkcm_config.hop2:
            outer = get_outer(post + pos)
            print(outer['concepts'])
            response, memory, memory_mask = predict.get_res(pos, post, inner, turn_num, memory, memory_mask,
                                                            outer['concepts'],
                                                            outer['labels'], outer['distances'], outer['relations'],
                                                            outer['head_ids'], outer['tail_ids'],
                                                            outer['triple_labels'],
                                                            outer['vocab_map'], outer['vocab_mask'])
        else:
            response, memory, memory_mask = predict.get_res(pos, post, inner, turn_num, memory, memory_mask)
        turn_num += 1
        post = input('please input post:')
        # post = response
        if turn_num == 8:
            break
    print('thank you, bye~')
