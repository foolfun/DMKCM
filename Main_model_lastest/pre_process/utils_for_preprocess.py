# -*- coding: utf-8 -*-
import pandas as pd
import nltk
from tqdm import tqdm
from Main_model_lastest.configs_manage import preProcess
import os
import json
# from Main_model_lastest.pre_process.knowledge.processROCstory import getHop1


# 处理数据时需要用到的一些函数
class PreProcessingHelper():
    def __init__(self, data_name="personaChat"):
        pre_ce = preProcess(data_name)
        candi_ = pre_ce.base_path + 'pre_process/convai2/candi_keyword.txt'
        candi_keyword_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), candi_)  # 这个是从wordnet里面筛选的
        _candiwords = [x.strip() for x in open(candi_keyword_path).readlines()]
        self.vocab = _candiwords
        self.data_name = data_name
        with open(pre_ce.keywords_file, 'r') as f:
            self.kws = f.read().splitlines()
        tmp_df = pd.read_csv(pre_ce.knowledge_path + "indexedStory.csv")
        self.ids = tmp_df["id"].tolist()
        self.s_kws = [eval(i) for i in tmp_df["s_kws"].tolist()]
        self.passages = tmp_df["passage"].tolist()
        self.relations = [eval(i) for i in tmp_df["relation"].tolist()]

    def repla(self, text):
        return text.lower().replace(".", " . ").replace(",", " , ").replace(";", " ; ").replace("?", " ? ") \
            .replace(r'! .', '!').replace("!", " ! ") \
            .replace("n't", ' not').replace("n ’ t", ' not') \
            .replace("i'm", 'i am').replace("i ’ m", 'i am') \
            .replace("'ll", ' will').replace("’ ll", 'will') \
            .replace('__SILENCE__', '<SILENCE>') \
            .replace("'ve", ' have').replace(" ’ ve", ' have').replace("\"", "").replace("  ", " ").strip()

    def getE(self, sent_keywords):
        vocab = self.vocab
        toks = []
        for i in sent_keywords:
            if i in vocab:
                toks.append(i)
        return toks

    def split_data(self, data_path):
        # personaChat、dailyDialog、hollE
        df = pd.read_csv(data_path + '{}_data_all.csv'.format(self.data_name))
        sum_cnt = len(df)
        split_rate = 0.1
        split_num = int(sum_cnt * split_rate)
        test_df = df[0:split_num]
        val_df = df[split_num:split_num * 2]
        train_df = df[split_num * 2:]
        test_df.to_csv(data_path + '{}_data_test.csv'.format(self.data_name), header=True, index=False)
        val_df.to_csv(data_path + '{}_data_val.csv'.format(self.data_name), header=True, index=False)
        train_df.to_csv(data_path + '{}_data_train.csv'.format(self.data_name), header=True, index=False)

    def process_data_with_knowledge(self, entity_file, num=10):

        with open(file=entity_file, encoding="utf-8") as f:
            entity_lines = f.read().splitlines()

        def getKnowledgeDial(line_dic):
            if line_dic["his_con"] == '<NULL>':
                his_con = line_dic["his_con"]
            else:
                his_con = nltk.word_tokenize(line_dic["his_con"])
            if line_dic["post"] == '<SILENCE>':
                post = line_dic["post"].lower()
            else:
                post = nltk.word_tokenize(line_dic["post"])
            response = nltk.word_tokenize(line_dic["response"])
            ents = line_dic["entities"][0]
            hop1_cont = [ents["s1"], ents["s2"], ents["s3"], ents["s4"], ents["s5"]]
            hop2_dic = ents["hop2_dic"]

            hop1 = []
            for v in hop1_cont:
                hop1.append(nltk.word_tokenize(v))
            return his_con, post, response, hop1, hop2_dic

        t_dialog = []
        data = []
        for line in tqdm(entity_lines):
            line_dic = eval(line)
            if (int(line_dic["_id"]) + 1) % 4 == 0:
                his_con, post, response, hop1, hop2_dic = getKnowledgeDial(line_dic)
                t_dialog.append([his_con, post, response, hop1, hop2_dic])
                data.append([t_dialog[0], t_dialog[1], t_dialog[2], t_dialog[3]])
                t_dialog = []
            else:
                his_con, post, response, hop1, hop2_dic = getKnowledgeDial(line_dic)
                t_dialog.append([his_con, post, response, hop1, hop2_dic])

        return data

    def split_str(self, sent):
        return sent.split(" ")

    def process_data_with_knowledge_drkit(self, entity_file):

        with open(file=entity_file, encoding="utf-8") as f:
            entity_lines = f.read().splitlines()

        def getKnowledgeDial(line_dic):
            if line_dic["his_con"] == '<NULL>':
                his_con = line_dic["his_con"]
            else:
                # his_con = nltk.word_tokenize(line_dic["his_con"])
                his_con = self.split_str(self.repla(line_dic["his_con"]))
            if line_dic["post"] == '<SILENCE>':
                post = line_dic["post"].lower()
            else:
                post = self.split_str(self.repla(line_dic["post"]))
            response = self.split_str(self.repla(line_dic["response"]))

            hop1_cont = getHop1(self.ids, self.s_kws, self.passages,
                                self.relations, line_dic["post"], line_dic["response"])
            # hop1_cont = line_dic["hop1"]
            if len(hop1_cont) == 0:
                # print("hop1 is []")
                return his_con, post, response, []
            hop1 = []
            for v in hop1_cont:
                hop1.append(self.split_str(self.repla(v)))
            return his_con, post, response, hop1

        t_dialog = []
        data = []
        for line in tqdm(entity_lines):
            line_dic = eval(line)
            if (int(line_dic["_id"]) + 1) % 4 == 0:  # 这是一段对话的最后一轮
                his_con, post, response, hop1 = getKnowledgeDial(line_dic)
                t_dialog.append([his_con, post, response, hop1])
                data.append([t_dialog[0], t_dialog[1], t_dialog[2], t_dialog[3]])
                t_dialog = []  # 一段新对话开始
            else:
                his_con, post, response, hop1 = getKnowledgeDial(line_dic)
                t_dialog.append([his_con, post, response, hop1])

        return data

    def getVocab(self, data_dialog_file, vocab_file):
        '''
        :param data_dialog_file: input_file
        :param vocab_file: output_file
        :return:
        '''
        data_df = pd.read_csv(data_dialog_file)
        vocab_dic = {}
        vocab_dic["<unk>"] = 0
        vocab_dic["<pad>"] = 1
        vocab_dic["<sos>"] = 2
        vocab_dic["<eos>"] = 3
        vocab_dic["<NULL>"] = 4
        vocab_dic["<silence>"] = 5
        indx = 6

        # 更新词表
        def updaterVocab(sents: list, vocabs: dict, index: int):
            for h in sents:
                if h in vocab_dic.keys():
                    continue
                else:
                    vocab_dic[h] = index
                    index += 1
            return vocabs, index

        # 构建词表
        for row in tqdm(data_df.itertuples()):
            t1 = eval(row.t1)
            post, response, hop1 = t1[1], t1[2], t1[3]
            vocab_dic, indx = updaterVocab(post, vocab_dic, indx)
            vocab_dic, indx = updaterVocab(response, vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[0], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[1], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[2], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[3], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[4], vocab_dic, indx)
            t2 = eval(row.t2)
            post, response, hop1 = t2[1], t1[2], t1[3]
            vocab_dic, indx = updaterVocab(post, vocab_dic, indx)
            vocab_dic, indx = updaterVocab(response, vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[0], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[1], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[2], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[3], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[4], vocab_dic, indx)
            t3 = eval(row.t3)
            post, response, hop1 = t3[1], t1[2], t1[3]
            vocab_dic, indx = updaterVocab(post, vocab_dic, indx)
            vocab_dic, indx = updaterVocab(response, vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[0], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[1], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[2], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[3], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[4], vocab_dic, indx)
            t4 = eval(row.t4)
            post, response, hop1 = t4[1], t1[2], t1[3]
            vocab_dic, indx = updaterVocab(post, vocab_dic, indx)
            vocab_dic, indx = updaterVocab(response, vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[0], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[1], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[2], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[3], vocab_dic, indx)
            vocab_dic, indx = updaterVocab(hop1[4], vocab_dic, indx)
        # 将构建好的词表写入文件
        print("词表写入文件...")
        with open(vocab_file, 'w') as vf:
            json_str = json.dumps(vocab_dic)
            vf.write(json_str)
        return vocab_dic

    def indexed_data(self, data_dialog_file, vocab_dic):
        # step1 读取数据集
        data_df = pd.read_csv(data_dialog_file)

        # step3 根据词表重新构建df，逐行读取并索引化存入变量，最终存入文件
        def list2index(sen_li: list):
            sen_li_index = []
            for i in sen_li:
                try:
                    sen_li_index.append(vocab_dic[i])
                except:
                    sen_li_index.append(vocab_dic['<UNK>'])
            return sen_li_index

        def make_t_dialog(t):
            # 用于构造新的index的数据
            his_con, post, response, hop1 = t[0], t[1], t[2], t[3]
            if his_con == "<NULL>":
                new_his_con = [vocab_dic[his_con]]
            else:
                new_his_con = list2index(his_con)
            new_post = list2index(post)
            new_response = list2index(response)
            new_hop1 = []
            for h in hop1:
                new_hop1.append(list2index(h))
            return [new_his_con, new_post, new_response, new_hop1]

        data = []
        for row in tqdm(data_df.itertuples()):
            t_dialog = []
            t_dialog.append(make_t_dialog(eval(row.t1)))
            t_dialog.append(make_t_dialog(eval(row.t2)))
            t_dialog.append(make_t_dialog(eval(row.t3)))
            t_dialog.append(make_t_dialog(eval(row.t4)))
            data.append([t_dialog[0], t_dialog[1], t_dialog[2], t_dialog[3]])
        return data

    # def process_data_for_concept(self, entity_file, data_path,data_type,num=10):
    #
    #     with open(file=entity_file, encoding="utf-8") as f:
    #         entity_lines = f.read().splitlines()
    #
    #     def getKnowledgeDial(line_dic):
    #         if line_dic["his_con"] == '<NULL>':
    #             his_con = line_dic["his_con"]
    #         else:
    #             his_con = nltk.word_tokenize(line_dic["his_con"])
    #         if line_dic["post"] == '<SILENCE>':
    #             post = line_dic["post"]
    #         else:
    #             post = nltk.word_tokenize(line_dic["post"])
    #         response = nltk.word_tokenize(line_dic["response"])
    #         hop1_cont = line_dic["hop1"]
    #
    #         hop1 = []
    #         for v in hop1_cont:
    #             hop1.append(nltk.word_tokenize(v))
    #         return his_con, post, response, hop1
    #
    #     post_li = []
    #     his_con_li = []
    #     response_li = []
    #     for line in tqdm(entity_lines):
    #         line_dic = eval(line)
    #         his_con, post, response, hop1 = getKnowledgeDial(line_dic)
    #         post_li.append(post)
    #         his_con_li.append(his_con)
    #         response_li.append(response)
    #
    #
    #     df1 = pd.DataFrame(data={"response": response_li})
    #     df1.to_csv(data_path + '/{}_sources.csv'.format(data_type), header=False,
    #                index=False)
    #
    #     df2 = pd.DataFrame(data={"post": post_li, "his_con": his_con_li})
    #     df2.to_csv(data_path + '/{}_target.csv'.format(data_type), header=False,
    #                index=False)
    #
    #     return
