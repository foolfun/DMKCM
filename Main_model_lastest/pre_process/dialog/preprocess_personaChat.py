# -*- coding: utf-8 -*-
'''
=================================preprocess task list=================================
1\ count the avg turns which is the final number in train
2\ make the original txt : [[a0,a0,a1],[a0a1,a2,a3],[a0a1a2a3,a4,a5],[a0a1a2a3a4a5,a6,a7]]
3\ make file for tfidf
4\ add knowledge to dial
======================================================================================
'''
from Main_model_lastest.configs_manage import preProcess
# from Main_model_lastest.pre_process.knowledge.process_ex_knowledge import add_knowledge_to_dial
import numpy as np
from tqdm import tqdm
import re
import json
import os
import pandas as pd
from Main_model_lastest.pre_process.utils_for_preprocess import PreProcessingHelper
from Main_model_lastest.pre_process.knowledge.process_ex_knowledge import add_knowledge_to_dial, \
    add_knowledge_to_dial_drkit_new
from drkit.hotpotqa.demo_for_dialog_lastest import GetHops
from Main_model_lastest.pre_process.knowledge.process_ex_knowledge import FindWordsfromConceptNet

data_name = "personaChat"
p_helper = PreProcessingHelper(data_name)

class PreProcessData():
    def __init__(self, data_path, source_file):
        super().__init__()
        self.data_path = data_path
        self.path = source_file
        self.data = []
        self.turn_nums = []

    def showData(self, write_flag=False):
        # show data
        # print(self.data)
        # print(self.turn_nums)
        ma = max(self.turn_nums)  # 25
        mi = min(self.turn_nums)  # 4
        tn = np.array(self.turn_nums)
        avg = tn.mean()  # 6.8
        print(f'max turn number is :{ma};min turn number is :{mi},average turn number is :{avg}')
        if write_flag:
            # write the turn_nums down the text
            print(self.turn_nums)
            with open(self.data_path + 'turn_num.txt', 'a+') as f:
                f.write(str(self.turn_nums) + '\n')
        return

    def process_data_for_tfidf(self, out_filename, post_id, num=0):
        pre_num = 0
        cnt = 1

        tmp_persona_num = 0
        pre_sents = ''
        tmp_cnt = 0
        t_drkit = []

        with open(self.path, 'r') as f_train, open(self.data_path + "tmp_data/" + out_filename, "a+") as fo_tfidf:
            for line in tqdm(f_train.readlines()):
                if cnt == num:
                    return self.data
                else:
                    cnt += 1
                if line != "":
                    # print(line)
                    line_num, _, text = line.strip().partition(' ')
                    line_num = int(line_num)

                    if line_num == 1 and pre_num > 1:
                        # get the max number of every turn
                        self.turn_nums.append(pre_num - tmp_persona_num)
                        tmp_persona_num = 0
                        pre_sents = ''
                        tmp_cnt = 0

                    # check the line type: curr_persona or text
                    if re.search(r'^your persona:', text):
                        tmp_persona_num += 1

                    elif tmp_cnt < 4:
                        sents = p_helper.repla(text).split('\t')[:2]
                        # find [his_con, post, response]
                        if pre_sents == '':
                            his_con = '<NULL>'
                        else:
                            his_con = pre_sents

                        post = sents[0]
                        response = sents[1]
                        t_drkit.append({"_id": post_id, "his_con": his_con, "post": post, "response": response})
                        post_id += 1

                        # # 保留上一轮对话内容
                        # pre_sents = sents[0] + " " + sents[1]

                        # 保留历史的所有内容
                        pre_sents += sents[0] + " " + sents[1]

                        tmp_cnt += 1  # 用于记录目前的轮次，为了控制对话为4轮

                    # 记录上一次的编号
                    pre_num = line_num
            json.dump(t_drkit, fo_tfidf)
        return post_id


if __name__ == '__main__':
    data_name = "personaChat"
    pre_con = preProcess(data_name)
    data_path = pre_con.data_path
    source_data_path = pre_con.source_data_path

    train_file_path = source_data_path + 'train_self_revised_no_cands.txt'
    valid_file_path = source_data_path + 'valid_self_revised_no_cands.txt'
    p1 = PreProcessData(data_path=data_path, source_file=train_file_path)
    p2 = PreProcessData(data_path=data_path, source_file=valid_file_path)

    process_tfidf = False
    process_k_drkit = False
    process_k = False
    process_split = False
    process_index = False

    # 为了处理成能够喂给link_question的数据，找到知识
    if process_tfidf:
        print("知识推理数据预处理......")
        out_filename1 = "personaChat_train_for_link_add_all_his.json"
        out_filename2 = "personaChat_valid_for_link_add_all_his.json"
        post_id = 1000000  # 100万
        if os.access(data_path + "tmp_data/" + out_filename1, os.F_OK) or os.access(
                data_path + "tmp_data/" + out_filename2, os.F_OK):
            print("====================Given " + out_filename1 + " or " + out_filename2 + " is exist.===============")
        else:
            post_id1 = p1.process_data_for_tfidf(out_filename=out_filename1, post_id=post_id)
            post_id2 = p2.process_data_for_tfidf(out_filename=out_filename2, post_id=post_id1)
            print("最后一条post_id为：", post_id2 - 1)
        print("进行知识扩展......")
        # data_path, data_name, data_type, entity_dir, bert_vocab_file
        add_knowledge_to_dial(data_path=pre_con.data_path, data_name="personaChat", data_type="valid",
                              entity_dir=pre_con.entity_dir, bert_vocab_file=pre_con.bert_vocab_file)
        add_knowledge_to_dial(data_path=pre_con.data_path, data_name="personaChat", data_type="train",
                              entity_dir=pre_con.entity_dir, bert_vocab_file=pre_con.bert_vocab_file)

    if process_k_drkit:
        print("知识推理数据预处理......")
        out_filename1 = "{}_train_for_link.json".format(data_name)
        out_filename2 = "{}_valid_for_link.json".format(data_name)
        post_id = 1000000  # 100万
        if os.access(data_path + "tmp_data/"+ out_filename1, os.F_OK) \
                or os.access(data_path + "tmp_data/"+ out_filename2, os.F_OK):
            print("====================Given " + out_filename1 +
                  " or " + out_filename2 + " is exist.===============")
        else:
            post_id1 = p1.process_data_for_tfidf(out_filename=out_filename1, post_id=post_id)
            post_id2 = p2.process_data_for_tfidf(out_filename=out_filename2, post_id=post_id1)
            print("最后一条post_id为：", post_id2 - 1)
        print("进行知识扩展......")
        # data_path, data_name, data_type, entity_dir, bert_vocab_file
        # gh = GetHops(20, 5)
        gh = GetHops(100, 10, data_name)
        fc = FindWordsfromConceptNet(conceptnet_path=pre_con.knowledge_path + "conceptnet_en.txt")
        add_knowledge_to_dial_drkit_new(data_path=pre_con.data_path, data_name=data_name, data_type="valid", gh=gh,
                                        fc=fc)
        add_knowledge_to_dial_drkit_new(data_path=pre_con.data_path, data_name=data_name, data_type="train", gh=gh,
                                        fc=fc)

    # 对话数据处理（包含知识） 使用story的知识，数据处理过程
    if process_k:
        print("对话数据处理（包含知识）......")
        data1 = p_helper.process_data_with_knowledge_drkit(pre_con.entity_file_train)
        # p1.showData()
        data2 = p_helper.process_data_with_knowledge_drkit(pre_con.entity_file_valid)
        # p2.showData()
        data = data1 + data2
        df = pd.DataFrame(data=data, columns=('t1', 't2', 't3', 't4'))
        # path = data_path + 'data_dialog_knowledge_all.csv'
        path = data_path + '{}_data_all.csv'.format(data_name)
        df.to_csv(path, header=True, index=False)

    if process_split:
        print("将数据分为测试集和训练集....")
        p_helper.split_data(data_path=pre_con.data_path)

    if process_index:
        print("训练集、测试集、验证集进行索引化.....")

        # step1 读取数据集
        print("读取数据集...")
        train_path = data_path + '{}_data_train.csv'.format(data_name)
        test_path = data_path + '{}_data_test.csv'.format(data_name)
        val_path = data_path + '{}_data_val.csv'.format(data_name)

        # step2 根据测试集构建词表并保存在文件中
        print("构建词表...")
        vocab_file = pre_con.vocab_file
        if os.access(vocab_file, os.F_OK):
            with open(vocab_file, 'r') as vf:
                vocab_dic = json.load(vf)
        else:
            vocab_dic = p_helper.getVocab(train_path,vocab_file)

        # step3 根据词表重新构建df，逐行读取并索引化存入变量，最终存入文件
        print("将原始文件索引化...")
        ind_train = p_helper.indexed_data(train_path,vocab_dic)
        ind_test = p_helper.indexed_data(test_path,vocab_dic)
        ind_val = p_helper.indexed_data(val_path,vocab_dic)

        ind_train_df = pd.DataFrame(data=ind_train, columns=('t1', 't2', 't3', 't4'))
        ind_test_df = pd.DataFrame(data=ind_test, columns=('t1', 't2', 't3', 't4'))
        ind_val_df = pd.DataFrame(data=ind_val, columns=('t1', 't2', 't3', 't4'))

        ind_train_df.to_csv(data_path + '{}_ind_data_train.csv'.format(data_name), header=True, index=False)
        ind_test_df.to_csv(data_path + '{}_ind_data_test.csv'.format(data_name), header=True, index=False)
        ind_val_df.to_csv(data_path + '{}_ind_data_val.csv'.format(data_name), header=True, index=False)
