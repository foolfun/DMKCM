# -*- coding: utf-8 -*-
'''
=================================preprocess task list=================================
1\ count the avg turns which is the final number in train
2\ make the original txt : [[a0,a0,a1],[a0a1,a2,a3],[a0a1a2a3,a4,a5],[a0a1a2a3a4a5,a6,a7]]
3\ make file for tfidf
4\ add knowledge to dial
======================================================================================
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
from Main_model_lastest.configs_manage import preProcess
from Main_model_lastest.pre_process.utils_for_preprocess import PreProcessingHelper
import json
import os
from Main_model_lastest.pre_process.knowledge.process_ex_knowledge import add_knowledge_to_dial, \
    add_knowledge_to_dial_drkit_new
from drkit.hotpotqa.demo_for_dialog_lastest import GetHops
from Main_model_lastest.pre_process.knowledge.process_ex_knowledge import FindWordsfromConceptNet

data_name = 'dailyDialog'
p_helper = PreProcessingHelper(data_name=data_name)


class PreProcessData():
    def __init__(self, data_path, source_file):
        super().__init__()
        self.data_path = data_path
        self.source_file = source_file
        self.data = []
        self.turn_nums = []
        self.mark = [',', '.', '!', '?', ':']

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
            # print(self.turn_nums)

            with open(self.data_path + 'turn_num.txt', 'w+') as f:
                f.write(str(self.turn_nums) + '\n')
        return

    def process_data_for_tfidf(self, out_filename, post_id, num=0):
        '''
        :param num: control the number of processing data , default is no control,value is 0
        :return: the data list which is processed
        '''

        cnt = 1
        t_drkit = []
        with open(self.source_file, 'r') as f_train, open(self.data_path+ "tmp_data/" + out_filename, "w+") as fo_tfidf:

            for line in tqdm(f_train.readlines()):
                if cnt == num:
                    return self.data
                else:
                    cnt += 1
                if line != "":
                    # replace the special mark
                    text = p_helper.repla(line)

                    ext = text.strip().split('__eou__')[:-1]
                    self.turn_nums.append(len(ext) / 2)
                    turns = len(ext) / 2
                    if turns < 4:
                        continue
                    else:
                        for i in range(0, 7, 2):
                            if i == 0:

                                t_drkit.append({"_id": post_id,
                                                "his_con": "<NULL>",
                                                "post": ext[i],
                                                "response": ext[i + 1]})
                                post_id += 1
                            else:
                                his_con = ""
                                for li in ext[:i]:
                                    his_con += li
                                t_drkit.append({"_id": post_id,
                                                "his_con": his_con,
                                                "post": ext[i],
                                                "response": ext[i + 1]})
                                post_id += 1

            json.dump(t_drkit, fo_tfidf)

        return post_id


if __name__ == '__main__':
    pre_con = preProcess(data_name)
    data_path = pre_con.data_path
    source_data_path = pre_con.source_data_path

    train_file = source_data_path + 'dialogues_train.txt'
    test_file = source_data_path + 'dialogues_test.txt'
    valid_file = source_data_path + 'dialogues_validation.txt'

    p1 = PreProcessData(data_path=data_path, source_file=train_file)
    p2 = PreProcessData(data_path=data_path, source_file=valid_file)
    p3 = PreProcessData(data_path=data_path, source_file=test_file)

    process_tfidf = False  # 用tfidf寻找知识
    process_k_drkit = False  # 用drkit寻找知识 两种方法进行对比后发现用drkit更好
    process_k = False  # 得到有知识的数据集
    process_split = False  # 分割数据集
    process_index = False  # 提前索引化数据集

    if process_tfidf:
        print("TFIDF数据预处理......")
        out_filename1 = "dailyDialog_train_for_link.json"
        out_filename2 = "dailyDialog_valid_for_link.json"
        out_filename3 = "dailyDialog_test_for_link.json"
        post_id = 1000000  # 100万
        if os.access(data_path+ "tmp_data/" + out_filename1, os.F_OK) \
                or os.access(data_path+ "tmp_data/" + out_filename2, os.F_OK) \
                or os.access(data_path + "tmp_data/"+ out_filename3, os.F_OK):
            print("====================Given " + out_filename1 +
                  " or " + out_filename2 + " or " + out_filename3 + " is exist.===============")
        else:
            post_id1 = p1.process_data_for_tfidf(out_filename=out_filename1, post_id=post_id)
            post_id2 = p2.process_data_for_tfidf(out_filename=out_filename2, post_id=post_id1)
            post_id3 = p3.process_data_for_tfidf(out_filename=out_filename3, post_id=post_id2)
            print("最后一条post_id为：", post_id3 - 1)
        print("进行知识扩展......")
        # data_path, data_name, data_type, entity_dir, bert_vocab_file
        data_name = "dailyDialog"
        add_knowledge_to_dial(data_path=pre_con.data_path, data_name=data_name, data_type="valid",
                              entity_dir=pre_con.entity_dir, bert_vocab_file=pre_con.bert_vocab_file)
        add_knowledge_to_dial(data_path=pre_con.data_path, data_name=data_name, data_type="train",
                              entity_dir=pre_con.entity_dir, bert_vocab_file=pre_con.bert_vocab_file)
        add_knowledge_to_dial(data_path=pre_con.data_path, data_name=data_name, data_type="test",
                              entity_dir=pre_con.entity_dir, bert_vocab_file=pre_con.bert_vocab_file)

    if process_k_drkit:
        print("知识推理数据预处理......")
        out_filename1 = "dailyDialog_train_for_link.json"
        out_filename2 = "dailyDialog_valid_for_link.json"
        out_filename3 = "dailyDialog_test_for_link.json"
        post_id = 1000000  # 100万
        if os.access(data_path + out_filename1, os.F_OK) \
                or os.access(data_path+ "tmp_data/" + out_filename2, os.F_OK) \
                or os.access(data_path + "tmp_data/"+ out_filename3, os.F_OK):
            print("====================Given " + out_filename1 +
                  " or " + out_filename2 + " or " + out_filename3 + " is exist.===============")
        else:
            post_id1 = p1.process_data_for_tfidf(out_filename=out_filename1, post_id=post_id)
            post_id2 = p2.process_data_for_tfidf(out_filename=out_filename2, post_id=post_id1)
            post_id3 = p3.process_data_for_tfidf(out_filename=out_filename3, post_id=post_id2)
            print("最后一条post_id为：", post_id3 - 1)
        print("进行知识扩展......")
        # data_path, data_name, data_type, entity_dir, bert_vocab_file
        data_name = "dailyDialog"
        gh = GetHops(100, 10, data_name)
        fc = FindWordsfromConceptNet(conceptnet_path=pre_con.knowledge_path + "conceptnet_en.txt")
        add_knowledge_to_dial_drkit_new(data_path=pre_con.data_path, data_name=data_name, data_type="valid",
                                        gh=gh, fc=fc)
        add_knowledge_to_dial_drkit_new(data_path=pre_con.data_path, data_name=data_name, data_type="train",
                                        gh=gh, fc=fc)
        add_knowledge_to_dial_drkit_new(data_path=pre_con.data_path, data_name=data_name, data_type="test",
                                        gh=gh, fc=fc)

    if process_k:
        # 对话数据处理（包含知识） 使用story的知识，数据处理过程
        print("对话数据处理（包含知识）......")
        data1 = p_helper.process_data_with_knowledge_drkit(pre_con.entity_file_train)
        data2 = p_helper.process_data_with_knowledge_drkit(pre_con.entity_file_valid)
        data3 = p_helper.process_data_with_knowledge_drkit(pre_con.entity_file_test)
        data = data1 + data2 + data3

        df = pd.DataFrame(data=data, columns=('t1', 't2', 't3', 't4'))
        # path = data_path + 'data_dialog_knowledge_all.csv'
        path = data_path + '{}_data_all.csv'.format(data_name)
        df.to_csv(path, header=True, index=False)

    if process_split:
        print("将数据分为测试集和训练集....")
        p_helper.split_data(pre_con.data_path)

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


