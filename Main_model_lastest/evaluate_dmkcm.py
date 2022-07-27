import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

from Main_model_lastest.predict_dmkcm import Predict
import pandas as pd
from Main_model_lastest.configs_manage import preProcess, DVKCM_config
from tqdm import tqdm
import json
import torch
import numpy as np
import random
import os

pre_con = preProcess()
base_data_path = pre_con.data_base_path


def list2str(li):
    st = ''
    for i in range(len(li)):
        if i == 0:
            st = li[i]
        else:
            st = st + ' ' + li[i]
    return st


def create_eval_data(data_names, device):
    for na in data_names:
        predict = Predict(na, itos, dmkcm_config, params_path, vocab, device)
        data_path = base_data_path + na
        data_df = pd.read_csv(data_path + '/{}_data_test.csv'.format(data_names[0]))
        labels = []
        pre_res = []
        # turn_num = 0
        # memory = None
        for line in tqdm(range(len(data_df))):
            t1 = eval(data_df.iloc[line]['t1'])
            t2 = eval(data_df.iloc[line]['t2'])
            t3 = eval(data_df.iloc[line]['t3'])
            t4 = eval(data_df.iloc[line]['t4'])

            inner_list_1 = [t1[3][0], t1[3][1], t1[3][2], t1[3][3], t1[3][4]]
            inner_list_2 = [t2[3][0], t2[3][1], t2[3][2], t2[3][3], t2[3][4]]
            inner_list_3 = [t3[3][0], t3[3][1], t3[3][2], t3[3][3], t3[3][4]]
            inner_list_4 = [t4[3][0], t4[3][1], t4[3][2], t4[3][3], t4[3][4]]

            lab1 = list2str(t1[2])
            lab2 = list2str(t2[2])
            lab3 = list2str(t3[2])
            lab4 = list2str(t4[2])
            labels.append(lab1)
            labels.append(lab2)
            labels.append(lab3)
            labels.append(lab4)

            response1, memory1, memory_mask1 = predict.get_res(his_con=t1[0], post=t1[1], inner=inner_list_1,
                                                               turn_num=0, memory=None)
            # print('label: ', lab1)
            response2, memory2, memory_mask2 = predict.get_res(his_con=t2[0], post=t2[1], inner=inner_list_2,
                                                               turn_num=1, memory=memory1, memory_mask=memory_mask1)
            # print('label: ', lab2)
            response3, memory3, memory_mask3 = predict.get_res(his_con=t3[0], post=t3[1], inner=inner_list_3,
                                                               turn_num=2, memory=memory2, memory_mask=memory_mask2)
            # print('label: ', lab3)
            response4, memory4, memory_mask4 = predict.get_res(his_con=t4[0], post=t4[1], inner=inner_list_4,
                                                               turn_num=3, memory=memory3, memory_mask=memory_mask3)
            # print('label: ', lab4)

            print('\n')
            print('response: ', response1)
            print('label: ', lab1)
            print('\n')

            print('response: ', response2)
            print('label: ', lab2)
            print('\n')

            print('response: ', response3)
            print('label: ', lab3)
            print('\n')

            print('response: ', response4)
            print('label: ', lab4)
            print('\n')

            pre_res.append(response1)
            pre_res.append(response2)
            pre_res.append(response3)
            pre_res.append(response4)


        lab = pd.DataFrame(data=labels)
        pre = pd.DataFrame(data=pre_res)
        lab.to_csv(data_path + '/evl_lab_data_dmkcm_epoch_' + name + '.csv', header=False, index=False)
        pre.to_csv(data_path + '/evl_pre_data_dmkcm_epoch_' + name + '.csv', header=False, index=False)
        return


def create_eval_hop2_data(data_names, device):
    kg_path = '/'.join(dmkcm_config.test_data_path.split('/')[:-1]) + '/kg_test.csv'
    if os.path.exists(kg_path):
        kgframe = pd.read_csv(kg_path)
        print('read kg from {}'.format(kg_path))
    else:
        kg = []
        with open('/'.join(dmkcm_config.test_data_path.split('/')[:-1]) + '/test_1hops_100_directed_triple_filter.json', 'r') as f:
            for line in f.readlines():
                line_data = json.loads(line)
                kg.append(line_data)
        kgframe = pd.DataFrame(columns=('t11', 't22', 't33', 't44'), index=None)
        for idx, line in tqdm(enumerate(kg), total=len(kg) // 4):
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
    for na in data_names:
        predict = Predict(na, itos, dmkcm_config, params_path, vocab, device)
        data_path = base_data_path + na
        # data_df = pd.read_csv(data_path + '/0605/data_dialog_knowledge_story_test0605.csv')
        data_df = pd.read_csv(data_path + '/{}_data_test.csv'.format(data_names[0]))
        labels = []
        pre_res = []
        # turn_num = 0
        # memory = None
        for line in tqdm(range(len(data_df))):
            if line < 620:
                continue

            t1 = eval(data_df.iloc[line]['t1'])
            t2 = eval(data_df.iloc[line]['t2'])
            t3 = eval(data_df.iloc[line]['t3'])
            t4 = eval(data_df.iloc[line]['t4'])
            t11 = eval(kgframe.iloc[line]['t11'])
            t22 = eval(kgframe.iloc[line]['t22'])
            t33 = eval(kgframe.iloc[line]['t33'])
            t44 = eval(kgframe.iloc[line]['t44'])

            inner_list_1 = [t1[3][0], t1[3][1], t1[3][2], t1[3][3], t1[3][4]]
            inner_list_2 = [t2[3][0], t2[3][1], t2[3][2], t2[3][3], t2[3][4]]
            inner_list_3 = [t3[3][0], t3[3][1], t3[3][2], t3[3][3], t3[3][4]]
            inner_list_4 = [t4[3][0], t4[3][1], t4[3][2], t4[3][3], t4[3][4]]

            lab1 = list2str(t1[2])
            lab2 = list2str(t2[2])
            lab3 = list2str(t3[2])
            lab4 = list2str(t4[2])
            labels.append(lab1)
            labels.append(lab2)
            labels.append(lab3)
            labels.append(lab4)

            response1, memory1, memory_mask1 = predict.get_res(his_con=t1[0], post=t1[1], inner=inner_list_1, turn_num=0, memory=None, concepts=t11[0], labels=t11[1], distances=t11[2], relations=t11[3], head_ids=t11[4], tail_ids=t11[5], triple_labels=t11[6], vocab_map=None, vocab_mask=None)
            print('label: ', lab1)
            response2, memory2, memory_mask2 = predict.get_res(his_con=t2[0], post=t2[1], inner=inner_list_2, turn_num=1, memory=memory1, memory_mask=memory_mask1, concepts=t22[0], labels=t22[1], distances=t22[2], relations=t22[3], head_ids=t22[4], tail_ids=t22[5], triple_labels=t22[6], vocab_map=None, vocab_mask=None)
            print('label: ', lab2)
            response3, memory3, memory_mask3 = predict.get_res(his_con=t3[0], post=t3[1], inner=inner_list_3, turn_num=2, memory=memory2, memory_mask=memory_mask2, concepts=t33[0], labels=t33[1], distances=t33[2], relations=t33[3], head_ids=t33[4], tail_ids=t33[5], triple_labels=t33[6], vocab_map=None, vocab_mask=None)
            print('label: ', lab3)
            response4, memory4, memory_mask4 = predict.get_res(his_con=t4[0], post=t4[1], inner=inner_list_4, turn_num=3, memory=memory3, memory_mask=memory_mask3, concepts=t44[0], labels=t44[1], distances=t44[2], relations=t44[3], head_ids=t44[4], tail_ids=t44[5], triple_labels=t44[6], vocab_map=None, vocab_mask=None)
            print('label: ', lab4)

            # print('\n')
            # print('response: ', response1)
            # print('label: ', lab1)
            # print('\n')
            #
            # print('response: ', response2)
            # print('label: ', lab2)
            # print('\n')
            #
            # print('response: ', response3)
            # print('label: ', lab3)
            # print('\n')
            #
            # print('response: ', response4)
            # print('label: ', lab4)
            # print('\n')

            pre_res.append(response1)
            pre_res.append(response2)
            pre_res.append(response3)
            pre_res.append(response4)

            # '''
            with open(
                    data_path + '/evl_data_dmkcm_' + name + '/generate_dmkcm_epoch' + str(cur_num) + '_' + name + '.txt',
                    "a+") as f:
                f.write("\n1HOP-1: " + list2str(t1[3][0]))
                f.write("\n1HOP-2: " + list2str(t1[3][1]))
                f.write("\n1HOP-3: " + list2str(t1[3][2]))
                f.write("\n1HOP-4: " + list2str(t1[3][3]))
                f.write("\n1HOP-5: " + list2str(t1[3][4]))
                f.write("\n2HOP: " + list2str(t11[0]))
                f.write("\nhis_con: " + list2str(t1[0]))
                f.write("\npost: " + list2str(t1[1]))
                f.write("\nlab: " + lab1)
                f.write("\ngenerated: " + response1 + "\n")

                f.write("\n1HOP-1: " + list2str(t2[3][0]))
                f.write("\n1HOP-2: " + list2str(t2[3][1]))
                f.write("\n1HOP-3: " + list2str(t2[3][2]))
                f.write("\n1HOP-4: " + list2str(t2[3][3]))
                f.write("\n1HOP-5: " + list2str(t2[3][4]))
                f.write("\n2HOP: " + list2str(t22[0]))
                f.write("\nhis_con: " + list2str(t1[1] + t1[2]))
                f.write("\npost: " + list2str(t2[1]))
                f.write("\nlab: " + lab2)
                f.write("\ngenerated: " + response2 + "\n")

                f.write("\n1HOP-1: " + list2str(t3[3][0]))
                f.write("\n1HOP-2: " + list2str(t3[3][1]))
                f.write("\n1HOP-3: " + list2str(t3[3][2]))
                f.write("\n1HOP-4: " + list2str(t3[3][3]))
                f.write("\n1HOP-5: " + list2str(t3[3][4]))
                f.write("\n2HOP: " + list2str(t33[0]))
                f.write("\nhis_con: " + list2str(t2[1] + t2[2]))
                f.write("\npost: " + list2str(t3[1]))
                f.write("\nlab: " + lab3)
                f.write("\ngenerated: " + response3 + "\n")

                f.write("\n1HOP-1: " + list2str(t4[3][0]))
                f.write("\n1HOP-2: " + list2str(t4[3][1]))
                f.write("\n1HOP-3: " + list2str(t4[3][2]))
                f.write("\n1HOP-4: " + list2str(t4[3][3]))
                f.write("\n1HOP-5: " + list2str(t4[3][4]))
                f.write("\n2HOP: " + list2str(t44[0]))
                f.write("\nhis_con: " + list2str(t3[1] + t3[2]))
                f.write("\npost: " + list2str(t4[1]))
                f.write("\nlab: " + lab4)
                f.write("\ngenerated:" + response4 + "\n")
        return


if __name__ == '__main__':
    dmkcm_config = DVKCM_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    print('正在使用显卡设备: ', torch.cuda.current_device())

    name = '*'

    # ================================
    dmkcm_config.dataChoice = 'dailyDialog'  # personaChat / dailyDialog
    dmkcm_config.only_hop1 = False
    dmkcm_config.memory = False
    dmkcm_config.hop2 = False
    dmkcm_config.paral = False
    dmkcm_config.his_mode = 'last_one_his'
    dmkcm_config.encoding_mode = 'inp_cat_2'

    vocab_name = '*'
    print('使用%s词表' % vocab_name)

    cur_num = 60
    params_path = dmkcm_config.checkPoint_path + dmkcm_config.dataChoice + '/DMKCM_epoch_model_' + name + '/dmkcm_' + name + '_epoch' + str(cur_num)

    word_dict_path = dmkcm_config.base_path + 'data/' + dmkcm_config.dataChoice + '/vocab_data/vocab_' + vocab_name + '.json'  # <sos>, <eos>
    vocab = json.load(open(word_dict_path))
    itos = {value: key for key, value in vocab.items()}

    # names = ['personaChat', 'dailyDialog']
    names = [dmkcm_config.dataChoice]
    if dmkcm_config.hop2:
        create_eval_hop2_data(names, device)
    else:
        create_eval_data(names, device)
