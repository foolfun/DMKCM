# -*- coding: utf-8 -*-
import time
import torch


class preProcess:
    def __init__(self, dataChoice='personaChat'):
        self.base = "*/Chat" # 项目所在路径
        self.base_path = self.base + "Main_model_lastest/"
        self.data_base_path = self.base_path + 'data/'
        self.drkit_path = self.base + "drkit/"

        # hollE dailyDialog personaChat,WOW
        self.dataChoice = dataChoice
        self.data_path = self.data_base_path + dataChoice + '/'
        self.vocab_file = self.data_path + "vocab_data/" + "{}_vocab.json".format(dataChoice)  # 这个是自己构建的词表
        self.source_data_path = self.data_path + 'source_data/'
        self.entity_file = self.data_base_path + dataChoice + '/' + dataChoice + '_entities.json'

        # 保存抽取的story的文件
        self.entity_file_train = self.data_base_path + dataChoice + '/tmp_data/' + dataChoice + '_entities_drkit_train.json'
        self.entity_file_valid = self.data_base_path + dataChoice + '/tmp_data/' + dataChoice + '_entities_drkit_valid.json'
        self.entity_file_test = self.data_base_path + dataChoice + '/tmp_data/' + dataChoice + '_entities_drkit_test.json'

        # 与story处理有关的路径
        self.knowledge_path = self.data_base_path + "knowledge/"
        self.source_story_data = self.knowledge_path + "ROCStories_winter2017.csv"
        self.story_idf_file = self.knowledge_path + "story_idf_dict.txt"
        self.keywords_file = self.knowledge_path + "candi_keyword.txt"

        # drkit处理知识的相关路径
        self.knowledge_path_drkit = self.drkit_path + "data/"
        self.bert_vocab_file = self.drkit_path + "wwm_uncased_L-24_H-1024_A-16/vocab.txt"
        self.entity_dir = self.knowledge_path_drkit + "story_preprocessed_corpus"
        self.story_for_index = self.knowledge_path_drkit + "story_for_index.json"
        self.story_for_parse = self.knowledge_path_drkit + "story_for_parse.json"

class DVKCM_config:
    def __init__(self):
        torch.cuda.set_device(0)
        dataChoice = 'dailyDialog'
        name = '*'
        self.debug = False
        self.save_ckp = True
        self.load_ckp = False
        self.print_train = False
        self.write_vocab_dict = True
        self.BATCH_SIZE = 4
        self.N_EPOCHS = 100
        self.only_hop1 = False
        self.memory = True
        self.hop2 = False
        self.pad_mode = None
        self.his_mode = 'last_one_his'
        self.encoding_mode = 'inp_cat_2'
        vocab_name = '0724'
        ####
        #  基本参数，不做调整
        self.name = name
        self.paral = False
        self.gpu = [0, 1]
        self.pad_post_len = 50
        self.pad_his_len = 150

        # base path
        self.base_path = '*/Chat/Main_model_lastest/'
        self.checkPoint_path = self.base_path + 'checkpoint/'
        self.dataChoice = dataChoice  # 选择数据集

        # checkpoint
        self.params_path = self.checkPoint_path + self.dataChoice + '/DMKCM_params_' + name + '.pt'
        # self.model_path = self.checkPoint_path + self.dataChoice + '/DMKCM_epoch_model_' + name + '.pt'
        self.model_path = self.checkPoint_path + self.dataChoice + '/DMKCM_epoch_model_' + name
        self.log_path = self.base_path + 'log/' + self.dataChoice + '/DMKCM_params_' + name + '.txt'

        # vocab
        self.word_dict_path = self.base_path + 'data/' + self.dataChoice + '/vocab_data/vocab_' + vocab_name + '.json'

        if dataChoice == 'personaChat':
            self.train_data_path = self.base_path + 'data/' + self.dataChoice + '/personaChat_data_train.csv'
            self.val_data_path = self.base_path + 'data/' + self.dataChoice + '/personaChat_data_val.csv'
            self.test_data_path = self.base_path + 'data/' + self.dataChoice + '/personaChat_data_test.csv'
        elif dataChoice == 'dailyDialog':
            self.train_data_path = self.base_path + 'data/' + self.dataChoice + '/dailyDialog_data_train.csv'
            self.val_data_path = self.base_path + 'data/' + self.dataChoice + '/dailyDialog_data_val.csv'
            self.test_data_path = self.base_path + 'data/' + self.dataChoice + '/dailyDialog_data_test.csv'

        self.learning_rate = 0.0001  # learning rate
        self.CLIP = 1

        self.ENC_EMB_DIM = 300
        self.DEC_EMB_DIM = 300
        self.ENC_HID_DIM = 512
        self.DEC_HID_DIM = 512
        self.ATTN_DIM = 64
        self.ENC_DROPOUT = 0.5
        self.DEC_DROPOUT = 0.5