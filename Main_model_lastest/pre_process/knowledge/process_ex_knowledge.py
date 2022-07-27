# coding=utf-8
'''
用于处理知识的文件
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import pandas as pd
import urllib.parse
import json
import os
from bert import tokenization
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from drkit import search_utils
from Main_model_lastest.pre_process.knowledge.data_utils import *
from Main_model_lastest.pre_process.knowledge.data_utils import KeywordExtractor, getKeywords
from Main_model_lastest.pre_process import tokenization_new
from Main_model_lastest.configs_manage import preProcess
from Main_model_lastest.pre_process.utils_for_preprocess import PreProcessingHelper


pp = PreProcessingHelper()
pre_con = preProcess()

# step1 计算story文件的idf
class dts_Target():
    def __init__(self, story_file):
        self.story_file = story_file

    def get_story(self):
        return pd.read_csv(self.story_file)

    def cal_idf(self):
        counter = collections.Counter()
        story_df = self.get_story()
        total = 0.
        titles = story_df['storytitle'].tolist()
        s1 = story_df['sentence1'].tolist()
        s2 = story_df['sentence2'].tolist()
        s3 = story_df['sentence3'].tolist()
        s4 = story_df['sentence4'].tolist()
        s5 = story_df['sentence5'].tolist()
        for i in tqdm(range(len(titles))):
            total += 6
            counter.update(set(kw_tokenize(titles[i])))
            counter.update(set(kw_tokenize(s1[i])))
            counter.update(set(kw_tokenize(s2[i])))
            counter.update(set(kw_tokenize(s3[i])))
            counter.update(set(kw_tokenize(s4[i])))
            counter.update(set(kw_tokenize(s5[i])))

        idf_dict = {}
        for k, v in counter.items():
            idf_dict[k] = np.log10(total / (v + 1.))
        return idf_dict

    def make_dataset(self, out_file):
        idf_dict = self.cal_idf()
        with open(out_file, "w") as f:
            f.write(str(idf_dict))


# step2 对外部知识（story）预处理
class ParseStories():
    def __init__(self, story_path, story_file, idf_file, out_file):
        self.out_file = out_file
        self.story_df = pd.read_csv(story_file)
        self.idf_file = idf_file
        print("正在预处理conceptnet")
        self.fc = FindWordsfromConceptNet(conceptnet_path=story_path + "conceptnet_en.txt")
        self.stop_words = set(stopwords.words('english'))

    def preProcess(self):
        titles = self.story_df['storytitle'].tolist()
        ids = self.story_df['storyid'].tolist()
        s1 = self.story_df['sentence1'].tolist()
        s2 = self.story_df['sentence2'].tolist()
        s3 = self.story_df['sentence3'].tolist()
        s4 = self.story_df['sentence4'].tolist()
        s5 = self.story_df['sentence5'].tolist()
        story = []
        for j in range(len(ids)):
            story.append(s1[j] + s2[j] + s3[j] + s4[j] + s5[j])

        # 这个是事先做好idf统计的文件(用于寻找关键词)
        print("读取idf文件....")
        with open(self.idf_file) as f:
            lines = f.readlines()
        dic = eval(lines[0])
        ke = KeywordExtractor(dic)

        print("预处理story....")
        with open(self.out_file, "w") as fs:
            # 这段是story的
            doc = []
            # for i in tqdm(range(10000)):
            for i in tqdm(range(len(titles))):
                # 获取关键词 如果是title就直接切分，如果不是title就寻找关键词
                t = getKeywords(sent=titles[i].lower(), ke=ke, is_title=True)
                # mark = [".", "?", "!", ":", ","]

                ls1 = getKeywords(sent=s1[i].lower(), ke=ke, is_title=False)
                ls2 = getKeywords(sent=s2[i].lower(), ke=ke, is_title=False)
                ls3 = getKeywords(sent=s3[i].lower(), ke=ke, is_title=False)
                ls4 = getKeywords(sent=s4[i].lower(), ke=ke, is_title=False)
                ls5 = getKeywords(sent=s5[i].lower(), ke=ke, is_title=False)
                # 组合
                cont = ls1 + ls2 + ls3 + ls4 + ls5
                key_list = list(set(t + cont))

                # 将list变成str
                str_key = ""
                for j in range(len(key_list)):
                    if j == " ":
                        continue
                    if j < len(key_list) - 1:
                        str_key += key_list[j] + " "
                    else:
                        str_key += key_list[j]

                # 扩展story关键词
                t_for_hop2 = getKeywords(sent=titles[i].lower(), ke=ke, is_title=False)  # 如果是title就直接切分
                ls1_for_hop2 = getKeywords(sent=s1[i].lower(), ke=ke, is_title=False)  # 如果不是title就寻找关键词
                ls2_for_hop2 = getKeywords(sent=s2[i].lower(), ke=ke, is_title=False)
                ls3_for_hop2 = getKeywords(sent=s3[i].lower(), ke=ke, is_title=False)
                ls4_for_hop2 = getKeywords(sent=s4[i].lower(), ke=ke, is_title=False)
                ls5_for_hop2 = getKeywords(sent=s5[i].lower(), ke=ke, is_title=False)
                keywords2expand = list(
                    set(t_for_hop2 + ls1_for_hop2 + ls2_for_hop2 + ls3_for_hop2 + ls4_for_hop2 + ls5_for_hop2))
                expand_key_list = self.fc.expandEntities(keywords2expand)
                mark = [".", "?", "!", ":", ","]
                new_expand_key_list = []

                for tmp_k in expand_key_list:
                    if tmp_k in mark or tmp_k in self.stop_words:
                        continue
                    else:
                        new_expand_key_list.append(tmp_k)

                doc.append({"id": ids[i],
                            "kb_id": urllib.parse.quote(str_key),
                            "title": str_key,
                            "real_title": titles[i].lower(),
                            "s1": s1[i].lower(),
                            "s2": s2[i].lower(),
                            "s3": s3[i].lower(),
                            "s4": s4[i].lower(),
                            "s5": s5[i].lower(),
                            "story": story[i].lower(),
                            "hop2": new_expand_key_list,
                            })

            fs.write("\n".join(json.dumps(d) for d in doc))


# Step3 对story构造索引
def makeIndex(vocab_file, story_file):
    # global entity2id, entity2title
    # Initialize tokenizer.
    tokenizer = tokenization_new.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    # Read paragraphs, mentions and entities.
    tf.logging.info("Reading all entities")
    entity2id, entity2title, entity2real_title, entity2s1, entity2s2, entity2s3, entity2s4, entity2s5 = {}, {}, {}, {}, {}, {}, {}, {}
    entity2hop2 = {}
    # with tf.gfile.Open(story_file) as sf, tf.gfile.Open(wiki_file) as wf:
    with tf.gfile.Open(story_file) as sf:
        for ii, line in tqdm(enumerate(sf)):
            orig_para = json.loads(line.strip())
            ee = orig_para["kb_id"].lower()
            if ee not in entity2id:
                entity2id[ee] = len(entity2id)
                entity2title[ee] = orig_para["title"]
                entity2real_title[ee] = orig_para["real_title"]
                entity2s1[ee] = orig_para["s1"]
                entity2s2[ee] = orig_para["s2"]
                entity2s3[ee] = orig_para["s3"]
                entity2s4[ee] = orig_para["s4"]
                entity2s5[ee] = orig_para["s5"]
                entity2hop2[ee] = orig_para["hop2"]
                # entity2cont[ee] = orig_para["story"]

    tf.logging.info("Found %d entities", len(entity2id))

    tf.logging.info("Reading paragraphs from %s", story_file)

    tf.logging.info("Saving entities metadata.")
    # entity2id, entity2title, entity2s1, entity2s2, entity2s3, entity2s4, entity2s5,entity2story
    json.dump([entity2id, entity2title, entity2real_title, entity2s1, entity2s2, entity2s3, entity2s4, entity2s5,
               entity2hop2],
              tf.gfile.Open(os.path.join(multihop_output_dir, "entities.json"), "w"))

    # Store entity tokens.
    tf.logging.info("Processing entities.")
    entity_ids = np.zeros((len(entity2id), max_entity_length),
                          dtype=np.int32)
    entity_mask = np.zeros((len(entity2id), max_entity_length),
                           dtype=np.float32)

    num_exceed_len = 0.
    for entity in tqdm(entity2id):
        ei = entity2id[entity]
        entity_tokens = tokenizer.tokenize(entity2title[entity])
        entity_token_ids = tokenizer.convert_tokens_to_ids(entity_tokens)

        if len(entity_token_ids) > max_entity_length:
            num_exceed_len += 1
            entity_token_ids = entity_token_ids[:max_entity_length]

        entity_ids[ei, :len(entity_token_ids)] = entity_token_ids
        entity_mask[ei, :len(entity_token_ids)] = 1.

    tf.logging.info("Saving %d entity ids. %d exceed max-length of %d.",
                    len(entity2id), num_exceed_len, max_entity_length)
    search_utils.write_to_checkpoint(
        "entity_ids", entity_ids, tf.int32,
        os.path.join(multihop_output_dir, "entity_ids"))
    search_utils.write_to_checkpoint(
        "entity_mask", entity_mask, tf.float32,
        os.path.join(multihop_output_dir, "entity_mask"))


# step4 知识与对话做tfidf，找到对话对应的知识
# 读取concept
class FindWordsfromConceptNet():
    def __init__(self, conceptnet_path):
        self.conceptnet_path = conceptnet_path
        self.new_concept_dic = self.get_concept_dic()
        self.stop_words = set(stopwords.words('english'))

    # 读取conceptnet的数据
    def readConceptData(self, conceptnet_path):
        with open(conceptnet_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        relation, e1, e2, scores = [], [], [], []
        for li in lines:
            ree = li.split('\t')
            relation.append(ree[0])
            e1.append(ree[1])
            e2.append(ree[2])
            scores.append(ree[3].strip())
        return e1, e2, scores

    # 获取列表的第二个元素
    def takeSecond(self, elem):
        return elem[1]

    def get_concept_dic(self):
        new_concept_dic = {}
        path = self.conceptnet_path + "my_concept.csv"
        if os.access(path, os.F_OK):
            df = pd.read_csv(path)
            tmp = zip(df['e1'], df['e2'])
            for i, l in enumerate(tqdm(tmp, total=df.shape[0])):
                new_concept_dic[l[0]] = l[1]
        else:
            e1, e2, scores = self.readConceptData(self.conceptnet_path)
            concept_dic = {}
            for i in range(len(e1)):
                # 过滤掉小于1的
                # if float(scores[i]) < 1.0:
                #     continue

                # if "_" in e2[i]:
                #     continue
                e2_val = e2[i].replace("_", " ")
                e1_val = e1[i].replace("_", " ")
                if e1[i] in concept_dic.keys():
                    concept_dic[e1_val].append([e2_val, float(scores[i])])
                else:
                    concept_dic[e1_val] = []
                    concept_dic[e1_val].append([e2_val, float(scores[i])])

            for k in concept_dic.keys():
                concept_dic[k].sort(key=self.takeSecond)
                num = len(concept_dic[k])
                for j in range(num):
                    if k in new_concept_dic.keys():
                        new_concept_dic[k].append(concept_dic[k][j][0])
                    else:
                        new_concept_dic[k] = []
                        new_concept_dic[k].append(concept_dic[k][j][0])
            data = {"e1": new_concept_dic.keys(), "e2": new_concept_dic.values()}
            df = pd.DataFrame(data=data)
            df.to_csv(path)
        return new_concept_dic

    # 用conceptNet扩展句子中的关键词
    def expandEntities(self, sent_keywords: list):
        sent_entities = sent_keywords[:]
        if isinstance(sent_keywords, str):
            sent_keywords = sent_keywords.split(' ')
        ents = []
        mark = [".", "?", "!", ":", ","]
        for e in sent_keywords:
            if e in self.stop_words or e in mark:
                continue
            try:
                con_dic = self.new_concept_dic[e]
                if isinstance(con_dic, str):
                    con_dic = eval(con_dic)
                ents += con_dic
            except:
                continue
        sent_entities += ents
        return list(set(sent_entities))

def tfidf_linking(dialogs, base_dir, tokenizer, top_k, all_top_k, batch_size=100):
    """Match dialogs to entities via Tf-IDF."""
    # Load entity ids and masks.
    tf.reset_default_graph()
    id_ckpt = os.path.join(base_dir, "entity_ids")
    entity_ids = search_utils.load_database(
        "entity_ids", None, id_ckpt, dtype=tf.int32)
    mask_ckpt = os.path.join(base_dir, "entity_mask")
    entity_mask = search_utils.load_database("entity_mask", None, mask_ckpt)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        tf.logging.info("Loading entity ids and masks...")
        np_ent_ids, np_ent_mask = sess.run([entity_ids, entity_mask])
    tf.logging.info("Building entity count matrix...")
    entity_count_matrix = search_utils.build_count_matrix(np_ent_ids, np_ent_mask)

    # Tokenize dialogs and build count matrix.
    tf.logging.info("Tokenizing post and response...")
    pos_toks, pos_masks = [], []

    for dialog in tqdm(dialogs):
        # 得到 post 的 toks 和 masks
        toks = tokenizer.tokenize(dialog['post'])

        tok_ids = tokenizer.convert_tokens_to_ids(toks)
        pos_toks.append(tok_ids)
        pos_masks.append([1 for _ in tok_ids])

    tf.logging.info("Building post count matrix...")
    dialog_count_matrix = search_utils.build_count_matrix(pos_toks, pos_masks)
    tf.logging.info("Building response count matrix...")

    # Tf-IDF.
    tf.logging.info("Computing IDFs...")
    idfs = search_utils.counts_to_idfs(entity_count_matrix, cutoff=1e-5)

    tf.logging.info("Computing entity Tf-IDFs...")
    ent_tfidfs = search_utils.counts_to_tfidf(entity_count_matrix, idfs)
    ent_tfidfs = normalize(ent_tfidfs, norm="l2", axis=0)

    tf.logging.info("Computing post TF-IDFs...")
    pos_tfidfs = search_utils.counts_to_tfidf(dialog_count_matrix, idfs)
    pos_tfidfs = normalize(pos_tfidfs, norm="l2", axis=0)

    tf.logging.info("Searching...")
    top_doc_indices = np.empty((len(dialogs), top_k), dtype=np.int32)
    top_doc_distances = np.empty((len(dialogs), top_k), dtype=np.float32)

    num_batches = len(dialogs) // batch_size
    tf.logging.info("Computing distances in %d batches of size %d",
                    num_batches + 1, batch_size)

    for nb in tqdm(range(num_batches + 1)):
        min_ = nb * batch_size
        max_ = (nb + 1) * batch_size
        if min_ >= len(dialogs):
            break
        if max_ > len(dialogs):
            max_ = len(dialogs)

        # post
        distances = pos_tfidfs[:, min_:max_].transpose().dot(ent_tfidfs).tocsr()

        for ii in range(min_, max_):
            my_distances = distances[ii - min_, :].tocsr()
            if len(my_distances.data) <= top_k:
                o_sort = np.argsort(-my_distances.data)
                top_doc_indices[ii, :len(o_sort)] = my_distances.indices[o_sort]
                top_doc_distances[ii, :len(o_sort)] = my_distances.data[o_sort]
                top_doc_indices[ii, len(o_sort):] = 0
                top_doc_distances[ii, len(o_sort):] = 0
            else:
                o_sort = np.argpartition(-my_distances.data, top_k)[:top_k]
                top_doc_indices[ii, :] = my_distances.indices[o_sort]
                top_doc_distances[ii, :] = my_distances.data[o_sort]

    # Load entity metadata and conver to kb_id.
    metadata_file = os.path.join(base_dir, "entities.json")
    entity2id, entity2title, entity2real_title, entity2s1, entity2s2, entity2s3, entity2s4, entity2s5, entity2hop2 = json.load(
        tf.gfile.Open(metadata_file))
    id2entity = {i: e for e, i in entity2id.items()}
    id2name = {i: entity2title[e] for e, i in entity2id.items()}
    id2real_name = {i: entity2real_title[e] for e, i in entity2id.items()}
    id2s1 = {i: entity2s1[e] for e, i in entity2id.items()}
    id2s2 = {i: entity2s2[e] for e, i in entity2id.items()}
    id2s3 = {i: entity2s3[e] for e, i in entity2id.items()}
    id2s4 = {i: entity2s4[e] for e, i in entity2id.items()}
    id2s5 = {i: entity2s5[e] for e, i in entity2id.items()}
    id2hop2 = {i: entity2hop2[e] for e, i in entity2id.items()}
    # id2cont = {i: entity2cont[e] for e, i in entity2id.items()}
    mentions = []
    for ii in tqdm(range(len(dialogs))):
        my_mentions = []
        pos_top_inds = []
        scores = []
        for m in range(top_k):
            pos_top_inds.append(top_doc_indices[ii, m])
            scores.append(top_doc_distances[ii, m])

        top_inds_dic = {}
        for i in range(len(pos_top_inds)):
            top_inds_dic[pos_top_inds[i]] = scores[i]

        # 排序
        top_inds_list = sorted(top_inds_dic.items(), key=lambda x: x[1], reverse=True)

        for top_inds, score in top_inds_list[:all_top_k]:
            hop1_dic = id2name[top_inds].split(" ")
            # hop2_dic = fc.expandEntities(hop1_dic)
            my_mentions.append({
                "kb_id": id2entity[top_inds],
                "hop1_score": str(score),
                "key_words": id2name[top_inds],
                "title": id2real_name[top_inds],
                "hop1_dic": hop1_dic,
                "s1": id2s1[top_inds],
                "s2": id2s2[top_inds],
                "s3": id2s3[top_inds],
                "s4": id2s4[top_inds],
                "s5": id2s5[top_inds],
                "hop2_dic": id2hop2[top_inds],
            })
        mentions.append(my_mentions)

    return mentions


def add_knowledge_to_dial(data_path, data_name, data_type, entity_dir, bert_vocab_file):
    dial_file = data_path + "{}_{}_for_link.json".format(data_name, data_type)
    output_file = data_path + "{}_entities_{}.json".format(data_name, data_type)

    print("Reading data...")
    with tf.gfile.Open(dial_file) as f:
        data = json.load(f)
    print("Done.")

    print("Entity linking %d dialogs...", len(data))
    all_dialogs = []
    tokenizer = tokenization_new.FullTokenizer(
        vocab_file=bert_vocab_file, do_lower_case=True)

    linked_entities = tfidf_linking(dialogs=data, base_dir=entity_dir, tokenizer=tokenizer, top_k=5,
                                    all_top_k=1)
    for ii, item in enumerate(data):
        all_dialogs.append({
            "_id":
                str(item["_id"]),
            "post":
                item["post"],
            "entities":
                linked_entities[ii],
            "response":
                item["response"],
            "his_con":
                item["his_con"],
        })

    print("Writing questions to output file...")
    f_out = tf.gfile.Open(output_file, "w")
    f_out.write("\n".join(json.dumps(q) for q in all_dialogs))
    f_out.close()
    print("===============================================")

def add_knowledge_to_dial_drkit_new(data_path, data_name, data_type, gh, fc):
    dial_file = data_path + "tmp_data/" + "{}_{}_for_link.json".format(data_name, data_type)
    output_file = data_path + "tmp_data/" + "{}_entities_drkit_story_{}.json".format(data_name, data_type)

    print("Reading data...")
    with tf.gfile.Open(dial_file) as f:
        data = json.load(f)
    print("Done.")

    print("Entity linking %d dialogs...", len(data))
    all_dialogs = []

    for ii, item in tqdm(enumerate(data), total=len(data)):
        sent = item["post"]
        hop1_dic, hop1_cont, hop1_cont_li = gh.get_hops(sent)
        entities = list(set(pp.getE(sent.split(" "))))  # 用空格进行词分割，关键词表里筛选一次
        #  扩展，通过分数取每个单词的相关单词
        expand_key_list = fc.expandEntities(entities)
        sents_dic = {}

        for i in range(len(hop1_cont)):
            s = pp.repla(hop1_cont[i])
            tmp_cnt = 0
            for j in expand_key_list:
                if j in s:
                    tmp_cnt += 1
            sents_dic[s] = tmp_cnt

        top_sents = sorted(sents_dic.items(), key=lambda x: x[1], reverse=True)
        hop1 = [top_sents[i][0] for i in range(5)]
        all_dialogs.append({
            "_id":
                str(item["_id"]),
            "post":
                item["post"],
            "hop1":
                hop1,
            "response":
                item["response"],
            "his_con":
                item["his_con"],
        })

    print("Writing knowledge_dialog to output file...")
    f_out = tf.gfile.Open(output_file, "w")
    f_out.write("\n".join(json.dumps(q) for q in all_dialogs))
    f_out.close()
    print("===============================================")


if __name__ == '__main__':
    global max_entity_length  # Maximum number of tokens per entity
    max_entity_length = 50
    make_idf = False
    preprocess_story = False
    index_story = False
    pre_con = preProcess()
    base = pre_con.base_path
    story_path = pre_con.knowledge_path
    vocab_file = pre_con.bert_vocab_file
    idf_file = pre_con.story_idf_file
    source_story_data = pre_con.source_story_data
    story_for_index = pre_con.story_for_index
    if make_idf:
        print("计算story word的idf......")
        input_file = source_story_data
        out_file = idf_file
        dataset = dts_Target(story_file=input_file)
        dataset.make_dataset(out_file)

    if preprocess_story:
        print("预处理知识.....")

        ps = ParseStories(story_path=story_path, story_file=source_story_data, idf_file=idf_file,
                          out_file=story_for_index)
        ps.preProcess()

    if index_story:
        print("给知识加入索引.....")
        multihop_output_dir = pre_con.entity_dir
        if not tf.gfile.Exists(multihop_output_dir):
            tf.gfile.MakeDirs(multihop_output_dir)
        makeIndex(vocab_file=vocab_file, story_file=story_for_index)
