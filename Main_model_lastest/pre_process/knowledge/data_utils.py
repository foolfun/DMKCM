import nltk
import os
from nltk.stem import WordNetLemmatizer
from Main_model_lastest.configs_manage import preProcess
_lemmatizer = WordNetLemmatizer()
pre_ce= preProcess()
candi_ = pre_ce.base_path+'pre_process/convai2/candi_keyword.txt'
candi_keyword_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),candi_)
                                    # 这个是从wordnet里面筛选的
_candiwords = [x.strip() for x in open(candi_keyword_path).readlines()]

def tokenize(example, ppln):
    for fn in ppln:
        example = fn(example)
    return example


def kw_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower, pos_tag, to_basic_form])  # 仅筛选出名词，动词，形容词


def simp_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower])


def nltk_tokenize(string):
    return nltk.word_tokenize(string)


def lower(tokens):
    if not isinstance(tokens, str):
        return [lower(token) for token in tokens]
    return tokens.lower()


def pos_tag(tokens):
    return nltk.pos_tag(tokens)


def to_basic_form(tokens):
    if not isinstance(tokens, tuple):
        return [to_basic_form(token) for token in tokens]
    word, tag = tokens
    if tag.startswith('NN'):
        pos = 'n'
    elif tag.startswith('VB'):
        pos = 'v'
    elif tag.startswith('JJ'):
        pos = 'a'
    else:
        return word
    return _lemmatizer.lemmatize(word, pos)




def is_candiword(a):
    if a in _candiwords:
        return True
    return False


class KeywordExtractor():
    def __init__(self, idf_dict=None):
        self.idf_dict = idf_dict

    @staticmethod
    def is_keyword_tag(tag):
        return tag.startswith('VB') or tag.startswith('NN') or tag.startswith('JJ')

    @staticmethod
    def cal_tag_score(tag):
        if tag.startswith('VB'):
            return 1.
        if tag.startswith('NN'):
            return 2.
        if tag.startswith('JJ'):
            return 0.5
        return 0.

    def idf_extract(self, string, top_N=3, con_kw=None):  # 返回
        tokens = simp_tokenize(string)
        seq_len = len(tokens)
        tokens = pos_tag(tokens)
        source = kw_tokenize(string)
        candi = []
        result = {}
        for i, (word, tag) in enumerate(tokens):
            score = self.cal_tag_score(tag)
            if not is_candiword(source[i]) or score == 0.:
                continue
            if con_kw is not None and source[i] in con_kw:
                continue
            # 做了个score计算
            score *= source.count(source[i])
            score *= 1 / seq_len
            score *= self.idf_dict[source[i]]
            candi.append((source[i], score))

            if score > 0.15:  # 原来是0.15
                if result.__contains__(source[i]) and score <= result[source[i]]:  # 同样的单词，选最大的score
                    continue
                else:
                    result[source[i]] = score
        # print("候选的所有单词有：", candi)
        # print("候选的单词大于0.15的单词有：", result)
        if result == {}:
            return []
        else:
            res = sorted(result.items(), key=lambda result: result[1], reverse=True)
            kws = []
            for r in res:
                kws.append(r[0])
            if len(kws) > top_N:  # 如果大于N个单词，就筛选出top_N
                return kws[:top_N]
            elif len(kws) > 0:  # 否则，就全部返回
                return kws

    def extract(self, string):
        tokens = simp_tokenize(string)
        tokens = pos_tag(tokens)
        source = kw_tokenize(string)
        kwpos_alters = []
        for i, (word, tag) in enumerate(tokens):
            if source[i] and self.is_keyword_tag(tag):
                kwpos_alters.append(i)
        kwpos, keywords, no_in_file_keywords = [], [], []
        for id in kwpos_alters:
            if is_candiword(source[id]):
                keywords.append(source[id])
            no_in_file_keywords.append(source[id])
        if len(list(set(keywords))) > 0:
            return list(set(keywords))
        else:
            return list(set(no_in_file_keywords))


def getKeywords(sent, ke, is_title):
    if is_title:
        # if len(ke.idf_extract(sent)) > 0:
        #     key_list = ke.idf_extract(sent)
        # if len(ke.extract(sent)) > 0:
        #     key_list = ke.extract(sent)
        # else:
        key_list = simp_tokenize(sent)
    else:
        if len(ke.idf_extract(sent)) > 0:
            key_list = ke.idf_extract(sent)
        elif len(ke.extract(sent)) > 0:
            key_list = ke.extract(sent)
        else:
            return []
    new_key_list = []
    if len(key_list) > 0:
        mark = [".", "?", "!", ":", ",",";"]
        for k in key_list:
            if k not in mark:
                new_key_list.append(k)
    return new_key_list


def getTokens(sent_keywords, vocab):
    toks = []
    for i in sent_keywords:
        if i in vocab:
            toks.append(i)
    return toks



# def getKeywordTokens(sent, ke, tokenizer, vocab):
#     if len(ke.idf_extract(sent,top_N=3)) > 0:
#         return getTokens(ke.idf_extract(sent), vocab)
#     if len(ke.extract(sent)) > 0:
#         return getTokens(ke.extract(sent), vocab)
#     else:
#           return tokenizer.tokenize(sent)
