import os
from Main_model_lastest.configs_manage import preProcess
from Main_model_lastest.pre_process import tokenization_new
from Main_model_lastest.pre_process.knowledge.process_ex_knowledge import FindWordsfromConceptNet
from nltk.corpus import stopwords
from drkit.hotpotqa.demo_for_dialog_lastest import GetHops
from Main_model_lastest.pre_process.utils_for_preprocess import PreProcessingHelper

class PredictorHelper_DRKIT():
    def __init__(self,data_name):
        """Setup BERT model."""
        pre_con = preProcess()
        candi_ = pre_con.base_path + 'pre_process/convai2/candi_keyword.txt'
        candi_keyword_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), candi_)
        # 这个是从wordnet里面筛选的
        # candi_keyword_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                                   candi_)  # 这个是从wordnet里面筛选的
        _candiwords = [x.strip() for x in open(candi_keyword_path).readlines()]
        self.vocab = _candiwords

        vocab_file = pre_con.bert_vocab_file
        self.tokenizer = tokenization_new.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        self.stop_words = set(stopwords.words('english'))
        self.fc = FindWordsfromConceptNet(conceptnet_path=pre_con.knowledge_path + "conceptnet_en.txt")
        self.pp = PreProcessingHelper()
        self.gh = GetHops(20,5,data_name=data_name)

    def getE(self, sent_keywords):
        vocab = self.vocab
        toks = []
        for i in sent_keywords:
            if i in vocab:
                toks.append(i)
        return toks

    def run(self, post):
        """Run model on given post and entity.

        Args:
          post: String.

        Returns:
          logits: Numpy array of logits for each output entity.
        """

        hop1_dic, hop1_cont, hop1_cont_li = self.gh.get_hops(post)
        entities = list(set(self.pp.getE(post.split(" "))))  # 用空格进行词分割，关键词表里筛选一次
        #  扩展，通过分数取每个单词的相关单词
        expand_key_list = self.fc.expandEntities(entities)
        sents_dic = {}
        # 这种是找五个故事的方法
        for i in range(len(hop1_cont)):
            s = self.pp.repla(hop1_cont[i])
            tmp_cnt = 0
            for j in expand_key_list:
                if j in s:
                    tmp_cnt += 1
            sents_dic[s] = tmp_cnt

        top_sents = sorted(sents_dic.items(), key=lambda x: x[1], reverse=True)
        hop1 = [top_sents[i][0] for i in range(5)]
        return hop1

if __name__ == '__main__':
    #test
    pre = PredictorHelper_DRKIT("dailyDialog")
    print(pre.run("i love you"))