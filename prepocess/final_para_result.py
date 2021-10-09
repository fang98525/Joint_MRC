"""段落召回标签构造"""

import sys

sys.path.append('/data1/home/fzq/projects/MRC/')
import pandas as pd
import re
from tqdm import tqdm
from config import Config
from snippts import split_text, find_lcsubstr
import jieba.posseg as pseg
import numpy as np
import random
import codecs
import os
import pickle
from gensim.summarization import bm25

config = Config()
maxlen = config.sequence_length - config.q_len
# 加载停用词
stop_words = config.processed_data + 'baidu_stopwords.txt'
stopwords = codecs.open(stop_words, 'r', encoding='utf8').readlines()
stopwords = [w.strip() for w in stopwords]
stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']


def load_context(file_path):
    # 读取全部政策文件
    docid2context = {}
    f = True
    for line in open(file_path):
        if f:
            f = False
            continue
        r = line.strip().split('\t')
        docid2context[r[0]] = '\t'.join([r[i] for i in range(1, len(r))])
    return docid2context    #文件id to内容


def get_para_df(docid2context):
    STOPS = (
        '\xa0'  # 换行符
        '\u3000'  # 顶格
        '\u2002'  # 空格
        '\u2003'  # 2空格
    )
    dealtext_list = []
    for docid in tqdm(docid2context):
        text = docid2context[docid]
        body = text
        body = body.strip()
        if body.__len__() <= maxlen:
            deal_text = [body]
        elif re.search('\xa0|\u3000|\u2002|\u2003', body):
            """段落里有明显的分割符"""
            deal_text = split_text(body, maxlen=maxlen, split_pat='default', greedy=False)[0]
        else:
            """使用空格作为分隔符"""
            split_pat = '([{ }]”?)'
            deal_text = split_text(body, maxlen=maxlen, split_pat=split_pat, greedy=False)[0]
        i = 0
        for text_ in deal_text:
            res_dict = {'paraid': docid + '_' + str(i), 'docid': docid, 'text': text_}
            dealtext_list.append(res_dict)
            i += 1
    para_df = pd.DataFrame(dealtext_list)
    return para_df


def refind_answer(train_df):
    answer_list = []
    for i in tqdm(range(len(train_df))):
        docid = train_df['docid'][i]
        answer = train_df['answer'][i].strip()
        try:
            contexts = para_df[para_df['docid'] == docid]['text']
        except:
            contexts = []
        flag = 0
        for context in contexts:
            if context.count(answer):
                flag = 1
        if flag == 1:
            answer_list.append(answer)
        else:
            """找答案字更多的段落作为答案"""
            _answer = ''
            for context in contexts:
                lcs = find_lcsubstr(context, answer)[0]
                if len(lcs) >= _answer.__len__():
                    _answer = lcs
            answer_list.append(_answer)
    return answer_list


def tokenization(text):
    # 对一篇文章分词、去停用词
    result = []
    words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result


def load_corpus(context_df):
    corpus = []
    paraid_list = []
    for i in tqdm(range(len(context_df))):
        # print(context_df['text'])
        text_list = tokenization(context_df['text'][i])
        corpus.append(text_list.copy())
        paraid_list.append(context_df['paraid'][i])
    return corpus, paraid_list


def generate_para(train_df, id2para, bm25Model, dev_num=0):
    """本地计算BM25构造样本"""
    Doc_train = []
    for i in tqdm(range(len(train_df))):
        answer = train_df['answer'].iloc[i]
        qid = train_df['id'].iloc[i]
        question = train_df['question'].iloc[i]
        doc_id = train_df['docid'].iloc[i]
        context = para_df[para_df['docid'] == doc_id]['text']
        """pos"""
        pos_text = []
        for text in context:
            if text.count(answer):
                start = text.find(answer)
                doc_dict = {'q_id': qid, 'query': question, 'start': start, 'context': text, 'answer': answer,
                            'score': 1}
                pos_text.append(text)
                Doc_train.append(doc_dict)
        """neg"""
        query = tokenization(question)
        scores = bm25Model.get_scores(query)
        scores = np.array(scores)
        sort_index = np.argsort(-scores)[:50]  # 500选三非正样本作负样本
        para_ids = [paraid_list[i] for i in sort_index]
        if dev_num:
            para_id = random.sample(para_ids, dev_num)
        else:
            para_id = random.sample(para_ids, len(pos_text))  # 从50选正样本个数做负样本
        for idx in para_id:
            neg_text = id2para[idx]
            while neg_text in pos_text:
                neg_text = id2para[random.sample(para_ids, 1)[0]]
            doc_dict = {'q_id': qid, 'query': question, 'start': -1, 'context': neg_text, 'answer': answer,
                        'score': 0}
            Doc_train.append(doc_dict)
    doc_df = pd.DataFrame(Doc_train)
    return doc_df


def test_generate_para(test_df, id2para, bm25Model, dev_num=25):
    """本地计算BM25构造样本"""
    Doc_train = []
    for i in tqdm(range(len(test_df))):
        qid = test_df['id'].iloc[i]
        question = test_df['question'].iloc[i]
        """bm25"""
        query = tokenization(question)
        scores = bm25Model.get_scores(query)
        scores = np.array(scores)
        sort_index = np.argsort(-scores)[:50]  # 500选三非正样本作负样本,直接得到topk数据的分数索引
        para_ids = [paraid_list[i] for i in sort_index]
        para_id = random.sample(para_ids, dev_num)
        for idx in para_id:
            neg_text = id2para[idx]
            doc_dict = {'q_id': qid, 'query': question, 'start': -1, 'context': neg_text, 'answer': '',
                        'score': 0}
            Doc_train.append(doc_dict)
    doc_df = pd.DataFrame(Doc_train)
    return doc_df


if __name__ == '__main__':

    """
    文件拆分成段落,并赋id
    """
    print("****GENERATE PARAGRAPH...****")
    docid2context = load_context(config.processed_data + 'NCPPolicies_context_20200301.csv')
    para_df = get_para_df(docid2context)  # 切分文档成段落

    para_df.to_csv(config.processed_data + 'para_context.csv', index=False, encoding='utf_8_sig')
    """训练集答案找回"""
    print("****REFIND ANSWER...****")
    train_df = pd.read_csv(config.processed_data + 'NCPPolicies_train_20200301.csv', sep='\t', error_bad_lines=False)
    answer_list = refind_answer(train_df)
    train_df['answer'] = answer_list

    """
    构造联合学习BERT训练集与验证集
    """
    print("****GENERATE TRAINDATA...****")
    stop_words = config.processed_data + 'stop_words.txt'
    stopwords = codecs.open(stop_words, 'r', encoding='utf8').readlines()
    stopwords = [w.strip() for w in stopwords]
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    if os.path.exists(config.processed_data + 'corpus_{}.pickle'.format(config.sequence_length)):
        with open(config.processed_data + 'corpus_{}.pickle'.format(config.sequence_length), 'rb') as cp:
            corpus = pickle.load(cp)
        with open(config.processed_data + 'pid.pickle_{}'.format(config.sequence_length), 'rb') as id:
            paraid_list = pickle.load(id)
    else:
        corpus, paraid_list = load_corpus(para_df)
        with open(config.processed_data + 'corpus.pickle_{}'.format(config.sequence_length), 'wb') as cp:
            pickle.dump(corpus, cp)
        with open(config.processed_data + 'pid.pickle_{}'.format(config.sequence_length), 'wb') as id:
            pickle.dump(paraid_list, id)

    if not os.path.exists(config.processed_data + 'bm25_corpus.pickle_{}'.format(config.sequence_length)):
        bm25Model = bm25.BM25(corpus)
        with open(config.processed_data + 'bm25_corpus.pickle_{}'.format(config.sequence_length), 'wb') as cp:
            pickle.dump(bm25Model, cp)
    else:
        with open(config.processed_data + 'bm25_corpus.pickle_{}'.format(config.sequence_length), 'rb') as bm25_m:
            bm25Model = pickle.load(bm25_m)

    doc_train = train_df.iloc[:int(len(train_df) * 0.8)]
    doc_dev = train_df[int(len(train_df) * 0.8):]

    doc_dev.to_csv(config.processed_data + 'dev_like_test.csv')  # 如同测试集的验证集

    id2para = dict(zip(list(para_df['paraid']), list(para_df['text'])))

    para_train = generate_para(doc_train, id2para, bm25Model)  # 生成BM25模型，找出topk个候选段落
    para_dev = generate_para(doc_dev, id2para, bm25Model, dev_num=25)  # 生成BM25模型，找出topk个候选段落 一个问题对应多个候选段落的个数，产生多个数据   标签：0 or 1 answer：
    para_train['score'] = 0
    para_dev['score'] = 0

    para_train.to_csv(config.processed_data + 'train.csv', index=False, encoding='utf_8_sig')
    para_dev.to_csv(config.processed_data + 'dev.csv', index=False, encoding='utf_8_sig')

    """测试集构建(一个问题对应K个候选段落)"""
    print("****GENERATE TESTDATA...****")
    """提取出的候选段落"""

    """
    Bm25
    测试集与训练集的候选段落是一样的
    """
    test_lt = []
    test_rs_pd = pd.read_csv(config.processed_data + 'NCPPolicies_test.csv', sep='\t')
    test_df = test_generate_para(test_rs_pd, id2para, bm25Model, dev_num=25)
    test_df.to_csv(config.processed_data + 'test.csv', index=False, encoding='utf_8_sig')
