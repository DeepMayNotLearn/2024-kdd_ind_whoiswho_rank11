#导入所用库
import pandas as pd
import numpy as np
import json
import gensim
from scipy.stats import kurtosis
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')
import logging
import re
import multiprocessing
# 设置 gensim 的日志级别为 INFO
logging.basicConfig(level=logging.INFO)

#设置随机种子
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
seed_everything(2024)

#读取数据
with open("train_author.json",encoding = 'utf-8') as f:
    train_author=json.load(f)

with open("ind_test_author_filter_public.json",encoding = 'utf-8') as f:
    test_author=json.load(f)

with open("ind_test_author_submit.json",encoding = 'utf-8') as f:
    submission=json.load(f)

with open("pid_to_info_all.json",encoding = 'utf-8') as f:
    pid_to_info=json.load(f)

#停用词
stopword = ['did','not','at','based','in','of','for','on','and','to','an','using','with','the','by','we','be','is','are','can']   #通用停用词
stopword1 = ['university','univ','china','department','dept','laboratory','lab','school','al','et',
                 'institute','inst','college','chinese','beijing','journal','science','international']                    ##特别停用词
stopwords_check = ['a', 'was', 'were', 'that', '2', 'key', '1', 'technology', '0', 'sciences', 'as',
                    'from', 'r', '3', 'academy', 'this', 'nanjing', 'shanghai', 'state', 's', 'research',
                    'p', 'results', 'peoples', '4', 'which', '5', 'high', 'materials', 'study', 'control',
                    'method', 'group', 'c', 'between', 'or', 'it', 'than', 'analysis', 'system',  'sci',
                    'two', '6', 'has', 'h', 'after', 'different', 'n', 'national', 'japan', 'have', 'cell',
                    'time', 'zhejiang', 'used', 'data', 'these','wuhan','tsinghua','kyoto']
stoplist= stopword+stopword1+stopwords_check
puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'

## 构造训练集语料库
trainData = []
for id,person_info in tqdm(train_author.items()):
    for text_id in person_info['normal_data']:
        st = []
        keyword= []
        venue= []
        org=[]
        feat=pid_to_info[text_id]
        for i in feat['keywords']:
            i = i.lower()
            keyword.append(i)
        if feat['venue'] is not None:
            venue = feat['venue'].strip()
            venue = venue.lower()
            venue = re.sub(puncs, ' ', venue)
            venue = re.sub(r'\s{2,}', ' ', venue).strip()
            venue = venue.split(' ')
            venue = [word for word in venue if len(word) > 1]
            venue = [word for word in venue if len(word) > 1 and word not in stoplist]
        st+= feat['title'].lower().split()+feat['abstract'].lower().split()+keyword+venue
        for i in range(len(feat['authors'])):
            org=feat['authors'][i]['org'].strip()
            org = org.lower()
            org = re.sub(puncs, ' ', org)
            org = re.sub(r'\s{2,}', ' ', org).strip()
            org = org.split(' ')
            org = [word for word in org if len(word) > 1 and word not in stoplist]
            st+=org
        trainData.append(st)
    for text_id in person_info['outliers']:
        st = []
        keyword= []
        venue= []
        org=[]
        feat=pid_to_info[text_id]
        for i in feat['keywords']:
            i = i.lower()
            keyword.append(i)
        if feat['venue'] is not None:
            venue = feat['venue'].strip()
            venue = venue.lower()
            venue = re.sub(puncs, ' ', venue)
            venue = re.sub(r'\s{2,}', ' ', venue).strip()
            venue = venue.split(' ')
            venue = [word for word in venue if len(word) > 1]
            venue = [word for word in venue if len(word) > 1 and word not in stoplist]
        st+= feat['title'].lower().split()+feat['abstract'].lower().split()+keyword+venue
        for i in range(len(feat['authors'])):
            org=feat['authors'][i]['org'].strip()
            org = org.lower()
            org = re.sub(puncs, ' ', org)
            org = re.sub(r'\s{2,}', ' ', org).strip()
            org = org.split(' ')
            org = [word for word in org if len(word) > 1 and word not in stoplist]
            st+=org
        trainData.append(st)

## 构造测试集语料库
testData = []
for id,person_info in tqdm(test_author.items()):
    for text_id in person_info['papers']:
        st = []
        keyword= []
        venue= []
        org=[]
        feat=pid_to_info[text_id]
        for i in feat['keywords']:
            i = i.lower()
            keyword.append(i)
        if feat['venue'] is not None:
            venue = feat['venue'].strip()
            venue = venue.lower()
            venue = re.sub(puncs, ' ', venue)
            venue = re.sub(r'\s{2,}', ' ', venue).strip()
            venue = venue.split(' ')
            venue = [word for word in venue if len(word) > 1]
            venue = [word for word in venue if len(word) > 1 and word not in stoplist]
        st+= feat['title'].lower().split()+feat['abstract'].lower().split()+keyword+venue
        for i in range(len(feat['authors'])):
            org=feat['authors'][i]['org'].strip()
            org = org.lower()
            org = re.sub(puncs, ' ', org)
            org = re.sub(r'\s{2,}', ' ', org).strip()
            org = org.split(' ')
            org = [word for word in org if len(word) > 1 and word not in stoplist]
            st+=org
        testData.append(st)


# 训练word2vec
vec_size = 100
model = Word2Vec(trainData+testData, min_count=1, vector_size=vec_size, window=10,sg = 1,epochs = 5,seed=0,workers=18)
model.save('w2v_kdd_100.model')
model.wv.save_word2vec_format('w2v_kdd_100.vector', binary=False)

# 将每篇文章中各个单词的词向量求均值表示文档向量
def get_doc_vec(comments_cut, vec_size=100):
    doc2vec = []
    for comment_cut in tqdm(comments_cut):
        doc_vec = np.zeros(vec_size)
        words_num = 0
        for word in comment_cut:
            # 若设置min_count>1，可能会筛去一些词，所以要判断当前词是否在字典中
            if word in model.wv:
                # 获取当前词的词向量
                word_vec = model.wv.__getitem__(word)
                doc_vec = doc_vec + word_vec
                words_num += 1
        # 计算文档向量
        if words_num > 0:
            doc_vec = doc_vec / words_num
        doc2vec.append(doc_vec)
    return np.array(doc2vec)
vec_size = 100
w2v_train = get_doc_vec(trainData, vec_size)
w2v_test = get_doc_vec(testData,vec_size)

train_w2v = pd.DataFrame(w2v_train,columns = [f'w2v_{i}' for i in range(100)])
train_w2v.to_csv('train_w2v.csv',index = False)

test_w2v = pd.DataFrame(w2v_test,columns = [f'w2v_{i}' for i in range(100)])
test_w2v.to_csv('test_w2v.csv',index = False)