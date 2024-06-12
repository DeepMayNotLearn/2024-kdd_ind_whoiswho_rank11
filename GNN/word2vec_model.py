import json
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tqdm import tqdm 
import logging
import re
import numpy as np
# 将每篇文章中各个单词的词向量求均值表示文档向量
def get_doc_vec(comments_cut, vec_size=256):
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
    return doc2vec

#读取数据
train_author=json.load(open('IND-WhoIsWho/train_author.json','r'))
with open("IND-WhoIsWho/ind_valid_author.json",encoding = 'utf-8') as f:
    valid_author=json.load(f)
with open("IND-WhoIsWho/ind_valid_author_submit.json",encoding = 'utf-8') as f:
    submission=json.load(f)
#读取论文信息
with open("IND-WhoIsWho/pid_to_info_all.json",encoding = 'utf-8') as f:
    pid_to_info=json.load(f)
#停用词
stoplist = []
with open("IND-WhoIsWho/stopwords_English.txt",'r',encoding = 'utf-8') as f:
    for i in f.readlines():
        stoplist.append(i.strip())
stopword = ['at','based','in','of','for','on','and','to','an','using','with','the','by','we','be','is','are','can']   #通用停用词
stopword1 = ['university','univ','china','department','dept','laboratory','lab','school','al','et',
                 'institute','inst','college','chinese','beijing','journal','science','international']                    ##特别停用词
stopwords_check = ['a', 'was', 'were', 'that', '2', 'key', '1', 'technology', '0', 'sciences', 'as',
                    'from', 'r', '3', 'academy', 'this', 'nanjing', 'shanghai', 'state', 's', 'research',
                    'p', 'results', 'peoples', '4', 'which', '5', 'high', 'materials', 'study', 'control',
                    'method', 'group', 'c', 'between', 'or', 'it', 'than', 'analysis', 'system',  'sci',
                    'two', '6', 'has', 'h', 'after', 'different', 'n', 'national', 'japan', 'have', 'cell',
                    'time', 'zhejiang', 'used', 'data', 'these','wuhan','tsinghua']
stoplist+=stopword+stopword1+stopwords_check
puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
## 构造全部论文语料库
trainData = []
for id,paper_info in tqdm(pid_to_info.items()):
    st = []
    keyword= []
    venue= []
    org=[]
    feat=pid_to_info[id]
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
# 训练word2vec
# 设置 gensim 的日志级别为 INFO
logging.basicConfig(level=logging.INFO)
vec_size = 256
model = Word2Vec(trainData, 
                 min_count=1, 
                 vector_size=vec_size, 
                 window=5,
                 sg=1,
                 workers=10, 
                 seed=0)
vec_size=256
#获取每篇文档向量
w2v_train = get_doc_vec(trainData, vec_size)
train_w2v_list=[x.tolist() for x in w2v_train]
keys_li=list(pid_to_info.keys())
#将文档向量添加到每篇论文中
for i in tqdm(range(len(keys_li))):
    key=keys_li[i]
    paper=pid_to_info[key]
    paper['w2v']=train_w2v_list[i]
#保存添加后的字典
with open('IND-WhoIsWho/pid_to_info_all_w2v.json', 'w') as file:
        # 使用 json.dump() 将列表写入文件
        json.dump(pid_to_info, file)