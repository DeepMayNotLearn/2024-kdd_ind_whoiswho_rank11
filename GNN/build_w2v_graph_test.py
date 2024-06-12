import json
import os
import networkx as nx
import pandas as pd
from pypinyin import lazy_pinyin
from tqdm import tqdm
import torch_geometric as pyg
import dgl
import numpy as np
import re
def co_author_find(name_list1,name_list2):
    x1=[x.split(' ') for x in name_list1]
    x2=[x.split(' ') for x in name_list2]
    co_author=[]
    for name_set1 in x1:
        name_set1=set(name_set1)
        for name_set2 in x2:
            name_set2=set(name_set2)
            temp_set=name_set1&name_set2
            if len(temp_set)==len(name_set1):
                co_author.append(" ".join(list(name_set1)))
    return co_author
#读取数据
train_author=json.load(open('IND-WhoIsWho/ind_test_author_filter_public.json','r'))
#读取论文信息
with open("IND-WhoIsWho/pid_to_info_all_w2v.json",encoding = 'utf-8') as f:
    pid_train=json.load(f)
#建图
save_path = 'w2v_graph/test_b_graph_list'
if os.path.exists(save_path):
    Graph_list=dgl.load_graphs(save_path)[0]
else:
    Graph_list=[]
start_idx=len(Graph_list)
graph_idx=-1
print(start_idx)
for id,person_info in tqdm(train_author.items()):
    name=person_info['name']
    normal_paper=person_info['papers']
    train_feats=[]
    graph_idx+=1
    if graph_idx<start_idx:
        continue
    #获取其论文的列表
    paper_list=[]
    for x in normal_paper:
        paper_dict=pid_train[x].copy()
        paper_dict['label']=1
        for index in range(256):
            paper_dict['w2v'+str(index)]=paper_dict['w2v'][index]
        del paper_dict['w2v']
        #提取特征
        #其他作者的特征
        org_len=0 #作者机构长度
        name_len=0 #作者名字长度
        # position=0 #作者在该篇文章的位置
        country_name=['China','Japan','Canada','Germany','France','Italy','Sweden','Australia','Singapore','Netherlands',] #机构所在国家
        country_cnt=[0,0,0,0,0,0,0,0,0,0]
        for index in range(len(paper_dict['authors'])):
            org_len+=len(paper_dict['authors'][index]['org'])
            name_len+=len(paper_dict['authors'][index]['name'])
            text = paper_dict['authors'][index]['org']
            for word in country_name:
                if word in text:
                    country_cnt[country_name.index(word)]+=1            
        #关键词的字符长度
        key_cnt=0
        for index in range(len(paper_dict['keywords'])):
            key_cnt+=len(paper_dict['keywords'][index])
        #期刊字词个数
        if paper_dict['venue'] is not None:
            venue_cnt=len(paper_dict['venue'].split(' '))
        else:
            venue_cnt=0
        try:
            train_feats.append(
                [len(paper_dict['title']),len(paper_dict['title'].split(' ')),#title字符长度，词个数
                 len(paper_dict['abstract']),len(paper_dict['abstract'].split(' ')), #abstract字符长度，词个数
                 len(paper_dict['abstract'])+len(fepaper_dictat['title']),len(paper_dict['abstract'].split(' '))+len(paper_dict['title'].split(' ')),#abstract+title字符长度，词个数
                 len(paper_dict['abstract'])+key_cnt,len(paper_dict['abstract'].split(' '))+len(paper_dict['keywords']),
                 # #abstract+keywords字符长度，词个数
                 len(paper_dict['title'])+key_cnt,len(paper_dict['title'].split(' '))+len(paper_dict['keywords']),
                 len(paper_dict['keywords']),key_cnt, #关键词字符长度，词个数
                 len(paper_dict['authors']),org_len,name_len, #作者个数，作者机构长度，名字长度
                 len(paper_dict['title'])+len(paper_dict['abstract'])+key_cnt, #abstract+title+keywords字符长度
                 len(paper_dict['title'].split(' '))+len(paper_dict['abstract'].split(' '))+len(paper_dict['keywords']),#abstract+title+keywords词个数
                 int(paper_dict['year']),len(paper_dict['venue']),venue_cnt,sum(country_cnt),
                 len(paper_dict['keywords'])/len(paper_dict['abstract'].split(' ')) #年份，期刊字符长度,期刊字词个数
                 ]+country_cnt)#作者机构所属国家个数
        except:
            train_feats.append(
               [len(paper_dict['title']),len(paper_dict['title'].split(' ')),#title字符长度，词个数
                 len(paper_dict['abstract']),len(paper_dict['abstract'].split(' ')), #abstract字符长度，词个数
                 len(paper_dict['abstract'])+len(paper_dict['title']),len(paper_dict['abstract'].split(' '))+len(paper_dict['title'].split(' ')),#abstract+title字符长度，词个数
                len(paper_dict['abstract'])+key_cnt,len(paper_dict['abstract'].split(' '))+len(paper_dict['keywords']),
                 # #abstract+keywords字符长度，词个数
                 len(paper_dict['title'])+key_cnt,len(paper_dict['title'].split(' '))+len(paper_dict['keywords']),
                 len(paper_dict['keywords']),key_cnt, #关键词字符长度，词个数
                 len(paper_dict['authors']),org_len,name_len, #作者个数，作者机构长度，名字长度
                 len(paper_dict['title'])+len(paper_dict['abstract'])+key_cnt, #abstract+title+keywords字符长度
                 len(paper_dict['title'].split(' '))+len(paper_dict['abstract'].split(' '))+len(paper_dict['keywords']),#abstract+title+keywords词个数
                 2000,0,venue_cnt,sum(country_cnt),len(paper_dict['keywords'])/len(paper_dict['abstract'].split(' '))  #年份，期刊字符长度,期刊字词个数
                 ]+country_cnt)#作者机构所属国家个数
        #对标题字段进行拼接
        if len(paper_dict['title'])>0:
            paper_dict['title']='title:'+paper_dict['title']
        else:
            paper_dict['title']='title:'+"None"
        #对摘要字段进行拼接
        if len(paper_dict['abstract'])>0:
            paper_dict['abstract']='abstract:'+paper_dict['abstract']
        else:
            paper_dict['abstract']='abstract:'+"None"
        #对关键词字段进行拼接
        if len(paper_dict['keywords'])>0:
            paper_dict['akeywords']='keywords:'+" ".join(paper_dict['keywords'])
        else:
            paper_dict['akeywords']='keywords:'+"None"
        #将论文作者处理出来
        paper_dict['author_name']=[]
        for authoer in paper_dict['authors']:
            author_name=authoer['name']
            #如果是中文，转为拼音
            if re.search(r'[\u4e00-\u9fff]',author_name):
                author_name=" ".join(lazy_pinyin(author_name))
            #去除名字中的符号
            author_name=re.sub(r'[^\u4e00-\u9fa5\w\s]', '', author_name)
            author_name=author_name.lower()
            paper_dict['author_name'].append(author_name)
        paper_dict['author_org']=[authoer['org'].lower() for authoer in paper_dict['authors']] 
        del paper_dict['authors']
        paper_dict['text']=paper_dict['title']+" "+paper_dict['abstract'][0:300]+" "+paper_dict['akeywords']+" venue:"+str(paper_dict['venue']) \
        +" year:"+str(paper_dict['year'])
        paper_list.append(paper_dict)
    #建立其论文 df
    paper_df=pd.DataFrame(paper_list)
    train_feats=np.array(train_feats)
    train_feats=pd.DataFrame(train_feats)
    train_cols=['len_title','title_cnt','len_abstract','abstract_cnt','len_title_abstract','abstract_title_cnt',
                       'abstract_keywords_len','cnt_abstract_keywords','title_keywords_len','cnt_title_keywords',
                       'len_keywords','keywords_cnt','len_authors','len_org','len_name','len_title_abstract_keywords',
                       'title_keywords_abstract_cnt','year','len_venue','venue_cnt','sum_country_cnt','key/abstract']+[f'country_{index}' for index in range(len(country_cnt))]
    train_feats.columns = train_cols
    paper_df.drop(['year'],axis=1,inplace=True)
    paper_df = pd.concat([paper_df,train_feats], axis=1)
    feat_col=['w2v'+str(index) for index in range(256)]+train_cols
    #开始根据其共同作者建立其边集
    edge_list=[]
    #初始化图
    G=nx.DiGraph()
    for index1,row1 in paper_df.iterrows():
        paper1=row1['id']
        label=row1['label']
        #用id作为节点名,各种拼接结果过一次scibert作为特征，进行节点的初始化
        node_feat=row1[feat_col].values
        G.add_node(paper1,x=list(node_feat),
                       y=[label])
    for index1,row1 in paper_df.iterrows():
        start_author=row1['author_name']
        paper1=row1['id']
        label1=row1['label']
        for index2,row2 in paper_df.iterrows():
            if index1==index2:
                continue
            end_author=row2['author_name']
            paper2=row2['id']
            if (len(normal_paper))>800:
                co_author=co_author_find(start_author,end_author)
            else:
                co_author=list(set(start_author)&set(end_author))
            co_author_link=len(co_author) 
            if co_author_link>1:
                edge_list.append({'paper1':paper1,'paper2':paper2,"co_author_num":co_author_link-1})
    edge_df=pd.DataFrame(edge_list)
    #在节点中添加边
    for eindex,erow in edge_df.iterrows():
        source_id=erow['paper1']
        target_id=erow['paper2']
        G.add_edge(source_id,target_id,edge_attr=[erow['co_author_num']])
    #开始根据其共同组织建立其边集
    edge_list=[]
    for index1,row1 in paper_df.iterrows():
        start_org=set(row1['author_org'])
        paper1=row1['id']
        label=row1['label']
        for index2,row2 in paper_df.iterrows():
            if index1==index2:
                continue
            end_org=set(row2['author_org'])
            paper2=row2['id']
            co_org=list(start_org&end_org)
            co_org_link=len(co_org)
            if co_org_link>1:
                edge_list.append({'paper1':paper1,'paper2':paper2,"co_org_num":co_org_link-1})
    edge_df=pd.DataFrame(edge_list)
    #在节点中添加边
    for eindex,erow in edge_df.iterrows():
        source_id=erow['paper1']
        target_id=erow['paper2']
        G.add_edge(source_id,target_id,edge_attr=[erow['co_org_num']])
    #开始根据其共同期刊建立其边集
    edge_list=[]
    for index1,row1 in paper_df.iterrows():
        start_venue=row1['venue']
        paper1=row1['id']
        label=row1['label']
        for index2,row2 in paper_df.iterrows():
            if index1==index2:
                continue
            end_venue=row2['venue']
            paper2=row2['id']
            if start_venue==end_venue:
                edge_list.append({'paper1':paper1,'paper2':paper2,"if_co_venue":1})
    edge_df=pd.DataFrame(edge_list)
    #在节点中添加边
    for eindex,erow in edge_df.iterrows():
        source_id=erow['paper1']
        target_id=erow['paper2']
        G.add_edge(source_id,target_id,edge_attr=[erow['if_co_venue']])
    #开始根据其共同关键字建立其边集
    edge_list=[]
    for index1,row1 in paper_df.iterrows():
        start_kw=set(row1['keywords'])
        paper1=row1['id']
        label=row1['label']
        for index2,row2 in paper_df.iterrows():
            if index1==index2:
                continue
            end_kw=set(row2['keywords'])
            paper2=row2['id']
            co_kw=list(start_kw&end_kw)
            co_kw_link=len(co_kw)
            if co_kw_link>0:
                edge_list.append({'paper1':paper1,'paper2':paper2,"co_kw_num":co_kw_link})
    edge_df=pd.DataFrame(edge_list)
    #在节点中添加边
    for eindex,erow in edge_df.iterrows():
        source_id=erow['paper1']
        target_id=erow['paper2']
        G.add_edge(source_id,target_id,edge_attr=[erow['co_kw_num']])
    device='cpu'
    try:
        G_data=dgl.from_networkx(G,node_attrs=['x','y']).to(device)
    except IndexError:
        G_data=dgl.from_networkx(G,node_attrs=['x','y']).to(device)
    except TypeError:
        G_data=dgl.from_networkx(G,node_attrs=['x','y']).to(device)
    Graph_list.append(G_data)
    #将图列表保存到本地文件
    dgl.save_graphs(save_path, Graph_list)