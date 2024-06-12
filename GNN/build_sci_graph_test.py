import json
import torch 
from transformers import BertTokenizer,BertModel 
import networkx as nx
import pandas as pd
from pypinyin import lazy_pinyin
from tqdm import tqdm
import torch_geometric as pyg
import dgl
import re
import os
#文本转scibert向量
def text2scibert(text):
    inputs=tokenizer(text,return_tensors='pt',max_length=512, truncation=True,padding=True)
    scibert.eval()
    device='cuda'
    scibert.to(device)
    with torch.no_grad():
        inputs['input_ids']=inputs['input_ids'].to(device)
        inputs['token_type_ids']=inputs['token_type_ids'].to(device)
        inputs['attention_mask']=inputs['attention_mask'].to(device)
        output=scibert(**inputs)['pooler_output'].detach()
    return output
#共同作者匹配算法
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
pid_train=json.load(open('IND-WhoIsWho/pid_to_info_all.json','r',encoding='utf-8'))
#加载模型
scibert=BertModel.from_pretrained('scibert_model')
tokenizer=BertTokenizer.from_pretrained('scibert_tokenizer')
# 从本地文件加载图数据
save_path = 'sci_graph/test_b_graph_list'
info_save_path='sci_graph/test_b_paper_info.json'
if os.path.exists(save_path):
    Graph_list=dgl.load_graphs(save_path)[0]
    Graph_info_list=json.load(open(info_save_path,'r',encoding='utf-8'))
else:
    Graph_list=[]
    Graph_info_list=[]
start_idx=len(Graph_list)
print(start_idx)
i=-1
for id,person_info in tqdm(train_author.items()):
     #个人信息
    name=person_info['name']
    normal_paper=person_info['papers']
    i+=1
    if i<start_idx:
        continue
    #获取其论文的列表
    paper_list=[]
    for x in normal_paper:
        paper_dict=pid_train[x].copy()
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
        paper_dict['text']="author :"+name+'co:'+" ".join(paper_dict['author_name'])+" org:"+" ".join(paper_dict['author_org'])+paper_dict['title']+ \
        " "+paper_dict['abstract'][0:300]+" "+paper_dict['akeywords']+" venue:"+str(paper_dict['venue']) \
        +" year:"+str(paper_dict['year'])
        paper_list.append(paper_dict)
    #建立其论文 df
    paper_df=pd.DataFrame(paper_list)
    text_li=paper_df['text'].tolist()
    n_len=256
    text_vec_list=[]
    index=0
    while(index<=(len(text_li)//n_len)):
        if index*n_len+n_len>len(text_li):
            text=text_li[index*n_len:]
        else:
            text=text_li[index*n_len:index*n_len+n_len]
        text_vec=text2scibert(text)
        text_vec_list.append(text_vec)
        index+=1
    result=torch.cat(text_vec_list,dim=0).cpu().numpy()
    # 将结果转换为 DataFrame
    result_df = pd.DataFrame(result, columns=['text2vec'+str(i) for i in range(768)])
    # 将新的 DataFrame 与原始的 paper_df 沿着列方向连接
    paper_df = pd.concat([paper_df, result_df], axis=1)
    #开始根据其共同作者建立其边集
    edge_list=[]
    #初始化图
    G=nx.DiGraph()
    for index1,row1 in paper_df.iterrows():
        paper1=row1['id']
        #用id作为节点名,各种拼接结果过一次scibert作为特征，进行节点的初始化
        node_feat=row1[['text2vec'+str(i) for i in range(768)]].values
        G.add_node(paper1,x=list(node_feat),id=index1)
    for index1,row1 in paper_df.iterrows():
        start_author=row1['author_name']
        paper1=row1['id']
        for index2,row2 in paper_df.iterrows():
            if index1==index2:
                continue
            end_author=row2['author_name']
            paper2=row2['id']
            if len(normal_paper)>800:
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
    # 转换前记录节点的原始ID和顺序
    original_ids = list(G.nodes())
    try:
        G_data=dgl.from_networkx(G,node_attrs=['x','id'],edge_attrs=['edge_attr']).to(device)
    except IndexError:
        G_data=dgl.from_networkx(G,node_attrs=['x','id']).to(device)
    paper_ids=[original_ids[i] for i in G_data.ndata['id'].tolist()]
    Graph_list.append(G_data)
    Graph_info_list.append({'id':id,'paper':paper_ids})
    # 指定保存路径和文件名
    # 将图列表保存到本地文件
    dgl.save_graphs(save_path, Graph_list)
    with open(info_save_path, 'w') as file:
        # 使用 json.dump() 将列表写入文件
        json.dump(Graph_info_list, file)