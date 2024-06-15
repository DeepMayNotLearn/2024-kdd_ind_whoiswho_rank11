import random
import os
import re
import dgl
import torch
import networkx as nx
import pandas as pd
import warnings
import torch_geometric as pyg
import pandas as pd
import networkx as nx
import numpy as np
import torch_geometric as pyg
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import Linear
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error,r2_score
from dgl import GCNNorm
from model import sci_model
from sklearn.metrics import roc_auc_score
from pypinyin import lazy_pinyin
from tqdm import tqdm
import json

#加载最优的模型
device=('cuda' if torch.cuda.is_available() else 'cpu')
model=sci_model().to(device)
model.load_state_dict(torch.load('model/sci_model.pt'))
warnings.filterwarnings("ignore")
#预测B榜测试集
test_save_path = 'w2v_graph/test_b_graph_list'
test_graph_list = dgl.load_graphs(test_save_path)[0]
test_save_path = 'coauthor_graph_new/test_b_graph_list'
test_graph_list1 = dgl.load_graphs(test_save_path)[0]
new_graph_list=[]
for i in tqdm(range(len(test_graph_list))):
    data1=test_graph_list[i]
    data2=test_graph_list1[i]
    if data1.ndata['x'].size()[0]==data2.ndata['x'].size()[0]:
        data1.ndata['x']=torch.cat([data1.ndata['x'],data2.ndata['x']],dim=1)
        new_graph_list.append(data1)
test_graph_list=new_graph_list  
test_papers_path = 'coauthor_graph_new/test_b_paper_info.json'
test_papers_json=json.load(open(test_papers_path,'r',encoding='utf-8'))
res=dict()
for i in tqdm(range(len(test_graph_list))):
    graph=test_graph_list[i].to(device)
    person_info=test_papers_json[i]
    person_id=person_info['id']
    paper_list=person_info['paper']
    output= model(graph).cpu().detach().numpy().reshape(1,-1).tolist()[0]
    person_p={paper_list[x]:output[x] for x in range(len(output))}
    res[person_id]=person_p
#保存结果
#B
with open('result/graph_res.json', 'w') as file:
        # 使用 json.dump() 将列表写入文件
        json.dump(res, file)