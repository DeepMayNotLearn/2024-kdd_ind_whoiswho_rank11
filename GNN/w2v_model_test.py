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
from model import w2v_model
from sklearn.metrics import roc_auc_score
from pypinyin import lazy_pinyin
from tqdm import tqdm
import json

#加载最优的模型
device=('cuda' if torch.cuda.is_available() else 'cpu')
model=torch.load('model/w2v_model.pt').to(device)
warnings.filterwarnings("ignore")
#可复现性设置
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)
set_seed(8019)

#开始预测结果
test_save_path = 'w2v_graph/test_b_graph_list'
test_graph_list = dgl.load_graphs(test_save_path)[0]
test_papers_path = 'sci_graph/test_b_paper_info.json'
test_papers_json=json.load(open(test_papers_path,'r',encoding='utf-8'))
#预测测试集
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
with open('result/w2v_res.json', 'w') as file:
        # 使用 json.dump() 将列表写入文件
        json.dump(res, file)