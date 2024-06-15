import pandas as pd
import networkx as nx
import torch_geometric as pyg
import numpy as np
import random
import os
import torch
import dgl
from tqdm import tqdm
import torch
import torch_geometric as pyg
from dgl.nn import GATv2Conv,SAGEConv,AGNNConv,SGConv,GINConv,GraphConv,APPNPConv,GATConv,DotGatConv,ChebConv,AvgPooling,GatedGraphConv,GatedGCNConv,GCN2Conv
from dgl.nn import TAGConv,RelGraphConv,GraphormerLayer,GCN2Conv
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from  torch_geometric.nn.conv import TransformerConv,ClusterGCNConv
from dgl import GCNNorm
from scipy.sparse.linalg import ArpackNoConvergence,ArpackError

class PyGLayer(torch.nn.Module):
    def __init__(self,name,layer_dims,residual=False):
        super().__init__()
        self.name=name
        self.norm=GCNNorm()
        self.input_dim=layer_dims[0]
        self.layer_num=len(layer_dims)-1
        self.appnp=APPNPConv(5,0.2,0.1)
        self.layers=nn.ModuleList()
        self.residual=residual
        num_head=1
        for i in range(self.layer_num):
            input_dim=layer_dims[i]
            output_dim=layer_dims[i+1]
            if self.residual:
                if i==0:
                    if self.name=='trans':
                        layer=TransformerConv(input_dim,output_dim,num_head)
                    if self.name=='clustergcn':
                        layer=ClusterGCNConv(input_dim,output_dim)
                else:
                    last_dim=layer_dims[i-1]
                    if self.name=='trans':
                        layer=TransformerConv(input_dim+last_dim,output_dim,num_head)
                    if self.name=='clustergcn':
                        layer=ClusterGCNConv(input_dim+last_dim,output_dim)
            else:
                if self.name=='trans':
                    layer=TransformerConv(input_dim,output_dim,num_head)
                if self.name=='clustergcn':
                        layer=ClusterGCNConv(input_dim,output_dim)
            self.layers.append(layer)
            
    def forward(self,data):
        data=self.norm(data)
        out=data.ndata['x']
        out=self.appnp(data,out)
        edge_index=torch.stack(data.edges(),dim=0)
        res=[out]
        if self.residual:
            for index in range(len(self.layers)):
                layer=self.layers[index]
                out=layer(out,edge_index).squeeze(1)
                res_out.append(out)
                if index!=(len(self.layers)-1) :
                    out=torch.cat([out,res_out[index]],dim=1)
                if index==(len(self.layers)-1):
                    out=F.mish(out)
        else:
            for index in range(len(self.layers)):
                layer=self.layers[index]
                out=layer(out,edge_index).squeeze(1)
                if index==(len(self.layers)-1):
                        out=F.mish(out)
        return out
        
                    

class GCNLayer(torch.nn.Module):
    def __init__(self,name,layer_dims,residual=False):
        super().__init__()
        self.name=name
        self.norm=GCNNorm()
        self.input_dim=layer_dims[0]
        self.output_dim=layer_dims[-1]
        self.layer_num=len(layer_dims)-1
        self.appnp=APPNPConv(5,0.2,0.1)
        self.layers=nn.ModuleList()
        self.residual=residual
        self.gnnformer=GraphormerLayer(self.output_dim,512,1)
        for i in range(self.layer_num):
            input_dim=layer_dims[i]
            output_dim=layer_dims[i+1]
            if self.residual:
                if i==0:
                    if self.name=='gatv2':
                        layer=GATv2Conv(input_dim,output_dim,1)
                    elif self.name=='gat':
                        layer=GATConv(input_dim,output_dim,1)
                    elif self.name=='sage':
                        layer=SAGEConv(input_dim,output_dim,'pool')
                    elif self.name=='sgc':
                        layer=SGConv(input_dim,output_dim,4)
                    elif self.name=='gin':
                        layer=GINConv(nn.Linear(input_dim,output_dim),'sum',learn_eps=True)
                    elif self.name=='dotgat':
                        layer=DotGatConv(input_dim,output_dim,1)
                    elif self.name=='cheb':
                        layer=ChebConv(input_dim,output_dim,2)
                    elif self.name=='gated':
                        layer=GatedGraphConv(input_dim,output_dim,5,5)
                    elif self.name=='tag':
                        layer=TAGConv(input_dim,output_dim,5)
                    elif self.name=='rel':
                        layer=RelGraphConv(input_dim,output_dim,4)
                    
                else:
                    last_dim=layer_dims[i-1]
                    if self.name=='gat':
                        layer=GATv2Conv(input_dim+last_dim,output_dim,1)
                    elif self.name=='gatv2':
                        layer=GATConv(input_dim+last_dim,output_dim,1)
                    elif self.name=='sage':
                        layer=SAGEConv(input_dim+last_dim,output_dim,'pool')
                    elif self.name=='sgc':
                        layer=SGConv(input_dim+last_dim,output_dim,4)
                    elif self.name=='gin':
                        layer=GINConv(nn.Linear(input_dim+last_dim,output_dim),'sum',learn_eps=True)
                    elif self.name=='dotgat':
                        layer=DotGatConv(input_dim+last_dim,output_dim,1)
                    elif self.name=='cheb':
                        layer=ChebConv(input_dim+last_dim,output_dim,2)
                    elif self.name=='gated':
                        layer=GatedGraphConv(input_dim+last_dim,output_dim,5,5)
                    elif self.name=='tag':
                        layer=TAGConv(input_dim+last_dim,output_dim,5)
                    elif self.name=='rel':
                        layer=RelGraphConv(input_dim+last_dim,output_dim,4)
            else:
                if self.name=='gatv2':
                    layer=GATv2Conv(input_dim,output_dim,1)
                elif self.name=='gat':
                    layer=GATConv(input_dim,output_dim,1)
                elif self.name=='sage':
                    layer=SAGEConv(input_dim,output_dim,'pool')
                elif self.name=='sgc':
                    layer=SGConv(input_dim,output_dim,4)
                elif self.name=='gin':
                    layer=GINConv(nn.Linear(input_dim,output_dim),'sum',learn_eps=True)
                elif self.name=='dotgat':
                        layer=DotGatConv(input_dim,output_dim,1)
                elif self.name=='cheb':
                        layer=ChebConv(input_dim,output_dim,2)
                elif self.name=='gated':
                        layer=GatedGraphConv(input_dim,output_dim,5,5)
                elif self.name=='tag':
                        layer=TAGConv(input_dim,output_dim,5)
                elif self.name=='rel':
                        layer=RelGraphConv(input_dim,output_dim,4)
                elif self.name=='gcnii':
                        layer=GCN2Conv(input_dim,i+1)
            self.layers.append(layer)

    def forward(self,data):
        data=self.norm(data)
        out=F.tanh(data.ndata['x'])
        res_out=[out]
        out=self.appnp(data,out)
        try:
            lambda_max=dgl.laplacian_lambda_max(data)
        except ArpackError or TypeError:
            lambda_max=[2.0]
        except TypeError:
            lambda_max=[2.0]
        if lambda_max[0]==0:
            lambda_max=[2.0]
        if self.residual:
            for index in range(len(self.layers)):
                layer=self.layers[index]
                try:
                    if self.name=='cheb':
                        out=layer(data,out,lambda_max).squeeze(1)
                    elif self.name=='gcnii':
                        out=layer(data,out,out).squeeze(1)
                    else:
                        out=layer(data,out).squeeze(1)
                except KeyError:
                    if self.name=='cheb':
                        out=layer(data,out,lambda_max).squeeze(1)
                    elif self.name=='gcnii':
                        out=layer(data,out,out).squeeze(1)
                    else:
                        out=layer(data,out).squeeze(1)
                res_out.append(out)
                if index!=(len(self.layers)-1) :
                    out=torch.cat([out,res_out[index]],dim=1)
                if index==(len(self.layers)-1):
                    out=F.mish(out)
        else:
            for index in range(len(self.layers)):
                layer=self.layers[index]
                try:
                    if self.name=='cheb':
                        out=layer(data,out,lambda_max).squeeze(1)
                    elif self.name=='gcnii':
                        out=layer(data,out,out).squeeze(1)
                    else:
                        out=layer(data,out).squeeze(1)
                except KeyError:
                    if self.name=='cheb':
                        out=layer(data,out,lambda_max).squeeze(1)
                    elif self.name=='gcnii':
                        out=layer(data,out,out).squeeze(1)
                    else:
                        out=layer(data,out).squeeze(1)
                if index==(len(self.layers)-1):
                    out=F.mish(out)
        return out


class w2v_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1=GCNLayer('gat',[288,512,256,128],False)
        self.gcn2=GCNLayer('sage',[288,512,256,128],False)
        self.gcn3=GCNLayer('cheb',[288,512,128,96],False)
        self.gcn4=GCNLayer('sgc',[288,512,128,64],False)
        gcn_dim=128+128+96+64
        self.attn=nn.MultiheadAttention(embed_dim=256*5+1056*1,num_heads=1,dropout=0)
        self.dnn= nn.Sequential(
            nn.Linear(gcn_dim+288,512),
            nn.Dropout(0.05),
            nn.Mish(),
            nn.Linear(512,512-128),
            nn.Dropout(0.05),
            nn.Mish(),
            nn.Linear(512-128,256),
            nn.Dropout(0.05),
            nn.Mish(),
            nn.Linear(256,196),
        )
        self.output= nn.Sequential(
            nn.Linear(196,128),
            nn.Mish(),
            nn.Linear(128,128),
        )
        self.feat = nn.Sequential(
            nn.Linear(gcn_dim+128,512),
            nn.Mish(),
            nn.Linear(512,256),
            nn.Mish(),
            nn.Linear(256,96)
        )
        self.fc=nn.Linear(96,1)
        
    def forward(self,data):
        data=data.add_self_loop()
        out1=self.gcn1(data)
        out2=self.gcn2(data)
        out3=self.gcn3(data)
        out4=self.gcn4(data)
        out=torch.cat([out1,out2,out3,out4,data.ndata['x']],dim=1)
        out=self.dnn(out)
        out=self.output(out)
        out=torch.cat([out1,out2,out3,out4,out],dim=1)
        out=self.feat(out)
        out=self.fc(out)
        return F.sigmoid(out).reshape(-1,1)

class sci_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1=GCNLayer('gat',[1056,768,512,320],False)
        self.gcn2=GCNLayer('sage',[1056,768,512,320],False)
        self.gcn3=GCNLayer('cheb',[1056,768,512,320],False)
        self.gcn4=GCNLayer('sgc',[1056,768,512,256,128],False)
        gcn_dim=320+320+320+128
        self.attn=nn.MultiheadAttention(embed_dim=256*5+1056*1,num_heads=1,dropout=0)
        self.dnn= nn.Sequential(
            nn.Linear(gcn_dim+1056,2048),
            nn.Dropout(0.05),
            nn.Mish(),
            nn.Linear(2048,2048-512),
            nn.Dropout(0.05),
            nn.Mish(),
            nn.Linear(2048-512,1024),
            nn.Dropout(0.05),
            nn.Mish(),
            nn.Linear(1024,768),
            nn.ReLU(),
            nn.Linear(768,512),
        )
        self.output= nn.Sequential(
            nn.Linear(512,256),
            nn.Mish(),
            nn.Linear(256,180),
            nn.Mish(),
            nn.Linear(180,1),
        )
        self.output_1 = nn.Sequential(
            nn.Linear(gcn_dim+10,512),
            nn.Linear(gcn_dim+1,512),
            nn.Mish(),
            nn.Linear(512,128),
            nn.Mish(),
            nn.Linear(128,1),
        )

    def forward(self,data):
        data=data.add_self_loop()
        out1=self.gcn1(data)
        out2=self.gcn2(data)
        out3=self.gcn3(data)
        out4=self.gcn4(data)
        out=torch.cat([out1,out2,out3,out4,data.ndata['x']],dim=1)
        out=self.dnn(out)
        out=self.output(out).reshape(-1,1)
        out=torch.cat([out1,out2,out3,out4,out],dim=1)
        out=self.output_1(out).reshape(-1,1)
        return F.sigmoid(out).reshape(-1,1)
