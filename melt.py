import json

res1=json.load(open('GNN/result/graph_res.json','r',encoding='utf-8'))
res2=json.load(open('GNN/result/w2v_res.json','r',encoding='utf-8'))
res3=json.load(open('lgb+cat+lgb/lgb_cat.json','r',encoding='utf-8'))
melt_res={}
for key,user_score in res1.items():
    melt_res_i={}
    res1_paper=res1[key]
    res2_paper=res2[key]
    res3_paper=res3[key]
    for kkey,user_score in user_score.items():
        melt_res_i[kkey]=0.7*(0.3*res2_paper[kkey]+0.7*res1_paper[kkey])+0.3*res3_paper[kkey]
    melt_res[key]=melt_res_i
with open('final_res', 'w') as file:
        # 使用 json.dump() 将列表写入文件
        json.dump(melt_res, file)