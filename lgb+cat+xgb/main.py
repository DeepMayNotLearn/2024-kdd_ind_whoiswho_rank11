#导入所用库
import pandas as pd
import numpy as np
import json
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from  lightgbm import LGBMClassifier,log_evaluation,early_stopping
from sklearn.preprocessing import LabelEncoder,StandardScaler,KBinsDiscretizer,OneHotEncoder,PolynomialFeatures
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import re
import random
import string
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from collections import Counter
import warnings
from utils import jaro_winkler_similarity,count_cnt,tras_pinyin,jaro_similarity,weight_roc_auc
from train_feat import extract_train_feat
from test_feat import extract_test_feat
warnings.filterwarnings('ignore')

#设置全局变量和种子
seed = 42
num_folds = 5 #5折交叉验证
TARGET_NAME = 'label'
np.random.seed(seed)
random.seed(seed)

#读取数据
with open("../train_author.json",encoding = 'utf-8') as f:
    train_author=json.load(f)

with open("../ind_test_author_filter_public.json",encoding = 'utf-8') as f:
    test_author=json.load(f)

with open("../ind_test_author_submit.json",encoding = 'utf-8') as f:
    submission=json.load(f)

with open("../pid_to_info_all.json",encoding = 'utf-8') as f:
    pid_to_info=json.load(f)

# 集合特征
keywords_all = []  # 所有的关键字集合
venue_all = []  # 所有的会议集合
authour_org_all = []  # 作者所有的机构集合
year_all = []  # 所有年份集合
for id, person_info in tqdm(train_author.items()):  # id:key,person_info:value(name,id)
    year_ = []
    keywords = []
    venue = []
    authour_org = []
    for text_id in person_info['normal_data']:  # 正样本
        feat = pid_to_info[text_id]  # 特征
        # 作者论文里所有的关键字集合
        keywords += feat['keywords']
        # 作者论文里所有的venue集合
        venue.append(feat['venue'])
        year_.append(feat['year'])

        if len(feat['authors']) != 0:
            authour_list = []  # 共同作者列表
            for index, person in enumerate(feat['authors']):
                # 对名字进行判断，为汉语转换为英语
                for s in person['name']:
                    if '\u4e00' <= s <= '\u9fff':
                        person['name'] = tras_pinyin(person['name'])
                        break
                authour_list.append(person['name'])

            s = -1
            for i in range(len(authour_list)):
                if jaro_winkler_similarity(authour_list[i], person_info['name']) > s:
                    s = jaro_winkler_similarity(authour_list[i].lower(), person_info['name'])
                    authour_org_ = feat['authors'][i]['org']  # z作者机构
            authour_org.append(authour_org_)

    for text_id in person_info['outliers']:  # 负样本
        feat = pid_to_info[text_id]  # 特征
        # 作者论文里所有的关键字集合
        keywords += feat['keywords']
        # 作者论文里所有的venue集合
        venue.append(feat['venue'])
        year_.append(feat['year'])

        if len(feat['authors']) != 0:
            authour_list = []  # 共同作者列表
            for index, person in enumerate(feat['authors']):
                # 对名字进行判断，为汉语转换为英语
                for s in person['name']:
                    if '\u4e00' <= s <= '\u9fff':
                        person['name'] = tras_pinyin(person['name'])
                        break
                authour_list.append(person['name'])

            s = -1
            for i in range(len(authour_list)):
                if jaro_winkler_similarity(authour_list[i], person_info['name']) > s:
                    s = jaro_winkler_similarity(authour_list[i].lower(), person_info['name'])
                    authour_org_ = feat['authors'][i]['org']  # z作者机构
            authour_org.append(authour_org_)

    year_all.append(count_cnt(year_))
    venue_all.append(count_cnt(venue))
    keywords_all.append(list(set(keywords)))
    authour_org_all.append(count_cnt(authour_org))

# 集合特征
test_keywords_all = []  # 所有的关键字集合
test_venue_all = []  # 所有的会议集合
test_authour_org_all = []  # 作者所有的机构集合
test_year_all = []  # 所有年份集合
for id, person_info in tqdm(test_author.items()):  # id:key,person_info:value(name,id)
    year_ = []
    keywords = []
    venue = []
    authour_org = []
    for text_id in person_info['papers']:  # 正样本
        feat = pid_to_info[text_id]  # 特征
        # 作者论文里所有的关键字集合
        keywords += feat['keywords']
        # 作者论文里所有的venue集合
        venue.append(feat['venue'])
        year_.append(feat['year'])

        if len(feat['authors']) != 0:
            authour_list = []  # 共同作者列表
            for index, person in enumerate(feat['authors']):
                # 对名字进行判断，为汉语转换为英语
                for s in person['name']:
                    if '\u4e00' <= s <= '\u9fff':
                        person['name'] = tras_pinyin(person['name'])
                        break
                authour_list.append(person['name'])

            s = -1
            for i in range(len(authour_list)):
                if jaro_winkler_similarity(authour_list[i], person_info['name']) > s:
                    s = jaro_winkler_similarity(authour_list[i], person_info['name'])
                    authour_org_ = feat['authors'][i]['org']  # z作者机构
            authour_org.append(authour_org_)

    test_year_all.append(count_cnt(year_))
    test_venue_all.append(count_cnt(venue))
    test_keywords_all.append(list(set(keywords)))
    test_authour_org_all.append(count_cnt(authour_org))

train_feats,train_authors_len= extract_train_feat(train_author,pid_to_info,venue_all,year_all,authour_org_all)
test_feats= extract_test_feat(test_author,pid_to_info,test_venue_all,test_year_all,test_authour_org_all)

for mode in ['mean', 'max', 'min', 'var', 'std', 'median']:
    for col in ['len_title', 'title_cnt', 'len_abstract', 'abstract_cnt', 'len_keywords', 'keywords_cnt', 'len_venue',
                'venue_cnt',
                'len_title_abstract', 'title_abstract_cnt', 'len_title_keywords', 'title_keywords_cnt',
                'len_keywords_abstract', 'keywords_abstract_cnt',
                'len_all', 'all_cnt',
                'year', 'keywords_abstract_ratio',
                'co_org_ratio', 'co_org_cnt', 'not_co_org_ratio', 'not_co_org_cnt', 'org_dif_cnt',
                'co_author_cnt', 'author_org_len', 'author_org_cnt', 'all_org_len', 'all_name_len', 'all_org_cnt',
                'all_name_cnt',
                'country_dif_cnt', 'country_dif_ratio', 'sum_country_cnt']:
        train_feats[f'{col}_{mode}'] = train_feats.groupby('author_name')[col].transform(mode)
        test_feats[f'{col}_{mode}'] = test_feats.groupby('author_name')[col].transform(mode)

for col in ['len_title', 'title_cnt', 'len_abstract', 'abstract_cnt', 'len_keywords', 'keywords_cnt', 'len_venue',
            'venue_cnt',
            'len_title_abstract', 'title_abstract_cnt', 'len_title_keywords', 'title_keywords_cnt',
            'len_keywords_abstract', 'keywords_abstract_cnt',
            'len_all', 'all_cnt',
            'co_org_ratio', 'co_org_cnt', 'not_co_org_ratio', 'not_co_org_cnt', 'org_dif_cnt',
            'year', 'keywords_abstract_ratio',
            'co_author_cnt', 'author_org_len', 'author_org_cnt', 'all_org_len', 'all_name_len', 'all_org_cnt',
            'all_name_cnt',
            'country_dif_cnt', 'country_dif_ratio', 'sum_country_cnt']:
    train_feats[f'{col}_max+min'] = train_feats[f'{col}_max'] + train_feats[f'{col}_min']
    test_feats[f'{col}_max+min'] = test_feats[f'{col}_max'] + test_feats[f'{col}_min']
    train_feats[f'{col}_max-min'] = train_feats[f'{col}_max'] - train_feats[f'{col}_min']
    test_feats[f'{col}_max-min'] = test_feats[f'{col}_max'] - test_feats[f'{col}_min']

    train_feats[f'{col}_max-median'] = train_feats[f'{col}_max'] - train_feats[f'{col}_median']
    test_feats[f'{col}_max-median'] = test_feats[f'{col}_max'] - test_feats[f'{col}_median']

    train_feats[f'{col}_median-min'] = train_feats[f'{col}_median'] - train_feats[f'{col}_min']
    test_feats[f'{col}_median-min'] = test_feats[f'{col}_median'] - test_feats[f'{col}_min']

# #年份与最大，最小，均值,最大值、最小值均值的差值
# train_feats['year_max_diff'] = train_feats['year']-train_feats['year_max']
# test_feats['year_max_diff'] = test_feats['year']-test_feats['year_max']

# train_feats['year_min_diff'] = train_feats['year']-train_feats['year_min']
# test_feats['year_min_diff'] = test_feats['year']-test_feats['year_min']

# train_feats['year_mean_diff'] = train_feats['year']-train_feats['year_mean']
# test_feats['year_mean_diff'] = test_feats['year']-test_feats['year_mean']
# # 该作者在该年发表了多少篇文章
# train_publish_cnt = train_feats.groupby(['author_name', 'year']).size().reset_index(name='publish_cnt')
# train_feats = pd.merge(train_feats, train_publish_cnt, on=['author_name', 'year'])
# test_publish_cnt = test_feats.groupby(['author_name', 'year']).size().reset_index(name='publish_cnt')
# test_feats = pd.merge(test_feats, test_publish_cnt, on=['author_name', 'year'])

poly_list =  ['len_title','title_cnt','len_abstract','abstract_cnt','len_keywords','keywords_cnt','len_venue','venue_cnt',
                       'len_title_abstract','title_abstract_cnt','len_title_keywords','title_keywords_cnt','len_keywords_abstract','keywords_abstract_cnt',
                       'len_all','all_cnt',
                       'year','keywords_abstract_ratio',
                       'co_org_ratio','co_org_cnt','not_co_org_ratio','not_co_org_cnt','org_dif_cnt',
                       'co_author_cnt','author_org_len','author_org_cnt','all_org_len','all_name_len','all_org_cnt','all_name_cnt',
                       'country_dif_cnt','country_dif_ratio','sum_country_cnt']

poly=PolynomialFeatures(degree=2,include_bias=False)
result=poly.fit_transform(train_feats[poly_list])
input_ploly_df = pd.DataFrame(result, columns=poly.get_feature_names_out(input_features=poly_list)).drop(poly_list,axis = 1)
train_feats = pd.concat([train_feats,input_ploly_df],axis = 1)

result=poly.fit_transform(test_feats[poly_list])
input_ploly_df = pd.DataFrame(result, columns=poly.get_feature_names_out(input_features=poly_list)).drop(poly_list,axis = 1)
test_feats = pd.concat([test_feats,input_ploly_df],axis = 1)

train_w2v = pd.read_csv('../train_w2v.csv')
test_w2v = pd.read_csv('../test_w2v.csv')

train_feats = pd.concat([train_feats,train_w2v],axis = 1)
test_feats = pd.concat([test_feats,test_w2v],axis = 1)

import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier

def fit_and_predict(model, train_feats=train_feats, test_feats=test_feats, name=0, choose_cols=0):
    X = train_feats[choose_cols].copy()
    y = train_feats[TARGET_NAME].copy()
    test_X = test_feats[choose_cols].copy()
    oof_pred_pro = np.zeros((len(X), 2))
    test_pred_pro = np.zeros((num_folds, len(test_X), 2))

    groups = train_feats['author_name']
    skf = GroupKFold(n_splits=num_folds)

    for fold, (train_index, valid_index) in (enumerate(skf.split(X, y, groups))):
        print(f"name:{name},fold:{fold}")

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        if name == 'lgb':
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                      callbacks=[log_evaluation(100), early_stopping(100)]
                      )
        if name == 'xgb':
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100,
                      early_stopping_rounds=100
                      )
        if name == 'cat':
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                      early_stopping_rounds=100
                      )
        oof_pred_pro[valid_index] = model.predict_proba(X_valid)
        # 将数据分批次进行预测.
        test_pred_pro[fold] = model.predict_proba(test_X)
    print(f"weight_roc_auc:{weight_roc_auc(y_true=y.values, y_pre=oof_pred_pro[:, 1],train_authors_len=train_authors_len)}")
    return oof_pred_pro, test_pred_pro

lgb_params={
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 32,
    "learning_rate": 0.05,
    "n_estimators":5000,
    "colsample_bytree": 0.9,
    "colsample_bynode": 0.9,
    "scale_pos_weight": 0.2,
    "random_state": seed,
    "reg_alpha": 0.3,
    "reg_lambda": 10,
    "extra_trees":True,
    'num_leaves':64,
    "verbose": -1,
    "max_bin":255,
    }

not_selec = ['author_name']
cols=[col for col in test_feats.columns if col not in not_selec]
cols = list(set(cols))
print(len(cols))

lgb_oof_pred_pro,lgb_test_pred_pro=fit_and_predict(model= LGBMClassifier(**lgb_params),
                                            train_feats=train_feats,test_feats=test_feats,
                                            name='lgb',choose_cols=cols)
xgb_params = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "max_depth": 12,
    "learning_rate": 0.05,
    "n_estimators": 5000,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "seed": seed,
    'tree_method':'gpu_hist',
    "reg_alpha": 0.3,
    "reg_lambda": 10,
    "num_leaves": 64,
    "max_bin": 255,
    "scale_pos_weight": 0.2,
    "verbose": 100,
    "eval_metric": "auc"
}

xgb_oof_pred_pro,xgb_test_pred_pro=fit_and_predict(model=xgb.XGBClassifier(**xgb_params),
                                            train_feats=train_feats,test_feats=test_feats,
                                            name='xgb',choose_cols=cols)

import catboost as cat
cat_params = {
    'loss_function': 'Logloss',  # 适用于二分类问题的逻辑回归
    "max_depth": 8,
    "learning_rate": 0.03,
    "n_estimators": 5000,
    "random_seed":seed,
    "max_bin": 255,
    'task_type': 'GPU',
    "verbose": 100,  # 不打印训练信息
    "eval_metric": "AUC"  # 设置评价指标为 AUC
}
cat_oof_pred_pro,cat_test_pred_pro=fit_and_predict(model=cat.CatBoostClassifier(**cat_params),
                                            train_feats=train_feats,test_feats=test_feats,name='cat',choose_cols=cols)

#stacking
stack_train = pd.DataFrame({'lgb':lgb_oof_pred_pro[:,1],
                           'xgb':xgb_oof_pred_pro[:,1],
                            'cat':cat_oof_pred_pro[:,1],
                          }
                          )
stack_valid = pd.DataFrame({'lgb':lgb_test_pred_pro.mean(axis=0)[:,1],
                           'xgb':xgb_test_pred_pro.mean(axis=0)[:,1],
                            'cat':cat_test_pred_pro.mean(axis=0)[:,1]
                           })

model = LGBMClassifier(**lgb_params)
X=stack_train[['lgb','xgb','cat']].copy()
y=train_feats[TARGET_NAME].copy()
test_X=stack_valid[['lgb','xgb','cat']].copy()

groups = train_feats['author_name']
#5折交叉验证
skf = GroupKFold(n_splits=num_folds)
lgb_stacking_oof_pred_pro=np.zeros((len(X),2))
lgb_stacking_test_pred =np.zeros((num_folds,len(test_X),2))

for fold, (train_index, valid_index) in (enumerate(skf.split(X, y,groups))):
        print(f"name:{'stacking'},fold:{fold}")
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model.fit(X_train,y_train,eval_set = [(X_valid,y_valid)],callbacks = [log_evaluation(300),early_stopping(300)])
        lgb_stacking_oof_pred_pro[valid_index]=model.predict_proba(X_valid)
        lgb_stacking_test_pred[fold]=model.predict_proba(test_X)
print(f"weight_roc_auc:{weight_roc_auc(y_true = y.values,y_pre = lgb_stacking_oof_pred_pro[:,1])}")

stack_train = pd.DataFrame({'lgb':lgb_oof_pred_pro[:,1],
                           'xgb':xgb_oof_pred_pro[:,1],
                            'cat':cat_oof_pred_pro[:,1],
                          }
                          )
stack_valid = pd.DataFrame({'lgb':lgb_test_pred_pro.mean(axis=0)[:,1],
                           'xgb':xgb_test_pred_pro.mean(axis=0)[:,1],
                            'cat':cat_test_pred_pro.mean(axis=0)[:,1]
                           })

model = cat.CatBoostClassifier(**cat_params)
X=stack_train[['lgb','xgb','cat']].copy()
y=train_feats[TARGET_NAME].copy()
test_X=stack_valid[['lgb','xgb','cat']].copy()

groups = train_feats['author_name']
#5折交叉验证
skf = GroupKFold(n_splits=num_folds)
cat_stacking_oof_pred_pro=np.zeros((len(X),2))
cat_stacking_test_pred =np.zeros((num_folds,len(test_X),2))

for fold, (train_index, valid_index) in (enumerate(skf.split(X, y,groups))):
        print(f"name:{'stacking'},fold:{fold}")
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],verbose=100,
                  early_stopping_rounds=100
                 )
        cat_stacking_oof_pred_pro[valid_index]=model.predict_proba(X_valid)
        cat_stacking_test_pred[fold]=model.predict_proba(test_X)
print(f"weight_roc_auc:{weight_roc_auc(y_true = y.values,y_pre = cat_stacking_oof_pred_pro[:,1])}")

lgb_stacking_test_pred_ =  lgb_stacking_test_pred.mean(axis = 0)[:,1]
cat_stacking_test_pred_ =  cat_stacking_test_pred.mean(axis = 0)[:,1]

cnt=0
for id,names in submission.items():
    for name in names:
        submission[id][name]=(lgb_stacking_test_pred_[cnt]+cat_stacking_test_pred_[cnt])/2
        cnt+=1
with open('lgb_cat.json', 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii=False, indent=4)
