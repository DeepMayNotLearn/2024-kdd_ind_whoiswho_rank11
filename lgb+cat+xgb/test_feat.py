import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import re
import random
import string
from utils import count_cnt,tras_pinyin,jaro_winkler_similarity,jaro_similarity
import warnings
warnings.filterwarnings('ignore')

seed = 42
np.random.seed(seed)
random.seed(seed)

def extract_test_feat(test_author,pid_to_info,test_venue_all,test_year_all,test_authour_org_all):
    test_feats = []
    test_authors = []
    test_authora_len = []
    l = -1
    for id, person_info in tqdm(test_author.items()):  # id:key,person_info:value(name,id)
        cnt = 0
        l += 1
        for text_id in person_info['papers']:  # 正样本
            cnt += 1
            test_authors.append(id)  # 作者id
            feat = pid_to_info[text_id]  # 特征

            # 标题字符长度，词个数
            len_title = 0
            title_cnt = 0
            if feat['title'] is not None and len(feat['title']) != 0:
                len_title, title_cnt = len(feat['title']), len(feat['title'].split(' '))

            # 摘要字符长度，词个数
            len_abstract = 0
            abstract_cnt = 0
            if feat['abstract'] is not None and len(feat['abstract']) != 0:
                len_abstract, abstract_cnt = len(feat['abstract']), len(feat['abstract'].split(' '))

            # 关键词字符长度，词个数
            len_keywords = 0
            keywords_cnt = 0
            if feat['keywords'] is not None and len(feat['keywords']) != 0:
                keywords_cnt = len(feat['keywords'])
                for i in range(len(feat['keywords'])):
                    len_keywords += len(feat['keywords'][i])

            # 期刊字符长度，词个数
            len_venue = 0
            venue_cnt = 0
            if feat['venue'] is not None and len(feat['venue']) != 0:
                venue_cnt = len(feat['venue'].split(' '))
                len_venue = len(feat['venue'])

            # 标题+摘要字符长度，词个数
            len_title_abstract = len_title + len_abstract
            title_abstract_cnt = title_cnt + abstract_cnt

            # 标题+关键字字符长度，词个数
            len_title_keywords = len_keywords + len_keywords
            title_keywords_cnt = keywords_cnt + keywords_cnt

            # 关键字+摘要字符长度，词个数
            len_keywords_abstract = len_keywords + len_abstract
            keywords_abstract_cnt = keywords_cnt + abstract_cnt

            # 标题字符长度+摘要字符长度+关键词字符长度
            len_all = len_title + len_abstract + len_keywords
            # 标题词个数+摘要词个数+关键词词个数
            all_cnt = title_cnt + abstract_cnt + keywords_cnt

            # 发表年份
            # 年份是否合理
            year_is_reanson = 0
            year = -1  # 为空用0表示
            if feat['year'] != '' and feat['year'] is not None:
                year = int(feat['year'])
                if year != 0:
                    year_is_reanson = 1

            # 关键词在摘要中的占比
            keywords_abstract_ratio = 0
            if abstract_cnt != 0:
                keywords_abstract_ratio = keywords_cnt / abstract_cnt

            co_author_cnt = len(feat['authors'])  # 共同作者个数
            org_is_null = 1  # 作者机构为空记特征为1
            co_org_ratio = 0  # 共同作者中与作者机构相同的作者个数占总作者个数比
            co_org_cnt = 0  # 共同作者中与作者机构相同的作者个数
            not_co_org_ratio = 0  # 共同作者中与作者机构不相同的作者个数占总作者个数比
            not_co_org_cnt = 0  # 共同作者中与作者机构不相同的作者个数
            org_dif_cnt = 0  # 不同机构的个数
            author_org = ''  # 作者机构
            author_org_len = 0  # 作者所在机构的字符长度
            author_org_cnt = 0  # 作者所在机构的词个数

            if len(feat['authors']) != 0:
                authour_list = []  # 共同作者列表
                for index, person in enumerate(feat['authors']):
                    if person['name'].lower() == person_info['name']:
                        for s in person['name']:
                            if '\u4e00' <= s <= '\u9fff':
                                person['name'] = tras_pinyin(person['name'])
                                break
                    authour_list.append(person['name'])
                s = -1
                for i in range(len(authour_list)):
                    if jaro_winkler_similarity(authour_list[i], person_info['name']) > s:
                        s = jaro_winkler_similarity(authour_list[i], person_info['name'])
                        authour_org = feat['authors'][i]['org']  # z作者机构

                # if len(feat['authors'])!=0:
                #     for index, person in enumerate(feat['authors']):
                #         if person['name'].lower() == person_info['name']:
                #             author_org = person['org'] #作者机构

                if len(author_org) != 0:  # 作者机构不为空
                    author_org_len = len(author_org)
                    author_org_cnt = len(author_org.split(' '))
                if len(author_org) != 0:  # 作者机构不为空记特征为0
                    org_is_null = 0
                for index, person in enumerate(feat['authors']):
                    if person['org'] == author_org:  # 与作者机构相同
                        co_org_cnt += 1
                    else:
                        org_dif_cnt += 1

                co_org_ratio = co_org_cnt / len(feat['authors'])
                not_co_org_cnt = len(feat['authors']) - co_org_cnt
                not_co_org_ratio = not_co_org_cnt / len(feat['authors'])

            country_name = ['China', 'Japan', 'Canada', 'Germany', 'France', 'Italy', 'Sweden', 'Australia', 'Singapore',
                            'Netherlands',
                            'Korea', 'UK', 'USA']  # 机构所在的国家
            country_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 出现次数
            all_org_len = 0  # 共同作者机构字符总长度
            all_name_len = 0  # 共同作者名字字符总长度
            all_org_cnt = 0  # 共同作者机构词总个数
            all_name_cnt = 0  # 共同作者名字词总个数
            country_dif_cnt = 0  # 共同作者中所属不同国家的机构数
            country_dif_ratio = 0  # 共同作者中所属不同国家的机构数占总的个数比
            for i in range(len(feat['authors'])):
                all_org_len += len(feat['authors'][i]['org'])
                all_name_len += len(feat['authors'][i]['name'])
                all_name_cnt += len(feat['authors'][i]['name'].split(' '))
                if feat['authors'][i]['org'] is not None:
                    all_org_cnt += len(feat['authors'][i]['org'].split(' '))
                text = feat['authors'][i]['org']
                for word in country_name:
                    if word in text:
                        country_cnt[country_name.index(word)] += 1
            country_dif_cnt = len(country_cnt) - country_cnt.count(0)
            if sum(country_cnt) != 0:
                country_dif_ratio = country_dif_cnt / sum(country_cnt)

            # 当前论文里wenue和该作者所有论文出现次数前5的wenue的jaro-winkler max，min，mean,amx-min
            if feat['venue'] is not None and len(feat['venue']) != 0:
                jaro_venue_all = []
                for i in test_venue_all[l]:
                    jaro_venue_all.append(jaro_winkler_similarity(feat['venue'], i))
                if len(jaro_venue_all) != 0:
                    max_jaro_venue = max(jaro_venue_all)
                    min_jaro_venue = min(jaro_venue_all)
                    max_min_jaro_venue = max_jaro_venue - min_jaro_venue
                    mean_jaro_venue = sum(jaro_venue_all) / len(jaro_venue_all)
                else:
                    max_jaro_venue = 0
                    min_jaro_venue = 0
                    mean_jaro_venue = 0
                    max_min_jaro_venue = 0
            else:
                max_jaro_venue = 0
                min_jaro_venue = 0
                mean_jaro_venue = 0
                max_min_jaro_venue = 0

            # 当前论文里year和该作者所有论文出现次数前5的year的jaro-winkler max，min，mean
            if feat['year'] != '' and feat['year'] is not None:
                jaro_year_all = []
                for i in test_year_all[l]:
                    jaro_year_all.append(jaro_winkler_similarity(str(feat['year']), str(i)))
                if len(jaro_year_all) != 0:
                    max_jaro_year = max(jaro_year_all)
                    min_jaro_year = min(jaro_year_all)
                    max_min_jaro_year = max_jaro_year - min_jaro_year
                    mean_jaro_year = sum(jaro_year_all) / len(jaro_year_all)
                else:
                    max_jaro_year = 0
                    min_jaro_year = 0
                    mean_jaro_year = 0
                    max_min_jaro_year = 0
            else:
                max_jaro_year = 0
                min_jaro_year = 0
                mean_jaro_year = 0
                max_min_jaro_year = 0

            # 当前论文里作者机构和该作者所有论文出现次数前5的作者机构的jaro-winkler max，min，mean
            jaro_org_all = []
            for i in test_authour_org_all[l]:
                jaro_org_all.append(jaro_winkler_similarity(authour_org, i))
            if len(jaro_org_all) != 0:
                max_jaro_org = max(jaro_org_all)
                min_jaro_org = min(jaro_org_all)
                max_min_jaro_org = max_jaro_org - min_jaro_org
                mean_jaro_org = sum(jaro_org_all) / len(jaro_org_all)
            else:
                max_jaro_org = 0
                min_jaro_org = 0
                mean_jaro_org = 0
                max_min_jaro_org = 0

            test_feats.append(
                [len_title, title_cnt,  # 标题字符长度，词个数
                 len_abstract, abstract_cnt,  # 摘要字符长度，词个数
                 len_keywords, keywords_cnt,  # 关键词字符长度，词个数
                 len_venue, venue_cnt,  # 期刊字符长度，词个数
                 len_title_abstract, title_abstract_cnt,  # 标题+摘要字符长度，词个数
                 len_title_keywords, title_keywords_cnt,  # 标题+关键字字符长度，词个数
                 len_keywords_abstract, keywords_abstract_cnt,  # 关键字+摘要字符长度，词个数
                 len_all,  # 标题字符长度+摘要字符长度+关键词字符长度
                 all_cnt,  # 标题词个数+摘要词个数+关键词词个数
                 year, year_is_reanson,  # 发表年份  #年份是否合理
                 keywords_abstract_ratio,  # 关键词在摘要中的占比
                 org_is_null,  # 作者机构是否为空
                 co_org_ratio,  # 共同作者中与作者机构相同的作者个数占总作者个数比
                 co_org_cnt,  # 共同作者中与作者机构相同的作者个数
                 not_co_org_ratio,  # 共同作者中与作者机构不相同的作者个数占总作者个数比
                 not_co_org_cnt,  # 共同作者中与作者机构不相同的作者个数
                 org_dif_cnt,  # 不同机构的个数
                 co_author_cnt,  # 共同作者个数
                 author_org_len,  # 作者所在机构的字符长度
                 author_org_cnt,  # 作者所在机构的词个数
                 all_org_len,  # 共同作者机构字符总长度
                 all_name_len,  # 共同作者名字字符总长度
                 all_org_cnt,  # 共同作者机构词总个数
                 all_name_cnt,  # 共同作者名字词总个数
                 country_dif_cnt,  # 共同作者中所属不同国家的机构数
                 country_dif_ratio,  # 共同作者中所属不同国家的机构数占总的个数比
                 sum(country_cnt),  # 共同作者中国家总个数
                 max_jaro_venue,#venue最大jaro
                 min_jaro_venue,#venue最小jaro
                 mean_jaro_venue,#venue平均jaro
                 max_min_jaro_venue,#venue最大减最小jaro

                 max_jaro_year,#year最大jaro
                 min_jaro_year,#year最小jaro
                 mean_jaro_year,#year平均jaro
                 max_min_jaro_year,#year最大减最小jaro

                 max_jaro_org,#org最大jaro
                 min_jaro_org,#org最小jaro
                 mean_jaro_org,#org平均jaro
                 max_min_jaro_org,#org最大减最小jaro
                 ] + country_cnt)  # 作者机构所属国家个数
        test_authora_len.append(cnt)
    test_feats = np.array(test_feats)
    print(f"valid_feats.shape:{test_feats.shape}")
    test_feats = pd.DataFrame(test_feats)
    test_feats.columns = ['len_title', 'title_cnt', 'len_abstract', 'abstract_cnt', 'len_keywords', 'keywords_cnt',
                          'len_venue', 'venue_cnt',
                          'len_title_abstract', 'title_abstract_cnt', 'len_title_keywords', 'title_keywords_cnt',
                          'len_keywords_abstract', 'keywords_abstract_cnt',
                          'len_all', 'all_cnt',
                          'year', 'year_is_reanson', 'keywords_abstract_ratio',
                          'org_is_null', 'co_org_ratio', 'co_org_cnt', 'not_co_org_ratio', 'not_co_org_cnt', 'org_dif_cnt',
                          'co_author_cnt', 'author_org_len', 'author_org_cnt', 'all_org_len', 'all_name_len', 'all_org_cnt',
                          'all_name_cnt',
                          'country_dif_cnt', 'country_dif_ratio', 'sum_country_cnt',
                          'max_jaro_venue', 'min_jaro_venue', 'mean_jaro_venue', 'max_min_jaro_venue',
                          'max_jaro_year', 'min_jaro_year', 'mean_jaro_year', 'max_min_jaro_year',
                          'max_jaro_org', 'min_jaro_org', 'mean_jaro_org', 'max_min_jaro_org'
                          ] + [f'country_{i}' for i in range(len(country_cnt))]
    test_feats['author_name'] = test_authors
    for col in [f'country_{i}' for i in range(len(country_cnt))]:
        test_feats[f'{col}_ratio'] = test_feats[col] /test_feats['sum_country_cnt']
    return test_feats