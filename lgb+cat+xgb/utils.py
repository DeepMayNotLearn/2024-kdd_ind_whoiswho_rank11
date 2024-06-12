import numpy as np
from sklearn.metrics import roc_auc_score
#jaro-winkler计算
def jaro_similarity(s1, s2):
    if len(s1) == 0 and len(s2) == 0:
        return 1.0

    match_distance = (max(len(s1), len(s2)) // 2) - 1

    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)

    # Count matches
    matches = 0
    for i in range(len(s1)):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(s2))

        for j in range(start, end):
            if s1[i] == s2[j] and not s2_matches[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    # Count transpositions
    transpositions = 0
    k = 0
    for i in range(len(s1)):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    transpositions //= 2

    # Calculate Jaro distance
    jaro = ((matches / len(s1)) + (matches / len(s2)) + ((matches - transpositions) / matches)) / 3
    return jaro

def jaro_winkler_similarity(s1, s2, scaling=0.1):
    jaro = jaro_similarity(s1, s2)

    # Calculate prefix scale
    prefix_length = 0
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix_length += 1
        else:
            break

    # Calculate Jaro-Winkler distance
    jaro_winkler = jaro + (prefix_length * scaling * (1 - jaro))
    return jaro_winkler

#将汉语名字转为拼音字符串
from pypinyin import pinyin, Style
def tras_pinyin(name):
    pinyin_result = pinyin(name, style=Style.NORMAL)
    name_pinyin = ' '.join([item for sublist in pinyin_result for item in sublist])
    return name_pinyin

from collections import Counter
def count_cnt(ls):
    counter = Counter(ls)
    # 获取出现次数最多的前5个元素及其计数
    most_common_five = counter.most_common(5)
    # 提取最常见的5个元素
    most_common_elements = [item[0] for item in most_common_five]
    return most_common_elements

def weight_roc_auc(y_true=0, y_pre=0,train_authors_len=0):
    # 计算每个作者的AUC值
    s = 0
    aucs = []
    for i in train_authors_len:
        s += i
        aucs.append(roc_auc_score(np.array(y_true[s - i:s]), np.array(y_pre[s - i:s])))
    # aucs
    # 计算每个作者的错误数量
    s = 0
    errs = []
    for i in train_authors_len:
        s += i
        cnt = 0
        p1 = y_pre[s - i:s]
        p2 = np.array(y_true[s - i:s])
        # print(p2)
        for k in range(len(p1)):
            if p1[k] != p2[k]:
                cnt += 1
        errs.append(cnt)
    # 计算总错误数量
    total_errors = np.sum(errs)
    # 计算每个作者的权重
    weights = errs / total_errors
    # # 计算加权AUC
    weighted_auc = np.sum(aucs * weights)
    return weighted_auc


# # 字符集合得分
# def character_set_score(strings):
#     total_chars = set(''.join(strings))  # 所有字符串中的唯一字符集合
#     total_unique_chars = len(total_chars)  # 不同字符的数量
#     total_chars_count = sum(len(s) for s in strings)  # 所有字符的总数
#
#     if total_chars_count == 0:
#         return 0  # 避免除以零
#
#     # 计算得分：不同字符数除以字符总数
#     score = total_unique_chars / total_chars_count
#     return score