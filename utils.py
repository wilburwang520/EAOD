import json

import numpy as np
from sklearn import metrics
import time


def load_rules():
    rules_path = './expert_rules.json'
    rules = {}
    with open(rules_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        rules = json_data
        fp.close()
    return rules


def load_data_test(data, rules, col):
    n = len(data)
    score = np.zeros((n, 1))
    mylabel = np.zeros((n, 1))
    id = []
    data_values = data
    s_time = time.time()
    for i in range(n):
        line = data_values[i]
        rules_values = rules.values()
        for rule in rules_values:
            count = 0
            for index in range(len(line)):
                if col[index] in rule:
                    if col[index].startswith("B"):
                        item = rule[col[index]]
                        if 'and' in item:
                            item = item.split(' ')
                            string = str(line[index]) + item[0] + ' and ' + str(line[index]) + item[2] + ' and ' + \
                                     str(line[index]) + item[4] + ' and ' + str(line[index]) + item[6]
                        elif ' ' in item:
                            item = item.split(' ')
                            string = item[0] + str(line[index]) + item[1]
                        else:
                            string = str(line[index]) + item
                    else:
                        item = rule[col[index]]
                        if 'and' in item:
                            item = item.split(' ')
                            string = str(int(line[index])) + item[0] + ' and ' + str(int(line[index])) + item[
                                2] + ' and ' + \
                                     str(int(line[index])) + item[4] + ' and ' + str(int(line[index])) + item[6]
                        elif ' ' in item:
                            item = item.split(' ')
                            string = item[0] + str(int(line[index])) + item[1]
                        else:
                            string = str(int(line[index])) + item

                    if eval(string):
                        count += 1
                    else:
                        break
            if count == len(rule):
                score[i] = 0.99
                mylabel[i] = 1
                id.append(i)

                break
    e_time = time.time()
    print('match rules time:', e_time - s_time)
    return score, mylabel, id


def get_data(scores, rate, is_normal):
    all_size = scores.shape[0]
    if is_normal:
        sorted_index = get_sorted_index(scores, order="descending")
    else:
        sorted_index = get_sorted_index(scores, order="ascending")
    size = int(all_size * rate)
    index_list = sorted_index[all_size - size: all_size]
    return index_list


def get_sorted_index(score, order="descending"):
    '''
    :param score:
    :param order:
    :return: index of sorted item in descending order
    e.g. [8,3,4,9] return [3,0,2,1]
    '''
    score_map = []
    size = len(score)
    for i in range(size):
        score_map.append({'index': i, 'score': score[i]})
    if order == "descending":
        reverse = True
    elif order == "ascending":
        reverse = False
    score_map.sort(key=lambda x: x['score'], reverse=reverse)
    keys = [x['index'] for x in score_map]
    return keys


# @nb.njit()
def get_rank(score):
    '''
    :param score:
    :return:
    e.g. input: [0.8, 0.4, 0.6] return [0, 2, 1]
    '''
    sort = np.argsort(score.reshape((1, len(score))))[0]
    size = score.shape[0]
    rank = np.zeros(size)
    for i in range(size):
        rank[sort[i]] = size - i - 1

    return rank


def evaluation(score, y_true):
    auc_roc = metrics.roc_auc_score(y_true, score)
    precision, recall, _ = metrics.precision_recall_curve(y_true, score)
    auc_pr = metrics.auc(recall, precision)
    return auc_roc, auc_pr
