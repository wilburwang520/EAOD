import math

import numpy as np


def cate_od(df):
    n = df.shape[0]
    UO = 0
    outliernum = n
    o = outliernum

    dictionary_as_of = {}
    hathx, OFx = calulate(df)
    for j in range(n):
        if hathx[j] > 0:
            dictionary_as_of[j] = OFx[j]
            UO = UO + 1

    if o > UO:
        o = UO

    dictionary_value = list(dictionary_as_of.values())
    min_values = np.min(dictionary_value)
    max_values = np.max(dictionary_value)
    for key in dictionary_as_of.keys():
        dictionary_as_of[key] = (dictionary_as_of[key] - min_values) / (max_values - min_values)

    sort_d = sorted(dictionary_as_of.items(), key=lambda d: d[1], reverse=True)
    score = np.zeros(n)
    for i in range(o):
        score[sort_d[i][0]] = sort_d[i][1]

    return score


def Wx_Hx(df, p, count):
    d = df.shape[1]
    H = np.zeros(d)
    W = np.zeros(d)
    for i in range(d):
        for j in range(count[i]):
            H[i] = H[i] - p[j][i] * math.log2(p[j][i])
        W[i] = 2 * (1 - 1 / (1 + math.exp(-H[i])))
    return W, H


def diet(x):
    return (x - 1) * math.log2(x - 1) - x * math.log2(x)


def OF_HatHX(df, df_unique, n_array, W, H):
    [n, d] = df.shape
    b = 1 / n
    a = 1 / (n - 1)
    hathx = np.zeros(n)
    OFx = np.zeros(n)

    for j in range(n):

        col_v = df.iloc[j].values

        for i in range(d):
            index = df_unique[i].index(col_v[i])
            hathx[j] = hathx[j] + W[i] * (math.log2(a) - (a / b) * math.log2(b)) - a * W[i] * H[i]
            if n_array[index][i] != 1:
                diet_v = diet(n_array[index][i])
                OFx[j] = OFx[j] + W[i] * diet_v
                hathx[j] = hathx[j] + a * W[i] * diet_v
    return OFx, hathx


def calulate(df):
    [total_num, d] = df.shape

    df_unique = []
    count = []
    column_name = list(df.columns.values)

    for i in range(d):
        df_unique += [[]]
    for i in range(d):
        df_unique[i] = np.unique(df[column_name[i]]).tolist()
        count.append(len(df_unique[i]))

    countmax = max(count)

    n = np.zeros((countmax, d))
    for i in range(d):
        for j in range(count[i]):
            y = df_unique[i][j]
            df_v = df[column_name[i]]
            len_listy = len(df_v.loc[df_v == y])
            n[j][i] = len_listy

    p = n / total_num
    W, H = Wx_Hx(df, p, count)
    OFx, hathx = OF_HatHX(df, df_unique, n, W, H)

    return hathx, OFx
