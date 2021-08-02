import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import WSPOD
from utils import load_data_test, load_rules, evaluation


def main(input_path):
    rounds = 100
    roc_auc_array = np.zeros((rounds, 1))
    pr_auc_array = np.zeros((rounds, 1))
    s_time = time.time()
    df = pd.read_csv(input_path)
    label = df['label']
    df.drop('label', axis=1, inplace=True)
    header = df.head()
    cols = sorted(header)
    cate_num = len([f_name for f_name in cols if f_name.startswith("A")])
    data = df[cols]
    rules = load_rules()
    score, mylabel, id = load_data_test(data.values, rules, cols)

    p, r, f1, _ = precision_recall_fscore_support(label, mylabel, average="binary")

    data.drop(['sport', 'dport'], axis=1, inplace=True)


    for i in range(rounds):
        wspod_score = WSPOD.fit(data, id, cate_num)
        roc_auc, pr_auc = evaluation(wspod_score, label)
        roc_auc_array[i] = roc_auc
        pr_auc_array[i] = pr_auc
    roc_auc = np.mean(roc_auc_array)
    pr_auc = np.mean(pr_auc_array)
    print_text = "{}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}".format(input_path, p, r, f1, roc_auc, pr_auc)
    doc = open('out.csv', 'a')
    print(print_text, file=doc)
    doc.close()
    print("all time:", (time.time() - s_time))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    input_root = "data/IDS17-sum.csv"
    if os.path.isdir(input_root):
        for file_name in os.listdir(input_root):
            if file_name.endswith(".csv"):
                input_path = os.path.join(input_root, file_name)
                main(input_path)
    else:
        input_path = input_root
        main(input_path)
