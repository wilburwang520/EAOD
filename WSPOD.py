from sklearn.preprocessing import MinMaxScaler

import utils
from Cate_OD import cate_od
from DeepAutoEncoder import DeepAutoEncoder
from NumeProcess import *


def ensemble_scores(score1, score2):
    '''
    :param score1:
    :param score2:
    :return: ensemble score
    @@ ensemble two score functions
    we use a non-parameter way to dynamically get the tradeoff between two estimated scores.
    It is much more important if one score function evaluate a object with high outlier socre,
    which should be paid more attention on these scoring results.
    instead of using simple average, median or other statistics
    '''

    objects_num = len(score1)

    [_max, _min] = [np.max(score1), np.min(score1)]
    s1 = (score1 - _min) / (_max - _min)
    [_max, _min] = [np.max(score2), np.min(score2)]
    s2 = (score2 - _min) / (_max - _min)

    rank1 = utils.get_rank(s1)
    rank2 = utils.get_rank(s2)
    n = len(s1)
    combine_score = np.zeros((n))

    alpha_list = (1. / (2 * (objects_num - 1))) * (rank2 - rank1) + 0.5
    for i in range(n):
        combine_score[i] = alpha_list[i] * s1[i] + (1. - alpha_list[i]) * s2[i]
    return combine_score


def fit(data, abnormalid, cate_num, rate=0.1):
    cate_data = data.iloc[:, :cate_num]
    nume_data = data.iloc[:, cate_num:]
    [_, nume_features_num] = nume_data.shape

    scaler = MinMaxScaler()
    nume_data_values = scaler.fit_transform(nume_data.values)

    abnormal_deep_ae = DeepAutoEncoder(input_dimension=nume_features_num)
    normal_deep_ae = DeepAutoEncoder(input_dimension=nume_features_num)
    id = abnormalid
    finialy_scores = np.zeros((data.shape[0]))

    batch_size = 128
    episode_max = 10000
    pre_id = data.index.difference(id)

    pre_cate_data = cate_data.loc[pre_id]
    pre_nume_data = nume_data_values[pre_id]

    abnormal_data = nume_data_values[id]

    cateObjectScore = cate_od(pre_cate_data.astype(int))

    abnormal_scores = ae_od(normal_data=abnormal_data, all_data=pre_nume_data, ae_model=abnormal_deep_ae,
                            is_normal=False, EPISODE_MAX=episode_max, BATCH_SIZE=batch_size, verbose=False)

    init_nume_scores = init_nume_od(pre_nume_data)
    init_scores = ensemble_scores(cateObjectScore, init_nume_scores)
    index_list = utils.get_data(init_scores, rate, is_normal=True)
    normal_data = pre_nume_data[index_list]

    normal_scores = ae_od(normal_data=normal_data, all_data=pre_nume_data, ae_model=normal_deep_ae,
                          is_normal=True, EPISODE_MAX=episode_max, BATCH_SIZE=batch_size, verbose=False)
    ae_scores = abnormal_scores + normal_scores
    scores = ensemble_scores(cateObjectScore, ae_scores)
    finialy_scores[abnormalid] = max(scores) + 1

    pre_id_values = pre_id.values
    for i in range(len(pre_id)):
        finialy_scores[pre_id_values[i]] = scores[i]

    return finialy_scores
