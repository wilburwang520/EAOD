import numpy as np

from NumeDate import AE_Data


def init_nume_od(nume_data):
    mean = np.nanmean(nume_data, axis=0)
    std = np.std(nume_data, axis=0)
    scores = np.zeros(nume_data.shape[0])
    for jj, obj in enumerate(nume_data):
        scores[jj] = np.nanmean(abs(obj - mean) / std)
    return scores


def ae_od(normal_data, all_data, ae_model,is_normal, EPISODE_MAX=10000, BATCH_SIZE=64, verbose=False):
    [n_obj, n_f] = all_data.shape
    train_data = AE_Data(normal_data)
    loss_list = np.zeros(200)
    for episode in range(EPISODE_MAX):
        batch_x = train_data.next_batch(BATCH_SIZE)
        train_loss = ae_model.train_model(batch_x)
        loss_list[episode % 200] = train_loss
        avg = 0.
        std = 0.

        if episode % 200 == 0 and episode // 200 != 0:
            std = np.std(loss_list)
            avg = np.average(loss_list)
            if std < 0.05*avg or avg < 1e-5:
                if verbose:
                    print('  DeepAE:{}, episode: {}, loss: {:.4}, avg,std: {:.4}, {:.4}'.
                          format(batch_x.shape, episode, train_loss, avg, std))
                break
            loss_list = np.zeros(200)

        if episode % 2000 == 0 and verbose:
            print('  DeepAE:{}, episode: {}, loss: {:.4}, avg,std: {:.4}, {:.4}'.
                  format(batch_x.shape, episode, train_loss, avg, std))

    nomaly_scores = np.zeros([n_obj])
    for i, obj in enumerate(all_data):
        nomaly_scores[i] = ae_model.test_model(obj.reshape([1, n_f]))


    min_values=np.min(nomaly_scores)
    max_values=np.max(nomaly_scores)
    normalizer_score =  (nomaly_scores-min_values)/(max_values-min_values)

    if is_normal:
        return normalizer_score
    else:
        return 1-normalizer_score
