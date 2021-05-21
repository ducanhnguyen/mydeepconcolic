import random

import numpy as np

from src.utils.feature_ranker_2d import feature_ranker, RANKING_ALGORITHM


def rank_pixel_S2(diff_pixel):  # top-left to bottom-right
    return diff_pixel


def rank_pixel_S1(diff_pixel):  # randomly
    diff_pixel = np.asarray(diff_pixel)
    random.shuffle(diff_pixel)
    return diff_pixel


def rank_pixel_S3(diff_pixel, adv, target_label, dnn):  # COI
    important_features = feature_ranker().find_important_features_of_a_sample(input_image=adv.reshape(28, 28, 1),
                                                                              n_rows=28, n_cols=28, n_channels=1,
                                                                              n_important_features=None,
                                                                              algorithm=RANKING_ALGORITHM.COI,
                                                                              gradient_label=target_label,
                                                                              classifier=dnn)
    f = []
    for idx in range(len(important_features)):
        jdx = important_features[idx][0] * 28 + important_features[idx][1]
        if jdx in diff_pixel:
            f.append(jdx)
    f = np.asarray(f)
    f = f[::-1]  # tinh importance tang dan

    return f
