import keras
import numpy as np
import os
from src.ae_attack_border.batch_optimizer import adaptive_optimize, create_ranking_matrix, get_name_of_ranking
from src.utils import utilities
from src.utils.feature_ranker_2d import RANKING_ALGORITHM

if __name__ == '__main__':
    size = 20
    TARGET_LABEL = 7
    ORIGIN_LABEL = 9
    STEP = 2
    ranking = RANKING_ALGORITHM.COI
    ATTACKED_MODEL_H5 = f"../../result/ae-attack-border/model/Alexnet.h5"

    dnn = keras.models.load_model(filepath=ATTACKED_MODEL_H5, compile=False)

    oris = np.load(
        "/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/CW/ori_alexnet.npy")
    oris = oris.reshape(-1, 784)
    # oris = np.tanh(oris) * 0.5 + 0.5
    oris = np.round(oris*255)/255
    print(f"{np.min(oris)} {np.max(oris)}")
    pred = np.argmax(dnn.predict(oris.reshape(-1, 28, 28, 1)), axis=1)
    print(f"#fail = {np.sum(pred != ORIGIN_LABEL)}")

    advs = np.load(
        "/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/CW/adv_alexnet.npy")
    advs = advs.reshape(-1, 784)
    advs = advs + 0.5
    advs = np.round(advs * 255) / 255
    print(f"{np.min(advs)} {np.max(advs)}")
    pred = np.argmax(dnn.predict(advs.reshape(-1, 28, 28, 1)), axis=1)
    print(f"#fail = {np.sum(pred != TARGET_LABEL)}")

    utilities.show_two_images(oris[0].reshape(28, 28), advs[0].reshape(28, 28), display=True)
    # future_ranking = create_ranking_matrix(advs[:size], oris[:size], dnn, ranking, target_label=TARGET_LABEL)
    # output_folder = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/CW/Alexnet/Alexnet_border_{get_name_of_ranking(ranking)}_step{STEP}"
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    # adaptive_optimize(oris_0_1=oris[:size], advs_0_1=advs[:size], adv_idxes=None, feature_ranking=future_ranking,
    #                   step=6, target_label=TARGET_LABEL,
    #                   dnn=dnn,
    #                   output_folder=output_folder, epsilons=None, ori_label=ORIGIN_LABEL)
