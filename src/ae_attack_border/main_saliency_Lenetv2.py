import os
import keras
from src.ae_attack_border.ae_custom_layer import concate_start_to_end
import tensorflow as tf
import csv as csv
from src.ae_attack_border.ae_reader import get_X_attack, generate_adv_for_single_attack_SALIENCE, \
    generate_adv_for_single_attack_ALL_FEATURE
from src.ae_attack_border.batch_optimizer import create_ranking_matrix, initialize_out_folder, adaptive_optimize
from src.ae_attack_border.data import wrongseeds_AlexNet, wrongseeds_LeNet_v2
from src.utils import utilities
from src.utils.feature_ranker_2d import feature_ranker, RANKING_ALGORITHM
import numpy as np

MNIST_N_ROW = 28
MNIST_N_COL = 28
MNIST_N_CHANNEL = 1
MNIST_N_CLASSES = 10
MNIST_N_FEATURES = 784

if __name__ == '__main__':
    MODEL = 'Lenet_v2'
    ATTACKED_MODEL_H5 = f"../../result/ae-attack-border/model/{MODEL}.h5"
    WRONG_SEEDS = wrongseeds_LeNet_v2
    IS_SALIENCY_ATTACK = True
    N_ATTACKING_SAMPLES = 1000
    N_CLASSES = 10

    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    X_train = X_train / 255
    dnn = keras.models.load_model(filepath=ATTACKED_MODEL_H5, compile=False)

    ORI_LABEL = 9
    TARGET_LABEL = 4  # 2nd label ##########################################################
    EPSILON_ALL = ["0,0", "0,1", "0,2", "0,3", "0,4", "0,5", "0,6", "0,7", "0,8", "0,9", "1,0"]
    # EPSILON_ALL = ["0,1",  "0,3", "0,5", "0,7", "0,9"]

    all_oris = None
    all_advs = None
    all_adv_idxes = None
    epsilons = []
    """
    GENERATE ADVERSARIES
    """
    for epsilon in EPSILON_ALL:
        print(f"Epsilon {epsilon}")
        # AE_MODEL_H5 = f"../../result/ae-attack-border/{MODEL}/saliency/autoencoder_models/ae_slience_map_{MODEL}_{ORI_LABEL}_{TARGET_LABEL}weight={epsilon}_1000autoencoder.h5"
        AE_MODEL_H5 = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/Lenet_v2/saliency/autoencoder_models/ae_slience_map_Lenet_v2_9_4weight={epsilon}_1000autoencoder.h5"
        if not os.path.exists(AE_MODEL_H5):
            continue

        X_attack, selected_seeds = get_X_attack(X_train, y_train, WRONG_SEEDS, ORI_LABEL,
                                                N_ATTACKING_SAMPLES=1000)

        if IS_SALIENCY_ATTACK:
            ae = keras.models.load_model(filepath=AE_MODEL_H5, compile=False,
                                         custom_objects={'concate_start_to_end': concate_start_to_end})
            n_adv, advs, oris, adv_idxes = generate_adv_for_single_attack_SALIENCE(X_attack, selected_seeds,
                                                                                   TARGET_LABEL, ae, dnn)
        else:
            ae = keras.models.load_model(filepath=AE_MODEL_H5, compile=False)
            print("generate_adv_for_single_attack_ALL_FEATURE")
            n_adv, advs, oris, adv_idxes = generate_adv_for_single_attack_ALL_FEATURE(X_attack,
                                                                                      selected_seeds,
                                                                                      TARGET_LABEL, ae, dnn)
        if len(advs) > 0:
            for idx in range(0, len(advs)):
                epsilons.append(epsilon)

            if all_oris is None:
                all_oris = oris
            else:
                all_oris = np.concatenate((all_oris, oris))

            if all_advs is None:
                all_advs = advs
            else:
                all_advs = np.concatenate((all_advs, advs))

            if all_adv_idxes is None:
                all_adv_idxes = adv_idxes
            else:
                all_adv_idxes = np.concatenate((all_adv_idxes, adv_idxes))

    """
    OPTIMIZE
    """
    oris = all_oris.reshape(-1, MNIST_N_FEATURES)
    oris = np.round(oris * 255) / 255
    print(f"oris shape = {oris.shape}")

    advs = all_advs.reshape(-1, MNIST_N_FEATURES)
    advs = np.round(advs * 255) / 255
    adv_idxes = all_adv_idxes.reshape(-1)

    ranking_algorithms = [RANKING_ALGORITHM.COI, RANKING_ALGORITHM.RANDOM, #RANKING_ALGORITHM.JSMA_KA,
                          RANKING_ALGORITHM.JSMA]
    size = None
    steps = [6, 30, 60]
    dict_ranking = dict()
    for step in steps:
        for ranking_algorithm in ranking_algorithms:
            name = None
            if ranking_algorithm == RANKING_ALGORITHM.JSMA_KA:
                name = "JSMA-KA"
            elif ranking_algorithm == RANKING_ALGORITHM.JSMA:
                name = "JSMA"
            elif ranking_algorithm == RANKING_ALGORITHM.COI:
                name = "COI"
            elif ranking_algorithm == RANKING_ALGORITHM.RANDOM:
                name = "RANDOM"
            elif ranking_algorithm == RANKING_ALGORITHM.SEQUENTIAL:
                name = "SEQUENTIAL"

            if name not in dict_ranking:
                dict_ranking[name] = create_ranking_matrix(advs[:size], oris[:size], dnn, ranking_algorithm,
                                                           TARGET_LABEL)

            output_folder = f"/Users/ducanhnguyen/Documents/mydeepconcolic/optimization_batch3/Lenet_v2/ae_saliency_{MODEL}_{ORI_LABEL}to{TARGET_LABEL}_ranking{name}_step{step}"
            initialize_out_folder(output_folder)
            adaptive_optimize(oris[:size], advs[:size], adv_idxes[:size], dict_ranking[name], step, TARGET_LABEL, dnn,
                              output_folder, epsilons, ORI_LABEL)
