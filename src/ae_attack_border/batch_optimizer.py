import os
import keras
from src.ae_attack_border.ae_custom_layer import concate_start_to_end
import tensorflow as tf
import csv as csv
from src.ae_attack_border.ae_reader import get_X_attack, generate_adv_for_single_attack_SALIENCE, \
    generate_adv_for_single_attack_ALL_FEATURE
from src.ae_attack_border.data import wrongseeds_AlexNet
from src.utils import utilities
from src.utils.feature_ranker_2d import feature_ranker, RANKING_ALGORITHM
import numpy as np

MNIST_N_ROW = 28
MNIST_N_COL = 28
MNIST_N_CHANNEL = 1
MNIST_N_CLASSES = 10
MNIST_N_FEATURES = 784


def adaptive_optimize(oris_0_1, advs_0_1, adv_idxes, ranking_algorithm, step, target_label, dnn, output_folder,
                      epsilons):
    """
    Optimize adversaries
    :param oris_0_1:  a set of original samples, shape = (-1, number of features) (0..1 values)
    :param advs_0_1: a set of adversarial samples corresponding to the original samples, shape = (-1, number of features)  (0..1 values)
    :param adv_idxes: index of origial samples corresponding to adversarial examples on the attacking set (could be None)
    :param ranking_algorithm:
    :param step: > 0
    :param target_label
    :param dnn: a CNN model
    :param output_folder: the output folder of optimization process
    :return: a set of optimized adversaries
    """
    n_restored_pixels_final = []
    oris_0_255 = np.round(oris_0_1 * 255).astype(int)
    advs_0_255 = np.round(advs_0_1 * 255).astype(int)

    optimized_advs_0_255 = np.copy(advs_0_255)
    # feature_ranking = create_ranking_matrix(advs, oris, dnn, ranking_algorithm, target_label)
    while step > 0:
        print("--------------------------------------------")
        print(f"Step = {step}")

        print("create_different_matrix")
        I = create_different_matrix(oris_0_255, optimized_advs_0_255)

        print("create_matrix_J")
        J = create_matrix_J(oris_0_255, feature_ranking, I)

        print("optimize")
        optimized_advs_0_255, n_restored_pixels = optimize(oris_0_255, optimized_advs_0_255, step, target_label, dnn, J)
        step = int(np.round(step / 2))

        # update the restored rate
        if len(n_restored_pixels_final) >= 1:
            latest = n_restored_pixels_final[-1]
        else:
            latest = 0
        for item in n_restored_pixels:
            n_restored_pixels_final.append(latest + item)

    n_restored_pixels_final = np.transpose(n_restored_pixels_final)  # convert into shape (#samples, #predictions)
    export_restored_rate(n_restored_pixels_final, oris_0_255, advs_0_255, output_folder)

    export_summaryv2(oris_0_255, advs_0_255, adv_idxes, optimized_advs_0_255, target_label, dnn, output_folder,
                     epsilons)


def optimize(oris_0_255, advs_0_255, step, target_label, dnn, J):
    """
    Optimize a set of adversaries concurrently, from the first adversarial feature to the last one
    :param oris_0_255:  a set of original samples, shape = (-1, number of features)
    :param advs_0_255: a set of adversarial samples corresponding to the original samples, shape = (-1, number of features)
    :param step:
    :param target_label:
    :param dnn:
    :return:
    """
    n_diff_features_before = np.sum(oris_0_255 != advs_0_255, axis=1)

    n_restored_pixels = []
    optimized_advs_0_255 = np.copy(advs_0_255)

    max_priority = np.max(J)
    num_iterations = np.math.ceil(max_priority / step)

    for idx in range(0, num_iterations):
        print(f"\n[{idx + 1}/{num_iterations}] Optimize the batch")
        clone = np.copy(optimized_advs_0_255)

        """
        Find out matrix of adversarial features
        """
        start_priority = step * idx + 1  # must start from 1 (1 is the highest priority)
        end_priority = start_priority + step
        ranking_matrix = create_J_instance(oris_0_255.shape, start_priority, end_priority, max_priority, J)
        if ranking_matrix is None:
            break

        """
        generate optimized adv
        """
        optimized_advs_0_255 = optimized_advs_0_255 - optimized_advs_0_255 * ranking_matrix + oris_0_255 * ranking_matrix
        pred = dnn.predict((optimized_advs_0_255 / 255).reshape(-1, MNIST_N_ROW, MNIST_N_COL, MNIST_N_CHANNEL))
        labels = np.argmax(pred, axis=1)

        # Revert the restoration for the adv which has failed restoration
        print(f"Reverting the restoration for the failed action")
        for jdx in range(0, len(labels)):
            if labels[jdx] != target_label:
                # need to restore
                for kdx in range(0, len(optimized_advs_0_255[jdx])):
                    optimized_advs_0_255[jdx][kdx] = clone[jdx][kdx]
        #
        n_diff_features_after = np.sum(oris_0_255 != optimized_advs_0_255, axis=1)
        n_restored_pixels.append(n_diff_features_before - n_diff_features_after)

    return optimized_advs_0_255, np.asarray(n_restored_pixels)


def create_J_instance(shape, start_priority, end_priority, max_priority, J):
    if start_priority > max_priority:  # when iterating over all adversarial features
        return None

    if end_priority > max_priority:
        end_priority = max_priority

    ranking_matrix = np.zeros(shape, dtype=int)
    for jdx in range(0, len(J)):
        for kdx in range(0, len(J[jdx])):
            if start_priority <= J[jdx][kdx] <= end_priority:
                # if the importance of a feature is in the valid priority
                ranking_matrix[jdx][kdx] = 1
    return ranking_matrix


def export_summaryv2(oris_0_255, advs_0_255, adv_idxes, optimized_advs, target_label, dnn, output_folder, epsilons):
    L0_before = utilities.compute_l0s(advs_0_255, oris_0_255, n_features=MNIST_N_FEATURES, normalized=True)
    L2_before = utilities.compute_l2s(advs_0_255, oris_0_255, n_features=MNIST_N_FEATURES)
    L0_after = utilities.compute_l0s(optimized_advs, oris_0_255, n_features=MNIST_N_FEATURES, normalized=True)
    L2_after = utilities.compute_l2s(optimized_advs, oris_0_255, n_features=MNIST_N_FEATURES)
    with open(get_summary_file(output_folder), mode='a') as f:
        seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        n_adv = len(advs_0_255)
        for idx in range(0, n_adv):
            seed.writerow([epsilons[idx], adv_idxes[idx], ORI_LABEL, TARGET_LABEL, L0_before[idx],
                           L0_after[idx],
                           L2_before[idx],
                           L2_after[idx]])
        f.close()


def export_restored_rate(n_restored_pixels_final, oris_0_255, advs_0_255, output_folder):
    restored_rate = np.zeros(shape=(len(oris_0_255), MNIST_N_FEATURES))
    n_samples = len(oris_0_255)
    for idx in range(0, n_samples):
        ori_0_255 = oris_0_255[idx]
        adv_0_255 = advs_0_255[idx]
        L0_before = utilities.compute_l0(adv_0_255, ori_0_255, normalized = True)

        for jdx in range(0, MNIST_N_FEATURES):
            if jdx < len(n_restored_pixels_final[idx]):
                restored_rate[idx][jdx] = n_restored_pixels_final[idx][jdx] / L0_before
            else:
                restored_rate[idx][jdx] = np.max(restored_rate[idx])
    restored_rate = np.average(restored_rate, axis=0)

    # export to file
    with open(get_restored_rate_file(output_folder), mode='w') as f:
        seed = csv.writer(f)
        for value in restored_rate:
            seed.writerow([str(np.round(value, 5))])
        f.close()


def create_matrix_J(oris, feature_ranking, I):
    J = np.zeros(oris.shape, dtype=int)
    for idx in range(0, len(feature_ranking)):
        item = feature_ranking[idx]
        max_row = max(J[idx])
        for jdx in range(0, len(item)):
            if isinstance(item[jdx], np.int64) or isinstance(item[jdx], np.int32):  # the position is an integer
                pos = item[jdx]
            else:  # the position is (row, col, channel)
                pos = MNIST_N_COL * item[jdx][0] + item[jdx][1]  # for 2-D image
            if I[idx][pos] == 1:  # if this position is corresponding to an adversarial feature
                J[idx][pos] = max_row + 1  # assign an integer priority
                max_row += 1
    return J


def create_different_matrix(oris_0_255, advs_0_255):
    """
    Create (0,1)-matrix
    :param oris_0_255:  a set of original samples, shape = (-1, number of features)
    :param advs_0_255: a set of adversarial samples corresponding to the original samples, shape = (-1, number of features)
    :return:
    """
    I = np.zeros(oris_0_255.shape, dtype=int)
    for idx in range(0, len(I)):
        for jdx in range(0, len(I[idx])):
            if oris_0_255[idx][jdx] != advs_0_255[idx][jdx]:
                I[idx][jdx] = 1
    return I


def create_ranking_matrix(advs: np.ndarray, oris: np.ndarray, dnn, ranking_algorithm: RANKING_ALGORITHM,
                          target_label: int):
    """
    Rank features
    :param advs: a set of adversarial samples corresponding to the original samples, shape = (-1, number of features)
    :param ranking_algorithm:
    :param target_label:
    :return: shape (#adversaries, #features, #channels). For black-white adversaries, #channels = 1
    """
    feature_ranking = []
    for idx in np.arange(0, len(advs)):
        print(f"[{idx + 1}/{len(advs)}] Ranking the features")
        ranking = None
        if ranking_algorithm == RANKING_ALGORITHM.COI or ranking_algorithm == RANKING_ALGORITHM.CO or ranking_algorithm == RANKING_ALGORITHM.ABS:
            ranking = feature_ranker().find_important_features_of_a_sample(
                input_image=advs[idx].reshape(MNIST_N_ROW, MNIST_N_COL, MNIST_N_CHANNEL),
                n_rows=MNIST_N_ROW,
                n_cols=MNIST_N_COL,
                n_channels=MNIST_N_CHANNEL,
                n_important_features=None,  # None: rank all features
                algorithm=ranking_algorithm,
                gradient_label=target_label,
                classifier=dnn)

        elif ranking_algorithm == RANKING_ALGORITHM.JSMA:
            diff_pixel_arr, diff_value_arr = feature_ranker().jsma_ranking_original(advs[idx], oris[idx], None,
                                                                                    target_label, dnn,
                                                                                    diff_pixels=np.arange(0,
                                                                                                          MNIST_N_FEATURES),
                                                                                    num_expected_features=None,
                                                                                    num_classes=MNIST_N_CLASSES)
            ranking = diff_pixel_arr[::-1]

        elif ranking_algorithm == RANKING_ALGORITHM.JSMA_KA:
            diff_pixel_arr, diff_value_arr = feature_ranker().jsma_ranking_borderV2(advs[idx], oris[idx], None,
                                                                                    target_label, dnn,
                                                                                    diff_pixels=np.arange(0,
                                                                                                          MNIST_N_FEATURES),
                                                                                    num_expected_features=None,
                                                                                    num_classes=MNIST_N_CLASSES)
            ranking = diff_pixel_arr[::-1]  # the first element is the most important element


        elif ranking_algorithm == RANKING_ALGORITHM.RANDOM:
            ranking = feature_ranker().random_ranking(diff_pixels=np.arange(0, MNIST_N_FEATURES))

        elif ranking_algorithm == RANKING_ALGORITHM.SEQUENTIAL:
            ranking = feature_ranker().sequence_ranking(diff_pixels=np.arange(0, MNIST_N_FEATURES))

        if ranking is not None:
            feature_ranking.append(ranking)
    return np.asarray(feature_ranking)


def initialize_out_folder(output_folder: str):
    """
    Configure out folder
    :param output_folder: the absolute path of output folder
    :return:
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    CSV_PATH = get_summary_file(output_folder)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='w') as f:
            seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            seed.writerow(
                ['epsilon', 'idx', 'true_label', 'pred_label', 'L0 (before)', 'L0 (after)', 'L2 (before)',
                 'L2 (after)'])
            f.close()


def get_summary_file(output_folder: str):
    return f"{output_folder}/out.csv"


def get_restored_rate_file(output_folder: str):
    return f"{output_folder}/restored_rate.csv"


if __name__ == '__main__':
    MODEL = 'Alexnet'
    ATTACKED_MODEL_H5 = f"../../result/ae-attack-border/model/{MODEL}.h5"
    WRONG_SEEDS = wrongseeds_AlexNet
    IS_SALIENCY_ATTACK = True
    N_ATTACKING_SAMPLES = 1000
    N_CLASSES = 10

    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    X_train = X_train / 255

    dnn = keras.models.load_model(filepath=ATTACKED_MODEL_H5, compile=False)

    ORI_LABEL = 9
    TARGET_LABEL = 7  # 2nd label ##########################################################
    EPSILON_ALL = ["0,0", "0,1", "0,2", "0,3", "0,4", "0,5", "0,6", "0,7", "0,8", "0,9", "1,0"]
    # EPSILON_ALL = ["0,1"]

    all_oris = None
    all_advs = None
    all_adv_idxes = None
    epsilons = []
    """
    GENERATE ADVERSARIES
    """
    for epsilon in EPSILON_ALL:
        print(f"Epsilon {epsilon}")
        AE_MODEL_H5 = f"../../result/ae-attack-border/{MODEL}/saliency/autoencoder_models/ae_slience_map_{MODEL}_{ORI_LABEL}_{TARGET_LABEL}weight={epsilon}_1000autoencoder.h5"
        # AE_MODEL_H5 = f"../../result/ae-attack-border/{MODEL}/allfeature/autoencoder_models/ae4dnn_{MODEL}_{ORI_LABEL}_{TARGET_LABEL}weight={epsilon}_1000autoencoder.h5"
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

    ranking_algorithms = [RANKING_ALGORITHM.JSMA_KA, RANKING_ALGORITHM.JSMA, RANKING_ALGORITHM.COI,
                          RANKING_ALGORITHM.RANDOM, RANKING_ALGORITHM.SEQUENTIAL]
    # ranking_algorithms = [RANKING_ALGORITHM.COI]
    size = 500
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

        steps = [6]
        feature_ranking = create_ranking_matrix(advs[:size], oris[:size], dnn, ranking_algorithm, TARGET_LABEL)
        for step in steps:
            output_folder = f"/Users/ducanhnguyen/Documents/mydeepconcolic/optimization_batch/ae_saliency_{MODEL}_{ORI_LABEL}to{TARGET_LABEL}_ranking{name}_step{step}"
            initialize_out_folder(output_folder)
            adaptive_optimize(oris[:size], advs[:size], adv_idxes[:size], ranking_algorithm, step, TARGET_LABEL, dnn,
                              output_folder, epsilons)
