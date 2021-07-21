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

def get_name_of_ranking(ranking_algorithm):
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
    return name

def adaptive_optimize(oris_0_1, advs_0_1, adv_idxes, feature_ranking, step, target_label, dnn, output_folder,
                      epsilons, ori_label):
    """
    Optimize adversaries
    :param oris_0_1:  a set of original samples, shape = (-1, number of features) (0..1 values)
    :param advs_0_1: a set of adversarial samples corresponding to the original samples, shape = (-1, number of features)  (0..1 values)
    :param adv_idxes: index of origial samples corresponding to adversarial examples on the attacking set (could be None)
    :param feature_ranking:
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

    export_summaryv2(oris_0_255, advs_0_255, adv_idxes, optimized_advs_0_255, target_label, ori_label, output_folder,
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
        # print(f"Creating clone - begin")
        # clone = np.copy(optimized_advs_0_255)
        # print(f"Creating clone - done")
        """
        Find out matrix of adversarial features
        """
        start_priority = step * idx + 1  # must start from 1 (1 is the highest priority)
        end_priority = start_priority + step - 1
        print(f"create_J_instance: priority {start_priority} -> {end_priority}")
        ranking_matrix = create_J_instance(oris_0_255.shape, start_priority, end_priority, max_priority, J)
        if ranking_matrix is None:
            break

        """
        generate optimized adv
        """
        print("reprediction")
        new_optimized_advs_0_255 = optimized_advs_0_255 - optimized_advs_0_255 * ranking_matrix + oris_0_255 * ranking_matrix
        pred = dnn.predict((new_optimized_advs_0_255 / 255).reshape(-1, MNIST_N_ROW, MNIST_N_COL, MNIST_N_CHANNEL))
        labels = np.argmax(pred, axis=1)

        # Revert the restoration for the adv which has failed restoration
        print(f"Reverting the restoration for the failed action")
        count = 0
        for jdx in range(0, len(labels)):
            if labels[jdx] != target_label:
                count += 1
                for kdx in range(0, len(new_optimized_advs_0_255[jdx])):
                    new_optimized_advs_0_255[jdx][kdx] = optimized_advs_0_255[jdx][kdx]  # revert the restoration
        print(f"#fail restorations = {count}/{len(labels)}")
        #
        n_diff_features_after = np.sum(oris_0_255 != new_optimized_advs_0_255, axis=1)
        n_restored_pixels.append(n_diff_features_before - n_diff_features_after)

        optimized_advs_0_255 = new_optimized_advs_0_255

    return optimized_advs_0_255, np.asarray(n_restored_pixels)


def create_J_instance(shape, start_priority, end_priority, max_priority, J):
    if start_priority > max_priority:  # when iterating over all adversarial features
        return None

    if end_priority > max_priority:
        end_priority = max_priority

    ranking_matrix = (J >= start_priority) & (J <= end_priority)
    # ranking_matrix = np.zeros(shape, dtype=int)
    # for jdx in range(0, len(J)):
    #     for kdx in range(0, len(J[jdx])):
    #         if start_priority <= J[jdx][kdx] <= end_priority:
    #             # if the importance of a feature is in the valid priority
    #             ranking_matrix[jdx][kdx] = 1
    return ranking_matrix


def export_summaryv2(oris_0_255, advs_0_255, adv_idxes, optimized_advs, target_label, ori_label, output_folder, epsilons):
    L0_before = utilities.compute_l0s(advs_0_255, oris_0_255, n_features=MNIST_N_FEATURES, normalized=True)
    L2_before = utilities.compute_l2s(advs_0_255, oris_0_255, n_features=MNIST_N_FEATURES)
    L0_after = utilities.compute_l0s(optimized_advs, oris_0_255, n_features=MNIST_N_FEATURES, normalized=True)
    L2_after = utilities.compute_l2s(optimized_advs, oris_0_255, n_features=MNIST_N_FEATURES)
    with open(get_summary_file(output_folder), mode='a') as f:
        seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        n_adv = len(advs_0_255)
        np.save(f"{output_folder}/optimized_adv.npy", optimized_advs)
        np.save(f"{output_folder}/adv.npy", advs_0_255)
        np.save(f"{output_folder}/origin.npy", oris_0_255)
        for idx in range(0, n_adv):
            if adv_idxes is None:
                adv_index = ''
            else:
                adv_index = adv_idxes[idx]

            if epsilons is None:
                epsilon = ''
            else:
                epsilon = epsilons[idx]

            seed.writerow([epsilon, adv_index, ori_label, target_label, L0_before[idx],
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
        L0_before = utilities.compute_l0(adv_0_255, ori_0_255, normalized=True)

        if L0_before == 0:
            continue

        for jdx in range(0, MNIST_N_FEATURES):
            if jdx < len(n_restored_pixels_final[idx]):
                restored_rate[idx][jdx] = n_restored_pixels_final[idx][jdx] / L0_before
            else:
                restored_rate[idx][jdx] = n_restored_pixels_final[idx][
                                              len(n_restored_pixels_final[idx]) - 1] / L0_before
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
        max_row = 0
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
    return oris_0_255 != advs_0_255
    # I = np.zeros(oris_0_255.shape, dtype=int)
    # for idx in range(0, len(I)):
    #     for jdx in range(0, len(I[idx])):
    #         if oris_0_255[idx][jdx] != advs_0_255[idx][jdx]:
    #             I[idx][jdx] = 1
    # return I


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
            ranking = ranking[::-1] # first element has lowest priority

        elif ranking_algorithm == RANKING_ALGORITHM.JSMA:
            diff_pixel_arr, diff_value_arr = feature_ranker().jsma_ranking_original(advs[idx], oris[idx], None,
                                                                                    target_label, dnn,
                                                                                    diff_pixels=np.arange(0,
                                                                                                          MNIST_N_FEATURES),
                                                                                    num_expected_features=None,
                                                                                    num_classes=MNIST_N_CLASSES)
            ranking = diff_pixel_arr  # first element has lowest priority

        elif ranking_algorithm == RANKING_ALGORITHM.JSMA_KA:
            diff_pixel_arr, diff_value_arr = feature_ranker().jsma_ranking_borderV2(advs[idx], oris[idx], None,
                                                                                    target_label, dnn,
                                                                                    diff_pixels=np.arange(0,
                                                                                                          MNIST_N_FEATURES),
                                                                                    num_expected_features=None,
                                                                                    num_classes=MNIST_N_CLASSES)
            ranking = diff_pixel_arr  # first element has lowest priority


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


