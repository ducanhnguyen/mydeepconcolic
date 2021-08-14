import csv as csv
import os
import random

import numpy as np

from src.utils import utilities
from src.utils.feature_ranker_2d import feature_ranker, RANKING_ALGORITHM

N_ROW = 32
N_COL = 32
N_CHANNEL = 3
N_CLASSES = 10
N_FEATURES = N_ROW * N_COL * N_CHANNEL


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


def analyze_speed(restored_rate, threshold, min_num_differences):
    is_improved = True

    if len(restored_rate) >= min_num_differences:
        n_unimprovement = 1
        for i in np.arange(len(restored_rate) - 1, 0, -1):
            delta = restored_rate[i] - restored_rate[i - 1]
            if delta < threshold:
                print(delta)
                n_unimprovement += 1
            else:
                break
        if n_unimprovement >= min_num_differences:
            is_improved = False
    return is_improved


def adaptive_optimize(oris_0_1, advs_0_1, adv_idxes, feature_ranking, step, target_label, dnn, output_folder,
                      epsilons, ori_label, threshold, min_num_differences):
    """
    Optimize adversaries
    :param oris_0_1:  a set of original samples, shape = (-1, row, col, channel) (0..1 values)
    :param advs_0_1: a set of adversarial samples corresponding to the original samples, shape = (-1, row, col, channel)  (0..1 values)
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

    while step > 0:

        print("--------------------------------------------")
        print(f"Step = {step}")

        print("create_matrix_J")
        J = create_matrix_J(oris_0_255, feature_ranking, oris_0_255 != optimized_advs_0_255)

        print("optimize")
        optimized_advs_0_255, n_restored_pixels_final, is_improved = optimize(oris_0_255, optimized_advs_0_255,
                                                                              advs_0_255,
                                                                              step,
                                                                              target_label, dnn, J,
                                                                              n_restored_pixels_final,
                                                                              threshold, min_num_differences)

        # adjust speed
        if not is_improved:
            print("STOP!!!")
            break
        else:
            step = int(np.round(step / 2))
            if step <= 0: #############################################################################################################################
                break

    n_restored_pixels_final = np.transpose(n_restored_pixels_final)  # convert into shape (#samples, #predictions)
    export_restored_rate(n_restored_pixels_final, oris_0_255, advs_0_255, output_folder)
    export_summaryv2(oris_0_255, advs_0_255, adv_idxes, optimized_advs_0_255, target_label, ori_label, output_folder,
                     epsilons)


def optimize(oris_0_255, current_optimized_advs_0_255, advs_0_255, step, target_label, dnn, J, n_restored_pixels_final,
             threshold,
             min_num_differences):
    """
    Optimize a set of adversaries concurrently, from the first adversarial feature to the last one
    :param oris_0_255:  a set of original samples, shape = (-1, number of features)
    :param current_optimized_advs_0_255: a set of adversarial samples corresponding to the original samples, shape = (-1, number of features)
    :param step:
    :param target_label:
    :param dnn:
    :return:
    """
    is_improved = True

    # latest = n_restored_pixels_final[-1] if len(n_restored_pixels_final) >= 1 else 0
    optimized_advs_0_255 = np.copy(current_optimized_advs_0_255)
    max_priority = np.max(J)
    num_iterations = np.math.ceil(max_priority / step)

    for iteration in range(0, num_iterations):
        print(f"\n[{iteration + 1}/{num_iterations}] Optimize the batch")
        n_diff_features_before = np.sum(oris_0_255 != optimized_advs_0_255, axis=(1, 2, 3))
        """
        Find out matrix of adversarial features
        """
        start_priority = step * iteration + 1  # must start from 1 (1 is the highest priority)
        end_priority = start_priority + step - 1
        print(f"create_J_instance: priority {start_priority} -> {end_priority}")
        ranking_matrix = create_J_instance(start_priority, end_priority, max_priority, J)
        if ranking_matrix is None:
            break

        """
        generate optimized adv
        """
        print("reprediction")
        new_optimized_advs_0_255 = optimized_advs_0_255 - optimized_advs_0_255 * ranking_matrix + oris_0_255 * ranking_matrix
        pred = dnn.predict((new_optimized_advs_0_255 / 255).reshape(-1, N_ROW, N_COL, N_CHANNEL))
        labels = np.argmax(pred, axis=1)

        # Revert the restoration for the adv which has failed restoration
        print(f"Reverting the restoration for the failed action")
        fail_sample_idxes = np.asarray(np.where(labels != target_label)).reshape(-1)
        new_optimized_advs_0_255[fail_sample_idxes] = optimized_advs_0_255[fail_sample_idxes]
        print(f"#fail restorations = {len(fail_sample_idxes)}/{len(labels)}")

        #
        optimized_advs_0_255 = new_optimized_advs_0_255
        n_diff_features_after = np.sum(oris_0_255 != optimized_advs_0_255, axis=(1, 2, 3))
        n_restored_pixels_final.append(
            n_diff_features_before - n_diff_features_after)  # shape = (#samples, #predictions)

        # check early stopping
        if len(n_restored_pixels_final) % 5 == 0 or iteration == num_iterations - 1:
            if threshold is not None and min_num_differences is not None:
                print("Check early stopping")
                restored_rate = export_restored_rate(np.transpose(n_restored_pixels_final)
                                                     # convert into shape (#samples, #predictions)
                                                     , oris_0_255, advs_0_255, None)
                print(f"restore rate: {np.round(restored_rate[-10:-1], decimals=3)}")

                is_improved = analyze_speed(restored_rate, threshold, min_num_differences)
                if not is_improved:
                    break
    return optimized_advs_0_255, n_restored_pixels_final, is_improved


def create_J_instance(start_priority, end_priority, max_priority, J):
    if start_priority > max_priority:  # when iterating over all adversarial features
        return None

    if end_priority > max_priority:
        end_priority = max_priority

    ranking_matrix = (J >= start_priority) & (J <= end_priority)
    return ranking_matrix


def export_summaryv2(oris_0_255, advs_0_255, adv_idxes, optimized_advs, target_label, ori_label, output_folder,
                     epsilons):
    L0_before = utilities.compute_l0s(advs_0_255, oris_0_255, n_features=N_FEATURES, normalized=True)
    L2_before = utilities.compute_l2s(advs_0_255, oris_0_255, n_features=N_FEATURES)
    L0_after = utilities.compute_l0s(optimized_advs, oris_0_255, n_features=N_FEATURES, normalized=True)
    L2_after = utilities.compute_l2s(optimized_advs, oris_0_255, n_features=N_FEATURES)
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
    """

    :param n_restored_pixels_final: shape=(# samples, # predictions)
    :param oris_0_255:
    :param advs_0_255:
    :param output_folder:
    :return:
    """
    n_samples = n_restored_pixels_final.shape[0]
    n_prediction = n_restored_pixels_final.shape[1]
    restored_rate = np.zeros(shape=n_restored_pixels_final.shape)

    for idx in range(0, n_samples):
        L0_before = utilities.compute_l0(oris_0_255[idx], advs_0_255[idx], normalized=True)

        tmp = [n_restored_pixels_final[idx][0]]
        for jdx in range(1, n_prediction):
            tmp.append(tmp[jdx - 1] + n_restored_pixels_final[idx][jdx])
        tmp = np.asarray(tmp)
        for jdx in range(0, n_prediction):
            restored_rate[idx][jdx] = (tmp[jdx] / L0_before) if (L0_before != 0) else 0
    restored_rate = np.average(restored_rate, axis=0)

    # export to file
    if output_folder is not None:
        with open(get_restored_rate_file(output_folder), mode='w') as f:
            seed = csv.writer(f)
            for value in restored_rate:
                seed.writerow([str(np.round(value, 5))])
            f.close()
    return restored_rate


def create_matrix_J(oris, feature_ranking, I):
    """
    :param oris: shape: (#samples, #width, #height, #channel)
    :param feature_ranking: shape: (#samples, #width * #height * #channel, #channel)
    :param I: shape: (#samples, #width, #height, #channel)
    :return:
    """
    J = np.zeros(oris.shape, dtype=int)  # shape: (#samples, #width, #height, #channel)
    n_samples = feature_ranking.shape[0]
    for sample_idx in range(0, n_samples):
        current_priority = 0

        n_features = feature_ranking.shape[1]
        for feature_idx in range(0, n_features):
            pixel = feature_ranking[sample_idx][feature_idx]  # value = (row position, col position, channel position)
            row_pos = pixel[0]  # first channel
            col_pos = pixel[1]  # second channel
            channel_pos = pixel[2]  # third channel
            if I[sample_idx][row_pos][col_pos][channel_pos]:
                J[sample_idx][row_pos][col_pos][channel_pos] = current_priority + 1
                current_priority += 1
    return J


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
                input_image=advs[idx].reshape(N_ROW, N_COL, N_CHANNEL),
                n_rows=N_ROW,
                n_cols=N_COL,
                n_channels=N_CHANNEL,
                n_important_features=None,  # None: rank all features
                algorithm=ranking_algorithm,
                gradient_label=target_label,
                classifier=dnn)
            ranking = ranking[::-1]  # first element has lowest priority

        elif ranking_algorithm == RANKING_ALGORITHM.JSMA:
            diff_pixel_arr, diff_value_arr = feature_ranker().jsma_ranking_original(advs[idx], oris[idx], None,
                                                                                    target_label, dnn,
                                                                                    diff_pixels=np.arange(0,
                                                                                                          N_FEATURES),
                                                                                    num_expected_features=None,
                                                                                    num_classes=N_CLASSES)
            ranking = diff_pixel_arr  # first element has lowest priority

        elif ranking_algorithm == RANKING_ALGORITHM.JSMA_KA:
            diff_pixel_arr, diff_value_arr = feature_ranker().jsma_ranking_borderV2(advs[idx], oris[idx], None,
                                                                                    target_label, dnn,
                                                                                    diff_pixels=np.arange(0,
                                                                                                          N_FEATURES),
                                                                                    num_expected_features=None,
                                                                                    num_classes=N_CLASSES)
            ranking = diff_pixel_arr  # first element has lowest priority


        elif ranking_algorithm == RANKING_ALGORITHM.RANDOM:
            ranking = []
            for row in range(0, N_ROW):
                for col in range(0, N_COL):
                    for channel in range(0, N_CHANNEL):
                        ranking.append([row, col, channel])
            ranking = np.asarray(ranking)
            random.shuffle(ranking)


        elif ranking_algorithm == RANKING_ALGORITHM.SEQUENTIAL:
            ranking = feature_ranker().sequence_ranking(diff_pixels=np.arange(0, N_FEATURES))

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
