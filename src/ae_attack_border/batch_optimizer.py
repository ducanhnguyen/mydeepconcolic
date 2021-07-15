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
MNIST_N_FEATURES = 784


def create_different_matrix(oris, advs):
    """
    Create (0,1)-matrix
    :param oris:  a set of original samples, shape = (-1, number of features)
    :param advs: a set of adversarial samples corresponding to the original samples, shape = (-1, number of features)
    :return:
    """
    I = np.zeros(oris.shape, dtype=int)
    for idx in range(0, len(I)):
        for jdx in range(0, len(I[idx])):
            if oris[idx][jdx] != advs[idx][jdx]:
                I[idx][jdx] = 1
    return I


def create_ranking_matrix(advs, ranking_algorithm, target_label):
    """
    :param advs: a set of adversarial samples corresponding to the original samples, shape = (-1, number of features)
    :param I:
    :param target_label:
    :return:
    """
    feature_ranking_matrix = []
    for idx in np.arange(0, len(advs)):
        print(f"Ranking the features in the {idx}-th adversarial example")
        adv = advs[idx]

        ranking = feature_ranker().find_important_features_of_a_sample(
            input_image=adv.reshape(MNIST_N_ROW, MNIST_N_COL, MNIST_N_CHANNEL),
            n_rows=MNIST_N_ROW,
            n_cols=MNIST_N_COL,
            n_channels=MNIST_N_CHANNEL,
            n_important_features=None,  # None: rank all features
            algorithm=ranking_algorithm,
            gradient_label=target_label,
            classifier=dnn)
        feature_ranking_matrix.append(ranking)

    return np.asarray(feature_ranking_matrix)


def adaptive_optimize(oris, advs, adv_idxes, ranking_algorithm, step, target_label, dnn, output_folder):
    """

    :param oris:  a set of original samples, shape = (-1, number of features)
    :param advs: a set of adversarial samples corresponding to the original samples, shape = (-1, number of features)
    :param adv_idxes: index of origial samples corresponding to adversarial examples on the attacking set
    :param ranking_algorithm:
    :param step: > 0
    :param target_label:
    :param dnn: a CNN model
    :return:
    """
    n_restored_pixels_final = []
    # Configure out folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    CSV_PATH = output_folder + '/out.csv'
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='w') as f:
            seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            seed.writerow(
                ['epsilon', 'idx', 'true_label', 'pred_label', 'L0 (before)', 'L0 (after)', 'L2 (before)',
                 'L2 (after)'])
            f.close()

    #
    optimized_advs = np.copy(advs)
    feature_ranking_matrix = create_ranking_matrix(advs, ranking_algorithm, target_label)
    while step > 0:
        print("--------------------------------------------")
        print(f"Step = {step}")
        I = create_different_matrix(oris, optimized_advs)

        J = np.zeros(oris.shape, dtype=int)
        for idx in range(0, len(feature_ranking_matrix)):
            item = feature_ranking_matrix[idx]
            for jdx in range(0, len(item)):
                pos = MNIST_N_ROW * item[jdx][0] + item[jdx][1]
                if I[idx][pos] == 1:  # only prioritize adversarial features
                    J[idx][pos] = max(J[idx]) + 1

        optimized_advs, n_restored_pixels = optimize(oris, optimized_advs, step, target_label, dnn, J)
        if len(n_restored_pixels_final) >= 1:
            latest = n_restored_pixels_final[-1]
        else:
            latest = 0
        for item in n_restored_pixels:
            n_restored_pixels_final.append(latest + item)
        step = int(np.round(step / 2))

    # export the restored rate
    n_restored_pixels_final = np.transpose(n_restored_pixels_final)
    restored_rate = np.zeros(shape=(len(oris), MNIST_N_FEATURES))
    for idx in range(0, len(oris)):
        ori = oris[idx]
        adv = advs[idx]
        L0_before = utilities.compute_l0(adv, ori)
        for jdx in range(0, MNIST_N_FEATURES):
            if jdx < len(n_restored_pixels_final[idx]):
                restored_rate[idx][jdx] = n_restored_pixels_final[idx][jdx] / L0_before
            else:
                restored_rate[idx][jdx] = np.max(restored_rate[idx])
    restored_rate = np.average(restored_rate, axis=0)
    with open(f"{output_folder}/restored_rate.csv", mode='w') as f:
        seed = csv.writer(f)
        for value in restored_rate:
            seed.writerow([str(np.round(value, 5))])
        f.close()

    # export the summary
    for idx in range(0, len(oris)):
        seed_idx = adv_idxes[idx]
        print(f"Exporting seed {seed_idx}")
        ori = oris[idx]
        adv = advs[idx]
        optimized_adv = optimized_advs[idx]
        L0_before = utilities.compute_l0(adv, ori)
        L2_before = utilities.compute_l2(adv, ori)
        L0_after = utilities.compute_l0(optimized_adv, ori)
        L2_after = utilities.compute_l2(optimized_adv, ori)
        highlight = utilities.highlight_diff(np.round(adv * 255), np.round(optimized_adv * 255))
        img_path = f"{output_folder}/{seed_idx}.png"

        pred = dnn.predict(optimized_advs.reshape(-1, 28, 28, 1))
        label = np.argmax(pred, axis=1)[0]

        if label == target_label:
            # utilities.show_four_images(x_28_28_first=ori.reshape(28, 28),
            #                            x_28_28_first_title=f"attacking sample",
            #                            x_28_28_second=adv.reshape(28, 28),
            #                            x_28_28_second_title=f"adv\ntarget = {target_label}\nL0(this, ori) = {L0_before}\nL2(this, ori) = {np.round(L2_before, 2)}",
            #                            x_28_28_third=optimized_adv.reshape(28, 28),
            #                            x_28_28_third_title=f"optimized adv\nL0(this, ori) = {L0_after}\nL2(this, ori) = {np.round(L2_after, 2)}\nuse step = {step}",
            #                            x_28_28_fourth=highlight.reshape(28, 28),
            #                            x_28_28_fourth_title="diff(adv, optimized adv)\nwhite means difference",
            #                            display=False,
            #                            path=img_path)

            with open(CSV_PATH, mode='a') as f:
                seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                seed.writerow([epsilon, seed_idx, ORI_LABEL, TARGET_LABEL, L0_before,
                               L0_after,
                               L2_before,
                               L2_after])
                f.close()
        else:
            print(f"Problem with seed {seed_idx} on the attacking set")


def optimize(oris, advs, step, target_label, dnn, J):
    """

    :param oris:  a set of original samples, shape = (-1, number of features)
    :param advs: a set of adversarial samples corresponding to the original samples, shape = (-1, number of features)
    :param step:
    :param target_label:
    :param dnn:
    :return:
    """
    n_restored_pixels = []  # the total of restored pixels of all adversarial examples
    optimized_advs = np.copy(advs)
    max_priority = np.max(J)
    n_diff_features_before = np.sum(np.round(oris * 255) != np.round(optimized_advs * 255), axis=1)
    for idx in range(0, 10000000):
        print(f"\nPerforming {idx}-th restoration for the batch")
        clone = np.copy(optimized_advs)

        # Find out matrix of adversarial features
        start_priority = step * idx + 1  # must start from 1 (highest priority)
        if start_priority > max_priority:  # when iterating over all adversarial features
            break

        end_priority = start_priority + step
        if end_priority > max_priority:
            end_priority = max_priority

        ranking_matrix = np.zeros(oris.shape, dtype=int)
        for jdx in range(0, len(J)):
            for kdx in range(0, len(J[jdx])):
                if start_priority <= J[jdx][kdx] <= end_priority:
                    # if the importance of features in the valid priority
                    ranking_matrix[jdx][kdx] = 1

        # generate optimized adv
        optimized_advs = optimized_advs - optimized_advs * ranking_matrix + oris * ranking_matrix
        pred = dnn.predict(optimized_advs.reshape(-1, MNIST_N_ROW, MNIST_N_COL, MNIST_N_CHANNEL))
        labels = np.argmax(pred, axis=1)

        # Revert the restoration for the adv which has failed restoration
        for jdx in range(0, len(labels)):
            if labels[jdx] != target_label:
                # need to restore
                print(f"The restoration fails. Reverting the restoration for {jdx}-th adversarial example")
                for kdx in range(0, len(optimized_advs[jdx])):
                    optimized_advs[jdx][kdx] = clone[jdx][kdx]
        #
        n_diff_features_after = np.sum(np.round(oris * 255) != np.round(optimized_advs * 255), axis=1)
        n_restored_pixels.append(n_diff_features_before - n_diff_features_after + 1)

    # pred = dnn.predict(optimized_advs.reshape(-1, 28, 28, 1))
    # labels = np.argmax(pred, axis=1)
    # for item in labels:
    #     if item != target_label:
    #         print("Fail")
    return optimized_advs, np.asarray(n_restored_pixels)


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
    EPSILON_ALL = ["0,1"]

    steps = [6]
    strategies = ['S3']
    for (STEP, RANKING_STRATEGY) in zip(steps, strategies):
        # Generate adv
        for epsilon in EPSILON_ALL:
            AE_MODEL_H5 = f"../../result/ae-attack-border/{MODEL}/saliency/autoencoder_models/ae_slience_map_{MODEL}_{ORI_LABEL}_{TARGET_LABEL}weight={epsilon}_1000autoencoder.h5"
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

            # OPTIMIZER
            step = 6
            oris = oris.reshape(-1, 784)
            advs = advs.reshape(-1, 784)
            ranking_algorithm = RANKING_ALGORITHM.COI
            adaptive_optimize(oris, advs, adv_idxes, ranking_algorithm, step, TARGET_LABEL, dnn,
                              output_folder="/Users/ducanhnguyen/Documents/mydeepconcolic/optimization_batch")
