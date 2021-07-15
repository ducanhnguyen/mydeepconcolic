import csv as csv
import os
import keras
from src.ae_attack_border.ae_custom_layer import concate_start_to_end
import tensorflow as tf

from src.ae_attack_border.ae_reader import get_X_attack, generate_adv_for_single_attack_SALIENCE, \
    generate_adv_for_single_attack_ALL_FEATURE
from src.ae_attack_border.data import wrongseeds_AlexNet
from src.ae_attack_border.pixel_ranking import rank_pixel_S3
from src.utils import utilities
from src.utils.feature_ranker_2d import feature_ranker, RANKING_ALGORITHM
import numpy as np


def getI(oris, advs):
    I = oris != advs
    J = np.zeros(oris.shape, dtype=int)
    for idx in range(0, len(I)):
        for jdx in range(0, len(I[idx])):
            if I[idx][jdx]:
                J[idx][jdx] = 1
    return J


def create_ranking(oris, advs, I, target_label):
    J = np.zeros(oris.shape, dtype=int)
    for idx in np.arange(0, len(advs)):
        print(f"Ranking {idx}")
        adv = advs[idx]
        important_features_2D = feature_ranker().find_important_features_of_a_sample(input_image=adv.reshape(28, 28, 1),
                                                                                     n_rows=28, n_cols=28, n_channels=1,
                                                                                     n_important_features=None,
                                                                                     algorithm=RANKING_ALGORITHM.COI,
                                                                                     gradient_label=target_label,
                                                                                     classifier=dnn)
        for item in important_features_2D:
            jdx = 28 * item[0] + item[1]
            if I[idx][jdx] == 1:  # only prioritize adversarial features
                J[idx][jdx] = max(J[idx]) + 1
    return J


def adaptive_optimize(oris, advs, step, target_label, dnn):
    optimized_advs = np.copy(advs)
    while (step > 0):
        optimized_advs = optimize(oris, optimized_advs, step=6, target_label=target_label, model=dnn)
        step = int(np.round(step / 2))

    # export
    for idx in range(0, len(oris)):
        print(f"exporting {idx}")
        ori = oris[idx]
        adv = advs[idx]
        optimized_adv = optimized_advs[idx]
        L0_before = utilities.compute_l0(adv, ori)
        L2_before = utilities.compute_l2(adv, ori)
        L0_after = utilities.compute_l0(optimized_adv, ori)
        L2_after = utilities.compute_l2(optimized_adv, ori)
        highlight = utilities.highlight_diff(np.round(adv * 255), np.round(optimized_adv * 255))
        img_path = f"/Users/ducanhnguyen/Documents/mydeepconcolic/optimization_batch/{idx}.png"
        utilities.show_four_images(x_28_28_first=oris[idx].reshape(28, 28),
                                   x_28_28_first_title=f"attacking sample",
                                   x_28_28_second=adv.reshape(28, 28),
                                   x_28_28_second_title=f"adv\ntarget = {TARGET_LABEL}\nL0(this, ori) = {L0_before}\nL2(this, ori) = {np.round(L2_before, 2)}",
                                   x_28_28_third=optimized_adv.reshape(28, 28),
                                   x_28_28_third_title=f"optimized adv\nL0(this, ori) = {L0_after}\nL2(this, ori) = {np.round(L2_after, 2)}\nuse step = {step}",
                                   x_28_28_fourth=highlight.reshape(28, 28),
                                   x_28_28_fourth_title="diff(adv, optimized adv)\nwhite means difference",
                                   display=False,
                                   path=img_path)


def optimize(oris, advs, step, target_label, model):
    oris = oris.reshape(-1, 784)
    advs = advs.reshape(-1, 784)
    I = getI(oris, advs)
    J = create_ranking(oris, advs, I, target_label)

    optimized_advs = np.copy(advs)
    max_priority = np.max(J)
    for idx in range(0, 10000):
        print(f"Performing {idx} modification")
        clone = np.copy(optimized_advs)

        # find adversarial features to restore
        start_priority = step * idx + 1  # must start from 1
        if start_priority > max_priority:
            break

        end_priority = start_priority + step
        if end_priority > max_priority:
            end_priority = max_priority

        J_tmp = np.zeros(oris.shape, dtype=int)
        for jdx in range(0, len(J)):
            for kdx in range(0, len(J[jdx])):
                if start_priority <= J[jdx][kdx] <= end_priority:
                    J_tmp[jdx][kdx] = 1

        # generate optimized adv
        optimized_advs = optimized_advs - optimized_advs * J_tmp + oris * J_tmp
        pred = model.predict(optimized_advs.reshape(-1, 28, 28, 1))
        labels = np.argmax(pred, axis=1)

        for idx in range(0, len(labels)):
            if labels[idx] != target_label:
                # need to restore
                print(f"restoring adv {idx}")
                for jdx in range(0, len(optimized_advs[idx])):
                    optimized_advs[idx][jdx] = clone[idx][jdx]
        #
        # break
    pred = model.predict(optimized_advs.reshape(-1, 28, 28, 1))
    labels = np.argmax(pred, axis=1)
    for item in labels:
        if item != target_label:
            print("Fail")
    return optimized_advs


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
            step = 6
            adaptive_optimize(oris, advs, step, TARGET_LABEL, dnn)
