import csv as csv
import os

import cv2
import keras
import numpy as np
import plotly.express as px
import tensorflow as tf
from numpy import save
from sklearn.decomposition import PCA

from src.ae_attack_border.adv_smoother import smooth_vet_can_step, smooth_vet_can_step_adaptive
from src.ae_attack_border.ae_custom_layer import concate_start_to_end
from src.ae_attack_border.data import wrongseeds_AlexNet, wrongseeds_LeNet, wrongseeds_LeNet_v2
from src.utils import utilities


def get_border(images: np.ndarray) -> np.ndarray:
    border_results = []
    for image in images:
        border_img = (image * 255).astype(np.uint8)
        border_img = np.array(cv2.Canny(border_img, 100, 200)).reshape((28, 28, 1))
        border_results.append(border_img)
    return np.array(border_results, dtype=np.float32) / 255.


def get_internal_images(images: np.ndarray, border_images=None) -> np.ndarray:
    internal_results = []
    if border_images is None:
        border_images = get_border(images)

    for border_image, image in zip(border_images, images):
        border_image_flat = border_image.flatten()
        image_flat = image.flatten()
        border_position = np.where(border_image_flat == 1.)
        internal_result = np.array(image_flat)
        internal_result[border_position] = 0
        internal_result = internal_result.reshape((28, 28, 1))

        internal_results.append(internal_result)

    return np.array(internal_results)


def get_X_attack(X_train, y_train, wrongseeds, ORI_LABEL, N_ATTACKING_SAMPLES, start=None, end=None):
    count = 0
    X_attack = []
    if end is None:
        end = len(X_train)
    if start is None:
        start = 0

    selected_seed = []
    for idx in range(start, end):
        if idx not in wrongseeds and y_train[idx] == ORI_LABEL:
            X_attack.append(X_train[idx])
            selected_seed.append(idx)
            count += 1
            if count == N_ATTACKING_SAMPLES:
                break
    X_attack = np.asarray(X_attack)
    selected_seed = np.asarray(selected_seed)
    # X_attack = X_attack[:N_ATTACKING_SAMPLES]
    # print(f'The shape of X_attack = {X_attack.shape}')
    return X_attack, selected_seed


def generate_adv_for_single_attack_ALL_FEATURE(X_attack, selected_seeds, TARGET_LABEL, ae, dnn):
    candidates = ae.predict(X_attack)

    # compute the number of adv
    candidates = np.asarray(candidates)
    Y_pred = dnn.predict(candidates.reshape(-1, 28, 28, 1))
    y_pred = np.argmax(Y_pred, axis=1)

    compare = y_pred == TARGET_LABEL
    n_adv = np.sum(compare)

    advs = []
    oris = []
    idxes = []
    for idx in range(len(candidates)):
        if compare[idx]:
            advs.append(candidates[idx])
            oris.append(X_attack[idx])
            idxes.append(selected_seeds[idx])
    advs = np.asarray(advs)
    oris = np.asarray(oris)
    idxes = np.asarray(idxes)
    return n_adv, advs, oris, idxes


def generate_adv_for_single_attack_SALIENCE(X_attack, selected_seeds, TARGET_LABEL, ae, dnn):
    '''
        AE just does output final adversarial examples, by using concate_start_to_end in https://github.com/testingforAI-vnuuet/AdvGeneration/blob/2745dbeac09bc6598ea91f885d3b827cf8b7e169/src/attacker/ae_slience_map.py#L185
    :param X_attack:
    :param selected_seeds:
    :param TARGET_LABEL:
    :param ae:
    :param dnn:
    :return:
    '''
    # border_origin_images = get_border(X_attack)
    # internal_origin_images = get_internal_images(X_attack, border_images=border_origin_images)
    # candidate_generated_borders = ae.predict(X_attack) * border_origin_images
    # candidates = np.clip(candidate_generated_borders + internal_origin_images, 0, 1)
    candidates = ae.predict(X_attack)

    # compute the number of adv
    candidates = np.asarray(candidates)
    # print(f'The shape of candidate adv = {candidates.shape}')
    Y_pred = dnn.predict(candidates.reshape(-1, 28, 28, 1))
    y_pred = np.argmax(Y_pred, axis=1)
    # print(f"candidate label = {y_pred}")

    compare = y_pred == TARGET_LABEL
    n_adv = np.sum(compare)
    # print(f"#adv = {n_adv}")

    advs = []
    oris = []
    idxes = []
    for idx in range(len(candidates)):
        if compare[idx]:
            advs.append(candidates[idx])
            oris.append(X_attack[idx])
            idxes.append(selected_seeds[idx])
    advs = np.asarray(advs)
    oris = np.asarray(oris)
    idxes = np.asarray(idxes)
    return n_adv, advs, oris, idxes

def generate_adv_for_single_attack_BORDER_PATTERN(X_attack, selected_seeds, TARGET_LABEL, ae, dnn):
    '''
    AE just does not output final adversarial examples
    :param X_attack:
    :param selected_seeds:
    :param TARGET_LABEL:
    :param ae:
    :param dnn:
    :return:
    '''
    border_origin_images = get_border(X_attack)
    internal_origin_images = get_internal_images(X_attack, border_images=border_origin_images)
    candidate_generated_borders = ae.predict(X_attack) * border_origin_images
    candidates = np.clip(candidate_generated_borders + internal_origin_images, 0, 1)

    # compute the number of adv
    candidates = np.asarray(candidates)
    # print(f'The shape of candidate adv = {candidates.shape}')
    Y_pred = dnn.predict(candidates.reshape(-1, 28, 28, 1))
    y_pred = np.argmax(Y_pred, axis=1)
    # print(f"candidate label = {y_pred}")

    compare = y_pred == TARGET_LABEL
    n_adv = np.sum(compare)
    # print(f"#adv = {n_adv}")

    advs = []
    oris = []
    idxes = []
    for idx in range(len(candidates)):
        if compare[idx]:
            advs.append(candidates[idx])
            oris.append(X_attack[idx])
            idxes.append(selected_seeds[idx])
    advs = np.asarray(advs)
    oris = np.asarray(oris)
    idxes = np.asarray(idxes)
    return n_adv, advs, oris, idxes


def generate_adv_for_all_classes(BASE_PATH, N_ATTACKING_SAMPLES, WRONG_SEEDS, N_CLASSES, dnn, X_train, y_train
                                 , start=None, end=None):
    adv_table = []
    for ori in range(N_CLASSES):
        # get X_attack
        X_attack, selected_seeds = get_X_attack(X_train, y_train, WRONG_SEEDS, ori, N_ATTACKING_SAMPLES=None,
                                                start=start, end=end)

        row = []
        for target in range(N_CLASSES):
            if ori != target:
                AE_MODEL_H5 = f"{BASE_PATH}_{ori}_{target}.h5"
                ae = keras.models.load_model(filepath=AE_MODEL_H5, compile=False)

                n_adv, _, _, idxes = generate_adv_for_single_attack_BORDER_PATTERN(
                    X_attack[:N_ATTACKING_SAMPLES],
                    target,
                    ae,
                    dnn)
                print(f'attacking {ori} -> {target}: n_adv = {n_adv}')
                row.append(n_adv)

                # for generalization
                n_adv2, _, _, idxes = generate_adv_for_single_attack_BORDER_PATTERN(
                    X_attack[N_ATTACKING_SAMPLES:], target, ae,
                    dnn)
                print(f'GENERALIZATION. # ori = {len(X_attack[2000:])} attacking {ori} -> {target}: n_adv = {n_adv2}')
                print("")
            else:
                row.append(0)

        adv_table.append(row)
    return adv_table


if __name__ == '__main__':
    MODEL = 'Alexnet'
    ATTACKED_MODEL_H5 = f"../../result/ae-attack-border/model/{MODEL}.h5"
    WRONG_SEEDS = wrongseeds_AlexNet
    IS_SALIENCY_ATTACK = False
    N_ATTACKING_SAMPLES = 1000
    N_CLASSES = 10

    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    X_train = X_train / 255

    dnn = keras.models.load_model(filepath=ATTACKED_MODEL_H5, compile=False)

    # ALL_ATTACK = False
    # if ALL_ATTACK:
    #     BASE_PATH = "../../result/ae-attack-border/epsilon=0,5/Alexnet/autoencoder_Alexnetborder"
    #     adv_table = generate_adv_for_all_classes(BASE_PATH, N_ATTACKING_SAMPLES, WRONG_SEEDS, N_CLASSES, dnn, X_train,
    #                                              y_train, start=None, end=None)

    SINGLE_ATTACK_MODE = True
    if SINGLE_ATTACK_MODE:
        # Configure constants

        ORI_LABEL = 9

        TARGET_LABEL = 7  # 2nd label ##########################################################

        EPSILON_ALL = ["0,0", "0,1", "0,2", "0,3", "0,4"]
        steps = [6, 6, 6, 1, 1, 1]
        strategies = ['S3', 'S2', 'S1', 'S3', 'S2', 'S1']
        for (STEP, RANKING_STRATEGY) in zip(steps, strategies):
            OUT_PATH = f"../../result/ae-attack-border/{MODEL}/allfeature/tmp_{RANKING_STRATEGY}step{STEP}/"

            # Configure out folder
            if not os.path.exists(OUT_PATH):
                os.mkdir(OUT_PATH)

            CSV_PATH = OUT_PATH + 'out.csv'
            if not os.path.exists(CSV_PATH):
                with open(CSV_PATH, mode='w') as f:
                    seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    seed.writerow(
                        ['epsilon', 'idx', 'true_label', 'pred_label', 'L0 (before)', 'L0 (after)', 'L2 (before)',
                         'L2 (after)'])
                    f.close()

            # Generate adv
            for epsilon in EPSILON_ALL:
                AE_MODEL_H5 = f"../../result/ae-attack-border/{MODEL}/allfeature/autoencoder_models/ae4dnn_{MODEL}_{ORI_LABEL}_{TARGET_LABEL}weight={epsilon}_1000autoencoder.h5"
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
                    # n_adv, advs, oris, adv_idxes = generate_adv_for_single_attack_BORDER_PATTERN(X_attack,
                    #                                                                              selected_seeds,
                    #                                                                              TARGET_LABEL, ae, dnn)
                    n_adv, advs, oris, adv_idxes = generate_adv_for_single_attack_ALL_FEATURE(X_attack,
                                                                                                 selected_seeds,
                                                                                                 TARGET_LABEL, ae, dnn)
                # n_adv, advs, oris = generate_adv_for_single_attack_ALL_PATTERN(X_attack, TARGET_LABEL, ae, dnn)

                print(f"AE_MODEL_H5 = {AE_MODEL_H5}\t\tepsilon = {epsilon}: #adv = {n_adv}")

                for ori, adv, seed_idx in zip(oris, advs, adv_idxes):
                    img_path = f"{OUT_PATH}/epsilon{epsilon}_{ORI_LABEL}to{TARGET_LABEL}_step{STEP}_idx{seed_idx}"
                    png_path = f"{OUT_PATH}/epsilon{epsilon}_{ORI_LABEL}to{TARGET_LABEL}_step{STEP}_idx{seed_idx}_L0.npy"
                    if os.path.exists(png_path):
                        print(f"{png_path} exists. Move to the next attacking samples!")
                        continue

                    print(f"Optimizing seed {seed_idx}")
                    # if seed_idx != 172:
                    #     continue
                    smooth_adv, highlight, L0_after, L0_before, L2_after, L2_before, restored_pixel_by_prediction = \
                        smooth_vet_can_step_adaptive(
                            ori, adv, dnn,
                            TARGET_LABEL,
                            STEP,
                            RANKING_STRATEGY)
                    per_pixel_by_prediction = restored_pixel_by_prediction / L0_before

                    save(png_path
                         , per_pixel_by_prediction)

                    # utilities.show_four_images(x_28_28_first=ori.reshape(28, 28),
                    #                            x_28_28_first_title=f"attacking sample\nidx = {seed_idx}",
                    #                            x_28_28_second=adv.reshape(28, 28),
                    #                            x_28_28_second_title=f"adv\ntarget = {TARGET_LABEL}\nL0(this, ori) = {L0_before}\nL2(this, ori) = {np.round(L2_before, 2)}",
                    #                            x_28_28_third=smooth_adv.reshape(28, 28),
                    #                            x_28_28_third_title=f"optimized adv\nL0(this, ori) = {L0_after}\nL2(this, ori) = {np.round(L2_after, 2)}\nuse step = {STEP}",
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

    PCA_ALL = False
    if PCA_ALL:
        # ORI_LABEL = 2
        # X_attack = get_X_attack(X_train, y_train, WRONG_SEEDS, ORI_LABEL, N_ATTACKING_SAMPLES=100)
        TWO_DiIMENSION = False
        borders = get_border(X_train) * (X_train.reshape(-1, 28, 28, 1))
        if TWO_DiIMENSION:
            components = PCA(n_components=2).fit_transform(borders.reshape(-1, 28 * 28))
            fig = px.scatter(components, x=0, y=1, color=y_train)
            fig.show()
        else:
            components = PCA(n_components=3).fit_transform(borders.reshape(-1, 28 * 28))
            fig = px.scatter_3d(components, x=0, y=1, z=2, color=y_train)
            fig.show()

    VISUALIZE_BORDER = False
    if VISUALIZE_BORDER:
        ori = 2
        TARGET_LABEL = 1
        # AE_MODEL_H5 = f"{BASE_PATH}_{ORI_LABEL}_{TARGET_LABEL}.h5"
        AE_MODEL_H5 = "/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/multi-epsilon/Alexnet/2_1_sigmoid/autoencoder_Alexnetborder_2_1_epsilon0,3.h5"
        ae = keras.models.load_model(filepath=AE_MODEL_H5, compile=False)
        X_attack, selected_seeds = get_X_attack(X_train, y_train, WRONG_SEEDS, ori, N_ATTACKING_SAMPLES)

        # utilities.show_two_images(x_28_28_left=X_attack[0].reshape(28, 28), x_28_28_right=borders[0].reshape(28, 28), display=True)
        # visualize_cnn(x_image_4D=X_attack[0].reshape(-1, 28, 28, 1), model=ae, specified_layer=None)

        out_ae = ae.predict(X_attack)
        borders = get_border(X_attack)  # * (X_attack.reshape(-1, 28, 28, 1))
        out_border = out_ae * borders
        candidate_adv = None

        internal_origin_images = get_internal_images(X_attack, border_images=borders)
        candidate_adv = np.clip(out_border + internal_origin_images, 0, 1)

        Y_pred = dnn.predict(candidate_adv.reshape(-1, 28, 28, 1))
        y_pred = np.argmax(Y_pred, axis=1)

        for seed_idx in range(len(y_pred)):
            if y_pred[seed_idx] == TARGET_LABEL:
                utilities.show_four_images(
                    x_28_28_first=X_attack[seed_idx].reshape(28, 28),
                    x_28_28_second=out_ae[seed_idx].reshape(28, 28),
                    x_28_28_third=out_border[seed_idx].reshape(28, 28),
                    x_28_28_fourth=candidate_adv[seed_idx].reshape(28, 28),
                    x_28_28_first_title="origin",
                    x_28_28_second_title="output of AE",
                    x_28_28_third_title="new border pixel",
                    x_28_28_fourth_title=f"candidate adv\npred = {y_pred[seed_idx]}",
                    display=True)
