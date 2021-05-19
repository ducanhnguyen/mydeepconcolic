import csv as csv
import os

import cv2
import keras
import numpy as np
import plotly.express as px
import tensorflow as tf
from numpy import save
from sklearn.decomposition import PCA

from src.ae_attack_border.data import wrongseeds_AlexNet
from src.utils import utilities
from src.utils.feature_ranker_2d import feature_ranker, RANKING_ALGORITHM


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


def visualize_border(X_attack):
    border_origin_images = get_border(X_attack)


def generate_adv_for_single_attack_ALL_PATTERN(X_attack, TARGET_LABEL, ae, dnn):
    candidate_generated_borders = ae.predict(X_attack)
    candidates = np.clip(candidate_generated_borders, 0, 1)

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
    for idx in range(len(candidates)):
        if compare[idx]:
            advs.append(candidates[idx])
            oris.append(X_attack[idx])
    advs = np.asarray(advs)
    oris = np.asarray(oris)
    return n_adv, advs, oris


def generate_adv_for_single_attack_BORDER_PATTERN(X_attack, selected_seeds, TARGET_LABEL, ae, dnn):
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

                n_adv, _, _ = generate_adv_for_single_attack_BORDER_PATTERN(X_attack[:N_ATTACKING_SAMPLES], target, ae,
                                                                            dnn)
                print(f'attacking {ori} -> {target}: n_adv = {n_adv}')
                row.append(n_adv)

                # for generalization
                n_adv2, _, _ = generate_adv_for_single_attack_BORDER_PATTERN(X_attack[N_ATTACKING_SAMPLES:], target, ae,
                                                                             dnn)
                print(f'GENERALIZATION. # ori = {len(X_attack[2000:])} attacking {ori} -> {target}: n_adv = {n_adv2}')
                print("")
            else:
                row.append(0)

        adv_table.append(row)
    return adv_table


"""
Delete every single pixel and check whether it is adversary
Poor performance
"""


def smooth_vet_can(ori, adv, dnn, target_label):
    ori = ori.reshape(-1)
    smooth_adv = np.array(adv).reshape(-1)
    high_light = np.array(adv).reshape(-1)
    n_restored_pixels = 0
    n_changed_pixels_before = np.sum(ori != smooth_adv)
    print(f"n_changed_pixels_before = {n_changed_pixels_before}")

    restored = []
    for idx in range(len(ori)):
        if ori[idx] != smooth_adv[idx]:
            old = smooth_adv[idx]
            smooth_adv[idx] = ori[idx]
            Y_pred = dnn.predict(smooth_adv.reshape(-1, 28, 28, 1))
            pred = np.argmax(Y_pred, axis=1)[0]
            if pred != target_label:
                smooth_adv[idx] = old
            else:
                n_restored_pixels += 1
                high_light[idx] = 2
            restored.append(n_restored_pixels)

    print(restored)
    n_changed_pixels_after = n_changed_pixels_before - n_restored_pixels
    print(f"n_changed_pixels_after = {n_changed_pixels_after}")
    return smooth_adv, high_light, n_changed_pixels_after, n_changed_pixels_before


def rank_pixel(diff_pixel):
    return diff_pixel


def smooth_vet_can_step(ori, adv, dnn, target_label, step):
    n_restored_pixels = 0
    restored_pixel_by_prediction = []

    # normalize
    ori_norm = ori.reshape(-1)
    smooth_adv_norm = np.array(adv).reshape(-1)
    original_adv_norm = np.array(adv).reshape(-1)
    if np.min(ori) >= 0 and np.max(ori) <= 1:
        ori_norm = np.round(ori_norm * 255)
        smooth_adv_norm = np.round(smooth_adv_norm * 255)
        original_adv_norm = np.round(original_adv_norm * 255)
    L0_before = utilities.compute_l0(smooth_adv_norm, ori_norm, normalized=True)
    print(f"L0_before = {L0_before}")

    # get different pixels
    diff_pixel = []
    for idx in range(len(ori_norm)):
        if ori_norm[idx] != smooth_adv_norm[idx]:
            diff_pixel.append(idx)
    diff_pixel = rank_pixel(diff_pixel)
    diff_pixel = np.asarray(diff_pixel)

    #
    count = 0
    old_indexes = []
    old_values = []
    per_l2_arr = []
    for idx in diff_pixel:
        if ori_norm[idx] != smooth_adv_norm[idx]:
            count += 1
            old_indexes.append(idx)
            old_values.append(smooth_adv_norm[idx])
            smooth_adv_norm[idx] = ori_norm[idx]

        if count == step or idx == len(diff_pixel) - 1:
            Y_pred = dnn.predict((smooth_adv_norm / 255).reshape(-1, 28, 28, 1))
            pred = np.argmax(Y_pred, axis=1)[0]

            if pred != target_label:
                for jdx, value in zip(old_indexes, old_values):
                    smooth_adv_norm[jdx] = value
            else:
                n_restored_pixels += count

            old_indexes = []
            old_values = []
            count = 0
            restored_pixel_by_prediction.append(n_restored_pixels)
            # per_l2_arr.append(utilities.compute_l2(ori, smooth_adv) / utilities.compute_l2(ori, adv))


    L0_after = utilities.compute_l0(ori_norm, smooth_adv_norm, normalized=True)
    print(f"L0_after = {L0_after}")

    L2_after = utilities.compute_l2(ori_norm, smooth_adv_norm)
    L2_before = utilities.compute_l2(ori_norm, original_adv_norm)

    highlight = highlight_diff(original_adv_norm, smooth_adv_norm)
    return smooth_adv_norm, highlight, L0_after, L0_before, L2_after, L2_before, np.asarray(
        restored_pixel_by_prediction)


def highlight_diff(img1_0_255, img2_0_255):
    shape = img2_0_255.shape
    img1_0_255 = img1_0_255.reshape(-1)
    img2_0_255 = img2_0_255.reshape(-1)
    highlight = []
    for idx in range(len(img1_0_255)):
        if img1_0_255[idx] != img2_0_255[idx]:
            highlight.append(1)
        else:
            highlight.append(0)
    highlight = np.asarray(highlight)
    return highlight.reshape(shape)


def smooth_simple_using_heuristic(ori, adv, dnn, target_label):
    '''
    Get important features
    '''
    important_features = feature_ranker().find_important_features_of_a_sample(input_image=adv.reshape(28, 28, 1),
                                                                              n_rows=28, n_cols=28, n_channels=1,
                                                                              n_important_features=None,
                                                                              algorithm=RANKING_ALGORITHM.COI,
                                                                              gradient_label=target_label,
                                                                              classifier=dnn)
    f = []
    for idx in range(len(important_features)):
        f.append(important_features[idx][0] * 28 + important_features[idx][1])
    f = np.asarray(f)
    f = f[::-1]  # tinh importance tang dan
    N_PIXELS = len(f)

    '''
    Generate smooth advs
    '''
    possible_delta_arr = np.arange(0, 101, 1)
    smooth_adv_arr = []
    ori = ori.reshape(-1)
    adv = adv.reshape(-1)
    for delta in possible_delta_arr:
        n_most = int(np.round(delta * N_PIXELS / 100))
        smooth_adv = np.array(adv).reshape(-1)
        # smooth_adv = np.array(ori).reshape(-1)
        for idx in range(n_most):
            real_idx = f[idx]
            # smooth_adv[real_idx] = adv[real_idx]
            smooth_adv[real_idx] = ori[real_idx]
        smooth_adv_arr.append(smooth_adv)

    smooth_adv_arr = np.asarray(smooth_adv_arr)

    '''
    Predict
    '''
    smooth_adv_arr = smooth_adv_arr.reshape(-1, 28, 28, 1)
    Y_pred = dnn.predict(smooth_adv_arr)
    y_pred = np.argmax(Y_pred, axis=1)
    for idx in range(len(y_pred) - 1, -1, -1):
        # for idx in range(len(y_pred)):
        if y_pred[idx] == TARGET_LABEL:
            smooth_adv = smooth_adv_arr[idx]
            n_changed_pixels_after = np.sum(smooth_adv.reshape(-1) != ori.reshape(-1))
            n_changed_pixels_before = np.sum(adv.reshape(-1) != ori.reshape(-1))
            high_light = np.asarray(adv.reshape(-1) == smooth_adv.reshape(-1))
            return smooth_adv, high_light, n_changed_pixels_after, n_changed_pixels_before

    return None


if __name__ == '__main__':
    ATTACKED_MODEL_H5 = f"../../result/ae-attack-border/model/Alexnet.h5"
    WRONG_SEEDS = wrongseeds_AlexNet

    N_ATTACKING_SAMPLES = 1000
    N_CLASSES = 10

    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    X_train = X_train / 255

    dnn = keras.models.load_model(filepath=ATTACKED_MODEL_H5, compile=False)

    ALL_ATTACK = False
    if ALL_ATTACK:
        BASE_PATH = "../../result/ae-attack-border/epsilon=0,5/Alexnet/autoencoder_Alexnetborder"
        adv_table = generate_adv_for_all_classes(BASE_PATH, N_ATTACKING_SAMPLES, WRONG_SEEDS, N_CLASSES, dnn, X_train,
                                                 y_train, start=None, end=None)

    SINGLE_ATTACK_MODE = True
    if SINGLE_ATTACK_MODE:
        ORI_LABEL = 9
        TARGET_LABEL = 7

        true_label_arr = []
        pred_label_arr = []
        L0_after_arr = []
        L0_before_arr = []
        epsilon_arr = []

        epsilon_all = ["0,0", "0,1", "0,2", "0,3", "0,4", "0,5", "0,6", "0,7", "0,8", "0,9", "1,0"]
        # epsilon_all = ["0,0", "0,1"]
        # AE_MODEL_H5 = f"{BASE_PATH}_{ORI_LABEL}_{TARGET_LABEL}.h5"

        OUT_PATH = "../../result/ae-attack-border/Alexnet/ae_border/autoencoder_models/OUT/"
        if not os.path.exists(OUT_PATH):
            os.mkdir(OUT_PATH)
        CSV_PATH = OUT_PATH + 'out.csv'
        with open(CSV_PATH, mode='w') as f:
            seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            seed.writerow(
                ['epsilon', 'idx', 'true_label', 'pred_label', 'L0 (before)', 'L0 (after)', 'L2 (before)',
                 'L2 (after)'])
            f.close()

        for epsilon in epsilon_all:
            AE_MODEL_H5 = f"../../result/ae-attack-border/Alexnet/ae_border/autoencoder_models" \
                          f"/ae_border_Alexnet_9_7weight={epsilon}_1000autoencoder.h5"
            ae = keras.models.load_model(filepath=AE_MODEL_H5, compile=False)
            X_attack, selected_seeds = get_X_attack(X_train, y_train, WRONG_SEEDS, ORI_LABEL, N_ATTACKING_SAMPLES=1000)

            n_adv, advs, oris, adv_idxes = generate_adv_for_single_attack_BORDER_PATTERN(X_attack, selected_seeds,
                                                                                         TARGET_LABEL, ae, dnn)
            # n_adv, advs, oris = generate_adv_for_single_attack_ALL_PATTERN(X_attack, TARGET_LABEL, ae, dnn)

            print(f"epsilon = {epsilon}: #adv = {n_adv}")

            for ori, adv, seed_idx in zip(oris, advs, adv_idxes):
                step = 1
                smooth_adv, highlight, L0_after, L0_before, L2_after, L2_before, restored_pixel_by_prediction = \
                    smooth_vet_can_step(
                        ori, adv, dnn,
                        TARGET_LABEL,
                        step)
                per_pixel_by_prediction = restored_pixel_by_prediction / L0_before

                epsilon_arr.append(epsilon)
                true_label_arr.append(ORI_LABEL)
                pred_label_arr.append(TARGET_LABEL)
                L0_after_arr.append(L0_after)
                L0_before_arr.append(L0_before)

                save(f"{OUT_PATH}/epsilon{epsilon}_{ORI_LABEL}to{TARGET_LABEL}_step{step}_idx{seed_idx}_L0.npy"
                     , per_pixel_by_prediction)

                utilities.show_four_images(x_28_28_first=ori.reshape(28, 28),
                                           x_28_28_first_title=f"original image\nidx = {seed_idx}",
                                           x_28_28_second=adv.reshape(28, 28),
                                           x_28_28_second_title=f"adv\ntarget = {TARGET_LABEL}\nL0(this, ori) = {L0_before}\nL2(this, ori) = {np.round(L2_before, 2)}",
                                           x_28_28_third=smooth_adv.reshape(28, 28),
                                           x_28_28_third_title=f"smooth adv\nL0(this, ori) = {L0_after}\nL2(this, ori) = {np.round(L2_after, 2)}\nuse step = {step}",
                                           x_28_28_fourth=highlight.reshape(28, 28),
                                           x_28_28_fourth_title="diff(adv, smooth-adv)\nwhite means difference",
                                           display=False,
                                           path=f"{OUT_PATH}/epsilon{epsilon}_{ORI_LABEL}to{TARGET_LABEL}_step{step}_idx{seed_idx}")

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
