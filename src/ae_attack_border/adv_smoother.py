import numpy as np

from src.ae_attack_border.pixel_ranking import rank_pixel_S3, rank_pixel_S1, rank_pixel_S2
from src.utils import utilities
from src.utils.feature_ranker_2d import feature_ranker, RANKING_ALGORITHM


def smooth_vet_can_step(ori, adv, dnn, target_label, step, strategy):
    n_restored_pixels = 0
    restored_pixel_by_prediction = []

    # normalize
    ori_0_255 = ori.reshape(-1)
    smooth_adv_0_255 = np.array(adv).reshape(-1)
    original_adv_0_255 = np.array(adv).reshape(-1)
    if np.min(ori) >= 0 and np.max(ori) <= 1:
        ori_0_255 = np.round(ori_0_255 * 255)
        smooth_adv_0_255 = np.round(smooth_adv_0_255 * 255)
        original_adv_0_255 = np.round(original_adv_0_255 * 255)

    L0_before = utilities.compute_l0(smooth_adv_0_255, ori_0_255, normalized=True)
    # print(f"L0_before = {L0_before}")

    # get different pixels
    diff_pixel_arr = []
    for diff_pixel_idx in range(len(ori_0_255)):
        if ori_0_255[diff_pixel_idx] != smooth_adv_0_255[diff_pixel_idx]:
            diff_pixel_arr.append(diff_pixel_idx)

    if strategy == 'S3':
        diff_pixel_arr = rank_pixel_S3(diff_pixel_arr, adv, target_label, dnn)
    elif strategy == 'S2':
        diff_pixel_arr = rank_pixel_S2(diff_pixel_arr)
    elif strategy == 'S1':
        diff_pixel_arr = rank_pixel_S1(diff_pixel_arr)

    diff_pixel_arr = np.asarray(diff_pixel_arr)

    #
    count = 0
    old_indexes = []
    old_values = []
    for diff_pixel_idx in diff_pixel_arr:
        if ori_0_255[diff_pixel_idx] != smooth_adv_0_255[diff_pixel_idx]:
            count += 1
            old_indexes.append(diff_pixel_idx)
            old_values.append(smooth_adv_0_255[diff_pixel_idx])
            smooth_adv_0_255[diff_pixel_idx] = ori_0_255[diff_pixel_idx]  # try to restore

        if count == step \
                or diff_pixel_idx == diff_pixel_arr[-1]:
            Y_pred = dnn.predict((smooth_adv_0_255 / 255).reshape(-1, 28, 28, 1))
            pred = np.argmax(Y_pred, axis=1)[0]

            if pred != target_label:
                # revert the changes
                for jdx, value in zip(old_indexes, old_values):
                    smooth_adv_0_255[jdx] = value
            else:
                n_restored_pixels += count

            old_indexes = []
            old_values = []
            count = 0
            restored_pixel_by_prediction.append(n_restored_pixels)

    L0_after = utilities.compute_l0(ori_0_255, smooth_adv_0_255, normalized=True)
    # print(f"L0_after = {L0_after}")

    L2_after = utilities.compute_l2(ori_0_255, smooth_adv_0_255)
    L2_before = utilities.compute_l2(ori_0_255, original_adv_0_255)

    highlight = utilities.highlight_diff(original_adv_0_255, smooth_adv_0_255)

    return smooth_adv_0_255, highlight, L0_after, L0_before, L2_after, L2_before, np.asarray(
        restored_pixel_by_prediction)


def smooth_vet_can_step_adaptive(ori, adv, dnn, target_label, initial_step, strategy):
    restored_pixel_arr = []
    L0 = []
    L2 = []
    smooth_adv_0_1 = adv.reshape(-1)

    smooth_adv_0_255 = None
    for idx in range(0, 5):
        smooth_adv_0_255, highlight, L0_after, L0_before, L2_after, L2_before, restored_pixel = \
            smooth_vet_can_step(ori, smooth_adv_0_1, dnn, target_label, initial_step, strategy)

        L0.append(L0_before)
        L0.append(L0_after)
        L2.append(L2_before)
        L2.append(L2_after)

        if len(restored_pixel_arr) >= 1:
            latest = restored_pixel_arr[-1]
        else:
            latest = 0
        for jdx in restored_pixel:
            restored_pixel_arr.append(jdx + latest)

        initial_step = int(np.round(initial_step / 2))
        if initial_step == 0:
            break
        else:
            smooth_adv_0_1 = smooth_adv_0_255 / 255

    restored_pixel_arr = np.asarray(restored_pixel_arr)

    highlight = utilities.highlight_diff(np.round(adv*255), smooth_adv_0_255)

    return smooth_adv_0_255, highlight, L0[-1], L0[0], L2[-1], L2[0], restored_pixel_arr


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
        if y_pred[idx] == target_label:
            smooth_adv = smooth_adv_arr[idx]
            n_changed_pixels_after = np.sum(smooth_adv.reshape(-1) != ori.reshape(-1))
            n_changed_pixels_before = np.sum(adv.reshape(-1) != ori.reshape(-1))
            high_light = np.asarray(adv.reshape(-1) == smooth_adv.reshape(-1))
            return smooth_adv, high_light, n_changed_pixels_after, n_changed_pixels_before

    return None
