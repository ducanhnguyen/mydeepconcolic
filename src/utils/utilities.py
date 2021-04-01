import numpy as np
import matplotlib.pyplot as plt

def compute_l2(adv: np.ndarray,
               ori: np.ndarray):  # 1d array, value in range of [0 .. 1]
    return np.linalg.norm(adv - ori)


def compute_l0(adv: np.ndarray,
               ori: np.ndarray):  # 1d array, value in range of [0 .. 1]
    l0_dist = 0
    for idx in range(len(adv)):
        if adv[idx] != ori[idx]:
            l0_dist += 1
    return l0_dist


def compute_linf(adv: np.ndarray,
                 ori: np.ndarray):  # 1d array, value in range of [0 .. 1]
    linf_dist = 0
    for idx in range(len(adv)):
        if np.abs(adv[idx] - ori[idx]) > linf_dist:
            linf_dist = np.abs(adv[idx] - ori[idx])
    return linf_dist


def compute_minimum_change(adv: np.ndarray,
                           ori: np.ndarray):  # 1d array, value in range of [0 .. 1]
    minimum_change = 99999999999
    for idx in range(len(adv)):
        delta = np.abs(adv[idx] - ori[idx])
        if 0 < delta < minimum_change:
            minimum_change = delta
    return minimum_change


def show_two_images(x_28_28_left, x_28_28_right, left_title="", right_title="", path = None, display = False):
    fig = plt.figure()
    fig1 = fig.add_subplot(1, 2, 1)
    fig1.title.set_text(left_title)
    plt.imshow(x_28_28_left, cmap="gray")

    fig2 = fig.add_subplot(1, 2, 2)
    fig2.title.set_text(right_title)
    plt.imshow(x_28_28_right, cmap='gray')

    if path is not None:
        plt.savefig(path, pad_inches=0, bbox_inches='tight')

    if display:
        plt.show()
