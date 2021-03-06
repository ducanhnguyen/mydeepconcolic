import numpy as np


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