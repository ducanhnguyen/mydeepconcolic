import numpy as np

from src.utils.utilities import compute_l0s, show_ori_adv_optmizedadv

if __name__ == '__main__':
    advs = np.load(
        "/Users/ducanhnguyen/Documents/mydeepconcolic/optimization_batch3/Alexnet/ae_border_Alexnet_9to7_rankingCOI_step60/adv.npy")
    advs = advs.reshape(-1, 28, 28)

    oris = np.load(
        "/Users/ducanhnguyen/Documents/mydeepconcolic/optimization_batch3/Alexnet/ae_border_Alexnet_9to7_rankingCOI_step60/origin.npy")
    oris = oris.reshape(-1, 28, 28)

    optimized_advs = np.load(
        "/Users/ducanhnguyen/Documents/mydeepconcolic/optimization_batch3/Alexnet/ae_border_Alexnet_9to7_rankingCOI_step60/optimized_adv.npy")
    optimized_advs = optimized_advs.reshape(-1, 28, 28)

    selected = []
    l0s = compute_l0s(optimized_advs.reshape(-1, 784), oris.reshape(-1, 784), n_features=784, normalized=True)
    for idx in range(len(l0s)):
        # if l0s[idx] <= 10:
        selected.append(idx)
    selected = selected[:7]
    highlight = optimized_advs != advs
    show_ori_adv_optmizedadv(oris[selected], advs[selected], optimized_advs[selected], highlight[selected],
                             display=True, path=None)
