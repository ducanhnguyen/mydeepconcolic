import csv

import matplotlib.pyplot as plt
import numpy as np

from src.model_loader import initialize_dnn_model
from src.saved_models.mnist_ann_keras import MNIST_ANN_KERAS
from src.saved_models.mnist_deepcheck import MNIST_DEEPCHECK
from src.saved_models.mnist_simard import MNIST_SIMARD
from src.saved_models.mnist_simple import MNIST_SIMPLE
from src.utils.utilities import compute_l0


def is_float(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    #
    model_object = MNIST_DEEPCHECK()
    model_object.set_num_classes(10)
    model_object.read_data(
        trainset_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/train.csv',
        testset_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/test.csv')

    base_path = "/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_deepcheck_10k_first_samples/"
    summary_path = base_path + 'summary.csv'
    NORMALIZATION_FACTOR = 255

    with open(summary_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        arr = []
        for row in csv_reader:
            arr_row = []
            for item in row:
                if is_float(item):
                    arr_row.append(float(item))
                else:
                    arr_row.append(None)
            arr.append(arr_row)

    # ....
    IDX_delta_first_prod_vs_second_prob = 8
    IDX_seed = 0
    IDX_adv_label = 6
    IDX_true_label = 5
    IDX_L0 = 1
    IDX_L2 = 2

    seed_arr = []
    l0_arr = []
    l2_arr = []
    true_label_arr = []
    adv_label_arr = []

    for row in arr:
        num_adv = 0
        n_sample = 0
        if row[IDX_adv_label] is not None:
            l0_arr.append(int(row[IDX_L0]))
            l2_arr.append(np.round(row[IDX_L2], 2))
            seed_arr.append(int(row[IDX_seed]))
            adv_label_arr.append(int(row[IDX_adv_label]))
            true_label_arr.append(int(row[IDX_true_label]))

    l2_arr, l0_arr, seed_arr = zip(*sorted(zip(l2_arr, l0_arr, seed_arr), reverse=True))
    for i in range(0, 10):
        print(f'seed {seed_arr[i]}, l0 = {l0_arr[i]}, l2 = {l2_arr[i]}')

    #
    nrow = 1
    ncol = 2
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "15"
    plt.rcParams['axes.titlepad'] = 15

    visualized_idx = 0
    for visualized_idx in range(0, 4):
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.02)
        seed = seed_arr[visualized_idx]

        # add origin
        ori = model_object.get_Xtrain()[seed]
        ori = ori.reshape(28, 28)
        fig1 = fig.add_subplot(nrow, ncol, 1)
        fig1.title.set_text(f'origin: {true_label_arr[visualized_idx]}\n')
        fig1.imshow(ori, cmap="gray")

        # add adv
        adv = []
        with open(base_path + str(seed) + '.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in csv_reader:
                adv = np.asarray(row).astype(int)  # 0..255 for MNIST
        adv = adv / NORMALIZATION_FACTOR
        adv = adv.reshape(28, 28)
        fig2 = fig.add_subplot(nrow, ncol, 2)
        fig2.title.set_text(f'adv: {adv_label_arr[visualized_idx]}\nL0 = {l0_arr[visualized_idx]}, L2 = {l2_arr[visualized_idx]}')

        fig2.imshow(adv, cmap="gray")

        # hide measurement
        fig1.set_yticklabels([])
        fig1.set_xticklabels([])
        fig1.axis('off')

        fig2.set_yticklabels([])
        fig2.set_xticklabels([])
        fig2.axis('off')

        plt.show()

        # break
        # png_comparison_image_path = directory + f'/{seed_index}_comparison.png'
        # plt.savefig(png_comparison_image_path, pad_inches=0, bbox_inches='tight')
