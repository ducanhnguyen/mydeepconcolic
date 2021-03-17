from src.adv_plot import is_float
from src.saved_models.mnist_ann_keras import MNIST_ANN_KERAS
from src.saved_models.mnist_deepcheck import MNIST_DEEPCHECK
import numpy as np
import csv

from src.saved_models.mnist_simard import MNIST_SIMARD
from src.saved_models.mnist_simple import MNIST_SIMPLE

if __name__ == '__main__':
    #
    model_object = MNIST_SIMARD()
    model_object.set_num_classes(10)
    model_object.read_data(
        trainset_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/train.csv',
        testset_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/test.csv')

    base_path = "/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_simard_10k_first_samples/"
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
    IDX_position_adv_label = 7
    IDX_true_label = 5
    IDX_L0 = 1
    IDX_L2 = 2

    seed_arr = []
    true_label_arr = []
    adv_label_arr = []
    adv_label_dict = {}
    for i in range(1, 11):
        adv_label_dict[i] = 0

    for row in arr:
        if row[IDX_position_adv_label] is not None:
            position_adv_label = int(row[IDX_position_adv_label])
            adv_label_arr.append(position_adv_label)
            # if position_adv_label == 10:
            adv_label_dict[position_adv_label] = adv_label_dict[position_adv_label] + 1

    for i in range(1, 11):
        adv_label_dict[i] = round(adv_label_dict[i] / len(adv_label_arr)*100,2)
    print(adv_label_dict)
