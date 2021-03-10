import csv

from src.log_analyzer import is_float


def load_summary(summary_path: str):
    with open(summary_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
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
    IDX_seed = 0
    IDX_adv_label = 6
    IDX_delta_first_prod_vs_second_prob = 8
    seed_adv_arr = []
    delta_prob_arr = dict()
    for row in arr:
        if row[IDX_adv_label] is not None:
            seed_adv_arr.append(int(row[IDX_seed]))
            delta_prob_arr[int(row[IDX_seed])] = row[IDX_delta_first_prod_vs_second_prob]
    return seed_adv_arr, delta_prob_arr


if __name__ == '__main__':
    seeds_mnist_ann_keras, delta_prob_arr = load_summary(
        '/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_ann_keras_10k_first_samples/summary.csv')
    seeds_mnist_ann_keras_v2, delta_prob_arr_v2 = load_summary(
        '/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_ann_keras_10k_first_samples_v2/summary.csv')
    seeds_mnist_ann_keras_v3, delta_prob_arr_v3 = load_summary(
        '/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_ann_keras_10k_first_samples_v3/summary.csv')

    joint_seeds = []
    for seed in seeds_mnist_ann_keras:
        if seed in seeds_mnist_ann_keras_v2 or seed in seeds_mnist_ann_keras_v3:
            joint_seeds.append(seed)
    print(joint_seeds)
    print(f'len = {len(seeds_mnist_ann_keras)}, {len(seeds_mnist_ann_keras_v2)}')
    print(f'len of joint seeds = {len(joint_seeds)}')
    # for seed in joint_seeds:
    #     print(
    #         f'joint seed {seed}: delta of prob = {delta_prob_arr[seed]}, {delta_prob_arr_v2[seed]}')
