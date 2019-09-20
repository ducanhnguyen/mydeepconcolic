import csv
import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    adversarial_samples_file = '/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f1/test_expansion.csv'
    assert (adversarial_samples_file.endswith('.csv'))

    full_comparison_csv = '/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f1/full_comparison.csv'
    assert (full_comparison_csv.endswith('.csv') and os.path.exists(full_comparison_csv))

    original_train_file = '/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/test.csv'
    assert (original_train_file.endswith('.csv') and os.path.exists(original_train_file))

    expansion_train_file = '/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f1/original_test_plus_expansion.csv'
    assert (expansion_train_file.endswith('.csv'))

    selected_seeds_folder = '/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f1/detail'
    assert (os.path.exists(selected_seeds_folder))

    # get selected seeds
    if os.path.exists(full_comparison_csv):
        df = pd.read_csv(full_comparison_csv)
        selected_seeds = dict()
        for idx, value in enumerate(df['is_valid']):
            if value == True:
                seed_index = int(df['seed_index'][idx])
                label = int(df['original_prediction'][idx])
                if seed_index in selected_seeds:
                    print(f'exist {seed_index}')
                selected_seeds[seed_index] = dict()
                selected_seeds[seed_index]['true_label'] = label
            else:
                print(int(df['seed_index'][idx]))
    print(f'size of selected_seeds = {len(selected_seeds.items())}')

    # get selective modified images
    selected_csv = []
    if os.path.exists(selected_seeds_folder) and len(selected_seeds.items()) > 0:
        files = os.listdir(selected_seeds_folder)
        for file in files:
            for seed_index, true_label in selected_seeds.items():
                if file.startswith(str(seed_index) + '_') and file.endswith('_new.csv'):
                    absolute_path = os.path.abspath(os.path.join(selected_seeds_folder, file))
                    selected_seeds[seed_index]['absolute_path'] = absolute_path
                    break
    print(f'selected_seeds = {selected_seeds}')

    # create expansion train set
    X_expansion = []
    for seed_index, dict in selected_seeds.items():
        label = int(dict['true_label'])
        pixels = pd.read_csv(dict['absolute_path'], header=None).to_numpy().reshape(-1)
        # full = label + pixels
        full = []
        full.append(label)
        for pixel in pixels:
            full.append(int(pixel * 255))  # 0..1 -> 0..255

        X_expansion.append(full)
    X_expansion = np.asarray(X_expansion)

    # export expansion train set to file
    with open(adversarial_samples_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for x_expansion in X_expansion:
            csv_writer.writerow(x_expansion)
    print(f'expansion of train set = {X_expansion.shape}')

    #  # merge the original train set and the expansion train set
    Xtrain_mnist = pd.read_csv(original_train_file).to_numpy()
    print(f'original shape of train set = {Xtrain_mnist.shape}')
    Xtrain_mnist = np.asarray(Xtrain_mnist)  # 0..1
    X_expansion = np.asarray(X_expansion)  # 0..1
    X_merge = np.concatenate((Xtrain_mnist, X_expansion), axis=0)
    print(f'X merge = {X_merge.shape}')
    with open(expansion_train_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for x_merge in X_merge:
            csv_writer.writerow(x_merge)
