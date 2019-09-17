import csv
import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    adversarial_samples = '/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/digit-recognizer/adversarial_samples.csv'
    if not os.path.exists(adversarial_samples):
        # get selected seeds
        full_comparison_csv = '/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/result/mnist_run1/[osx] full/full_comparison.csv'
        if os.path.exists(full_comparison_csv):
            df = pd.read_csv(full_comparison_csv)
            selected_seeds = dict()
            for idx, value in enumerate(df['is_valid']):
                if value == True:
                    seed_index = int(df['seed_index'][idx])
                    label = int(df['original_prediction'][idx])
                    selected_seeds[seed_index] = dict()
                    selected_seeds[seed_index]['true_label'] = label
        print(f'size of selected_seeds = {len(selected_seeds.items())}')

        # get selective modified images
        seeds_dir = '/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/result/mnist_run1/[osx] full'
        selected_csv = []
        if os.path.exists(seeds_dir) and len(selected_seeds.items()) > 0:
            files = os.listdir(seeds_dir)
            for file in files:
                for seed_index, true_label in selected_seeds.items():
                    if file.startswith(str(seed_index) + '_') and file.endswith('_new.csv'):
                        absolute_path = os.path.abspath(os.path.join(seeds_dir, file))
                        selected_seeds[seed_index]['absolute_path'] = absolute_path
                        break
        print(f'selected_seeds = {selected_seeds}')

        # get expansion
        X_expansion = []
        for seed_index, dict in selected_seeds.items():
            label = int(dict['true_label'])
            pixels = pd.read_csv(dict['absolute_path'], header=None).to_numpy().reshape(-1)

            # full = label + pixels
            full = []
            full.append(label)
            for pixel in pixels:
                full.append(int(pixel*255)) # 0..1 -> 0..255

            X_expansion.append(full)
        X_expansion = np.asarray(X_expansion)

        # export expansion to file
        with open(adversarial_samples, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for x_expansion in X_expansion:
                csv_writer.writerow(x_expansion)

    else:
        X_expansion = pd.read_csv(adversarial_samples, header=None).to_numpy()
    print(f'expansion of train set = {X_expansion.shape}')

    # expand the train set
    Xtrain_mnist = pd.read_csv(
        '/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/digit-recognizer/train.csv').to_numpy()
    print(f'original shape of train set = {Xtrain_mnist.shape}')

    # merge
    Xtrain_mnist = np.asarray(Xtrain_mnist)  #0..1
    X_expansion = np.asarray(X_expansion) #0..1
    X_merge = np.concatenate((Xtrain_mnist, X_expansion), axis=0)
    print(f'X merge = {X_merge.shape}')
    expansion = '/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/digit-recognizer/train_expansion.csv'
    with open(expansion, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for x_merge in X_merge:
            csv_writer.writerow(x_merge)
