import os
import pandas as pd
import numpy as np
'''
Remove duplicated seeds
'''
if __name__ == '__main__':
    full_comparison_csv = '/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f2/full_comparison.csv'
    assert (full_comparison_csv.endswith('.csv') and os.path.exists(full_comparison_csv))
    df = pd.read_csv(full_comparison_csv)

    # get duplicated seed in the csv
    seeds = []
    duplicated_seeds = list()
    for idx in range(len(df['seed_index'])):
        seed_idx = df.iloc[idx]['seed_index']
        if seed_idx not in seeds:
            seeds.append(seed_idx)
        else:
            duplicated_seeds.append(idx)
    duplicated_seeds = np.asarray(duplicated_seeds)

    # remove the duplicated seed in the csv
    df = df.drop(duplicated_seeds)

    # export to file
    clean_full_comparison_csv = '/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f2/full_comparison_clean.csv'
    df.to_csv(clean_full_comparison_csv)
    print('export to csv done')