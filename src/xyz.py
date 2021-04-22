from src.log_analyzer import read_data
import numpy as np

from os import listdir
def getcsv(path: str):
    adv_seeds = []
    print(f'analyze {path}')
    arr = read_data(path)
    IDX_adv_label = 6
    IDX_seed = 0

    import random
    random.shuffle(arr)

    for row in arr:
        if row[IDX_adv_label] is not None:
            adv_seeds.append(row[IDX_seed])
    return adv_seeds


def getname(path: str):
    # '/Users/ducanhnguyen/Documents/mydeepconcolic/result/bestFeatureAttack_delta255_or/result/mnist_ann_keras/summary.csv'
    # -> bestFeatureAttack_delta255_or
    return path.split("/")[-4:-3][0]


def get_index(path: str):
    possible_names = ['bestFeatureAttack_delta255_or', 'bestFeatureAttack_delta255_secondLabelTarget',
                      'bestFeatureAttack_delta255_upperbound',
                      'nonzeroAttack_delta100_or', 'nonzeroAttack_delta100_secondLabelTarget',
                      'nonzeroAttack_delta100_upperBound',
                      'edgeAttack_delta100_or', 'edgeAttack_delta100_secondLabelTarget',
                      'edgeAttack_delta100_upperBound'
                      ]
    for i in range(len(possible_names)):
        if possible_names[i] == path:
            return i
    return -1


if __name__ == '__main__':
    dataset = 'mnist_simple'
    paths = [
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/bestFeatureAttack_delta255_or/result/{dataset}/natural',
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/bestFeatureAttack_delta255_secondLabelTarget/result/{dataset}/natural',
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/bestFeatureAttack_delta255_upperbound/result/{dataset}/natural',
        #
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/nonzeroAttack_delta100_or/result/{dataset}/natural',
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/nonzeroAttack_delta100_secondLabelTarget/result/{dataset}/natural',
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/nonzeroAttack_delta100_upperBound/result/{dataset}/natural',
        # #
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_or/result/{dataset}/natural',
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_secondLabelTarget/result/{dataset}/natural',
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_upperBound/result/{dataset}/natural'
    ]

    adv_seeds = []
    for path in paths:
        print(path)
        for f in listdir(path):
            adv_seeds.append(f)
        print(len(adv_seeds))
        print()

    # compute total adv
    print(len(adv_seeds))