from src.log_analyzer import read_data
import numpy as np


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
    N_ATTACK = 9
    dataset = 'mnist_simple'
    paths = [
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/bestFeatureAttack_delta255_or/result/{dataset}/summary.csv',
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/bestFeatureAttack_delta255_secondLabelTarget/result/{dataset}/summary.csv',
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/bestFeatureAttack_delta255_upperbound/result/{dataset}/summary.csv',
        #
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/nonzeroAttack_delta100_or/result/{dataset}/summary.csv',
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/nonzeroAttack_delta100_secondLabelTarget/result/{dataset}/summary.csv',
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/nonzeroAttack_delta100_upperBound/result/{dataset}/summary.csv',
        #
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_or/result/{dataset}/summary.csv',
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_secondLabelTarget/result/{dataset}/summary.csv',
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/edgeAttack_delta100_upperBound/result/{dataset}/summary.csv'
    ]
    IDX_bestFeatureAttack = 0
    IDX_edgeAttack = 1
    IDX_nonzeroAttack = 2

    adv_seeds = dict()
    for path in paths:
        idx = get_index(getname(path))
        adv_seeds[idx] = getcsv(path)

    # compute joint adv
    summary = np.zeros(shape=(N_ATTACK, N_ATTACK), dtype=int)
    for k1, v1 in adv_seeds.items():
        for k2, v2 in adv_seeds.items():
            joint = set(v1).intersection(set(v2))
            summary[k1][k2] = len(joint)

    # create matrix
    normalize = np.zeros(shape=(N_ATTACK, N_ATTACK), dtype=float)
    s = '['
    for i in range(summary.shape[0]):
        s += '['
        for j in range(summary.shape[1]):
            normalize[i][j] = np.round(summary[i][j] * 1.0 / summary[j][j], 2)
            s += f'{normalize[i][j]},'
        s += '],\n'

    s = s.replace(',],', '],')
    s = s + ']'
    print(s)

    # compute total adv
    adv_all = []
    for k1, v1 in adv_seeds.items():
        for e in v1:
            adv_all.append(e)
    print(len(adv_all))