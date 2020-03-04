'''
Use in fgsm
'''
import os
import csv
import pandas as pd
from keras.models import Sequential

from src.saved_models.mnist_ann_keras import MNIST

if __name__ == '__main__':
    adversarial_samples_file = '/home/pass-la-1/PycharmProjects/mydeepconcolic/result/mnist fgsm e=1:255 *(1 to 9)/expansion.csv'
    assert (adversarial_samples_file.endswith('.csv'))

    original_train_file = '/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/digit-recognizer/test.csv'
    assert (original_train_file.endswith('.csv') and os.path.exists(original_train_file))

    expansion_train_file = '/home/pass-la-1/PycharmProjects/mydeepconcolic/result/mnist fgsm e=1:255 *(1 to 9)/original_test_plus_expansion.csv'
    assert (expansion_train_file.endswith('.csv'))

    selected_seeds_folder = '/home/pass-la-1/PycharmProjects/mydeepconcolic/result/mnist fgsm e=1:255 *(1 to 9)/detail'
    assert (os.path.exists(selected_seeds_folder))

    # load model
    model_object = MNIST()
    model_object.set_num_classes(10)
    model = model_object.load_model(
        weight_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/mnist_ann_keras_f1_original.h5',
        structure_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/mnist_ann_keras_f1_original.json',
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/digit-recognizer/train.csv')
    model_object.read_data(
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/digit-recognizer/train.csv',
        testset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/digit-recognizer/test.csv')
    model = model_object.get_model()
    assert (isinstance(model, Sequential))
    print(model.summary())

    # read data
    labels = []
    if os.path.exists(selected_seeds_folder):
        full = []
        files = os.listdir(selected_seeds_folder)
        for file in files:
            if file.endswith(".csv"):
                absolute_path = os.path.abspath(os.path.join(selected_seeds_folder, file))
                df = pd.read_csv(absolute_path, header=None).to_numpy().reshape(-1)

                # normalize pixel to [0..255]
                tmp = []
                for pixel in df:
                    tmp.append(int (pixel * 255))

                # add label
                seed = str(file).split('_')[0]
                seed = int (seed)
                _, label = model_object.get_an_observation_from_test_set(seed)
                tmp.append(label)

                full.append(tmp)

    # write data
    print(f"writing to {adversarial_samples_file}")
    with open(adversarial_samples_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(full)
