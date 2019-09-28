from keras import Model
from tensorflow.python import keras
import numpy as np

from src.saved_models.fashion_mnist_ann_keras import *
from src.utils import keras_model, keras_layer


class K_MULTISECTION_NEURON_COVERAGE:
    def __init__(self):
        self.__X = None  # (None, ?)
        self.__model = None

    def load_model(self):
        assert (isinstance(self.get_model(), keras.engine.sequential.Sequential))
        activation_layers = keras_model.get_activation_layers(self.get_model())
        # ignore the last activation layer
        models = Model(inputs=self.get_model().input,
                       outputs=[item[0].output for item in activation_layers if
                                item[1] != len(self.get_model().layers) - 1])

        assert (len(self.get_X().shape) == 2)
        input = self.get_X().reshape(len(self.get_X()), -1)

        prediction = models.predict(input)  # (#activation_layers, #inputs, #hidden_units)
        return activation_layers, prediction

    def compute_the_number_of_neurons(self, activation_layers):
        # count the number of active neurons
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer

        # compute the number of neurons
        n_neurons = 0
        for idx in range(n_coverage_layers):
            layer_index_in_model = activation_layers[idx][1]
            n_units = keras_layer.get_number_of_units(self.get_model(), layer_index_in_model)

            for unit_idx in range(n_units):
                n_neurons += 1
        return n_neurons

    def compute_k_multisection_neuron_coverage(self, k):
        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer
        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))

        # compute the coverage
        covs = []
        N = 0
        for idx in range(n_coverage_layers):
            layer_index_in_model = activation_layers[idx][1]
            n_units = keras_layer.get_number_of_units(self.get_model(), layer_index_in_model)

            for unit_idx in range(n_units):
                # get lower and upper
                u = np.max(prediction[idx][unit_idx])# (#activation_layers, #inputs, #hidden_units)
                l = np.min(prediction[idx][unit_idx])
                fragment = np.abs(u - l) / k
                print(f'layer {idx}, unit {unit_idx}: [{l}, {u})')

                # run over
                N += 1
                n_active_times = 0

                for threshold in range(k):
                    # compute the fragment
                    start = l + threshold * fragment
                    if threshold == k - 1:
                        end = u
                    else:
                        end = start + fragment
                    #
                    is_active = False
                    for j in range(n_observations):
                        if prediction[idx][j][unit_idx] >= start and prediction[idx][j][unit_idx] < end:
                            is_active = True
                            break
                    if is_active:
                        n_active_times += 1
                assert (n_active_times<=k)

                # compute coverage
                covs.append(n_active_times)

        # final coverage
        final_coverage = np.sum(covs) / (k*N)
        print(final_coverage)
        return final_coverage

    def set_X(self, X):
        self.__X = X

    def get_X(self):
        return self.__X

    def get_model(self):
        return self.__model

    def set_model(self, model):
        assert (isinstance(model, keras.engine.sequential.Sequential))
        self.__model = model

if __name__ == '__main__':
    # construct model 1
    model_object1 = FASHION_MNIST()
    model_object1.set_num_classes(10)
    model1 = model_object1.load_model(
        weight_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/fashion_mnist_ann_keras_f1_original.h5',
        structure_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/fashion_mnist_ann_keras_f1_original.json',
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/train.csv')
    model_object1.read_data(
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/train.csv',
        testset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f1/original_test_plus_expansion.csv')
    print(model1.summary())

    # construct model 2
    '''
    model_object2 = FASHION_MNIST()
    model_object2.set_num_classes(10)
    model2 = model_object2.load_model(
        weight_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/fashion_mnist_ann_keras_f1_original.h5',
        structure_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/fashion_mnist_ann_keras_f1_original.json',
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/train.csv')
    model_object2.read_data(
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/train.csv',
        testset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/test.csv')
    print(model2.summary())
    '''
    # compute neuron coverage
    nc_computator = K_MULTISECTION_NEURON_COVERAGE()
    nc_computator.set_model(model_object1.get_model())
    nc_computator.set_X(model_object1.get_Xtest())
    nc_computator.compute_k_multisection_neuron_coverage(k=1000)
    # fashion mnist: xtest = 0.8383928571428572 (k=100)
    # xtest_expansion: 0.8392857142857143 (k=100)

    # fashion mnist: xtest = 0.6989107142857143 (k=1000)
    # xtest_expansion: 0.6989107142857143(k=1000)