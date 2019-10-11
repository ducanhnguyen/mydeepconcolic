from src.deepgauge.abstract_coverage_computation import *

from src.saved_models.fashion_mnist_ann_keras import *
from src.utils import keras_layer
import matplotlib.pyplot as plt

class NC_COMPUTATION(abstract_coverage_computation):
    def __init__(self):
        super(NC_COMPUTATION, self).__init__()

    def compute_nc_coverage(self, thresholds, plot=True):
        assert (len(thresholds) >= 1)
        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer
        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))
        print(f'number of observations = {n_observations}')
        n_neurons = self.compute_the_number_of_neurons(activation_layers)

        coverage_thresholds = []
        n_active_neurons_dict = dict()
        for threshold in thresholds:
            print(f'Analyzing threshold {threshold}')
            n_active_neuron = 0

            for idx in range(n_coverage_layers):
                layer_index_in_model = activation_layers[idx][1]
                n_units = keras_layer.get_number_of_units(self.get_model(), layer_index_in_model)

                for unit_idx in range(n_units):
                    for j in range(n_observations):
                        if prediction[idx][j][unit_idx] >= threshold:
                            n_active_neuron += 1  # increase the number of active neurons
                            break
            n_active_neurons_dict[threshold] = n_active_neuron
            #print(f'threshold {threshold}: nc cov = {n_active_neuron / n_neurons}')
            coverage_thresholds.append(n_active_neuron / n_neurons)

        # plot
        if plot:
            print(f'n_neurons = {n_neurons}')
            nc_coverages = []
            for threshold in thresholds:
                cov = n_active_neurons_dict[threshold] / n_neurons
                # print(f'threshold = {threshold}: nc coverage = {cov}')
                nc_coverages.append(cov)
            plt.plot(thresholds, nc_coverages)
            plt.xlabel('threshold')
            plt.ylabel('nc coverage')
            plt.ylim(0, 1)  # neuron coverage is in range of [0..1]
            plt.show()

        return coverage_thresholds

if __name__ == '__main__':
    # construct model
    model_object = FASHION_MNIST()
    model_object.set_num_classes(10)
    model = model_object.load_model(
        weight_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/fashion_mnist_ann_keras_f2_original.h5',
        structure_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/fashion_mnist_ann_keras_f2_original.json',
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/train.csv')
    model_object.read_data(
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/train.csv',
        #testset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/test.csv')
        testset_path = '/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f2/original_test_plus_expansion.csv')
    print(model.summary())

    # compute neuron coverage
    nc_computator = NC_COMPUTATION()
    nc_computator.set_model(model_object.get_model())
    nc_computator.set_X(model_object.get_Xtest())

    thresholds = np.arange(start=0, stop = 20, step = 1)
    coverage = nc_computator.compute_nc_coverage(thresholds=thresholds, plot = True)
    print(coverage)