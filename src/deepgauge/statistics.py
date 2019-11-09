from src.deepgauge.abstract_coverage_computation import *

from src.saved_models.fashion_mnist_ann_keras import *
from src.utils import keras_layer
import matplotlib.pyplot as plt
import numpy


class STATISTICS(abstract_coverage_computation):
    def __init__(self):
        super(STATISTICS, self).__init__()

    def get_range_of_a_neuron_given_an_observation(self, neuron_layer_index, neuron_unit_index, observation_index):
        assert (neuron_layer_index >= 0)
        assert (neuron_unit_index >= 0)

        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer

        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))  # (None, 1)
        assert (n_observations >= 1)

        # find value of neurons given an observation and a neuron
        neuron_value = -1 # undefined
        for idx in range(n_coverage_layers):
            layer_index_in_model = activation_layers[idx][1]
            if layer_index_in_model == neuron_layer_index:
                neuron_value = prediction[idx][observation_index][neuron_unit_index]
                break
        return neuron_value

    def get_range_of_a_neuron(self, neuron_layer_index, neuron_unit_index):
        assert (neuron_layer_index >= 0)
        assert (neuron_unit_index >= 0)

        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer

        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))  # (None, 1)
        assert (n_observations >= 1)

        # find the index
        neuron_values = []
        for idx in range(n_coverage_layers):
            layer_index_in_model = activation_layers[idx][1]
            if layer_index_in_model == neuron_layer_index:
                # find value of neurons given an observation and a neuron
                for j in range(n_observations):
                    neuron_value = prediction[idx][j][neuron_unit_index]
                    neuron_values.append(neuron_value)
                break

        # get range
        return np.min(neuron_values), np.max(neuron_values)

    def get_range_of_neurons(self):
        bounds = self.draw_boxplot_of_units(plot=False)['bound']
        neurons = []
        for key, value in bounds.items():
            # key: 'layer_idx:1,unit_idx:0'
            layer_idx_str = key.split(",")[0]
            layer_idx = int(layer_idx_str.replace("layer_idx:", ""))

            unit_idx_str = key.split(",")[1]
            unit_idx = int(unit_idx_str.replace("unit_idx:", ""))

            upper = value['upper']
            lower = value['lower']

            neurons.append([layer_idx, unit_idx, upper, lower])

        keys = ['layer_idx', 'unit_idx', 'upper', 'lower']
        return neurons, keys

    def draw_boxplot_of_layers_over_inputs(self):
        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer
        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))
        # print(f"n_observations = {n_observations}")

        # compute the layer distribution
        layer_distributions = dict()
        neuron_values_over_layer = []
        labels = []
        for idx in range(n_coverage_layers):
            layer_index_in_model = activation_layers[idx][1]
            layer_distributions[layer_index_in_model] = dict()
            n_units = keras_layer.get_number_of_units(self.get_model(), layer_index_in_model)

            neuron_values = []
            for unit_idx in range(n_units):
                for j in range(n_observations):
                    neuron_value = prediction[idx][j][unit_idx]
                    neuron_values.append(neuron_value)
            neuron_values_over_layer.append(neuron_values)
            labels.append(self.get_model().layers[layer_index_in_model].name)

        # https://matplotlib.org/3.1.1/gallery/pyplots/boxplot_demo_pyplot.html#sphx-glr-gallery-pyplots-boxplot-demo-pyplot-py
        plt.boxplot(neuron_values_over_layer, labels=labels, showfliers=False)
        plt.xlabel('layer')
        plt.ylabel('neuron value')
        plt.title(f'Boxplot of neuron values over layers')
        plt.show()

    def create_key(self, layer_idx, unit_idx):
        return "layer_idx:" + str(layer_idx) + ",unit_idx:" + str(unit_idx)

    def draw_boxplot_of_units(self, plot=False):
        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer
        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))  # (None, 1)
        print(f"n_observations = {n_observations}")
        bounds = dict()  # store upper value and lower value of a unit

        for idx in range(n_coverage_layers):
            layer_index_in_model = activation_layers[idx][1]
            n_units = keras_layer.get_number_of_units(self.get_model(), layer_index_in_model)

            labels = []
            neuron_values_over_layer = []
            for unit_idx in range(n_units):
                key = self.create_key(layer_index_in_model, unit_idx)
                bounds[key] = dict()
                labels.append(f'{unit_idx}')
                neuron_values = []
                for j in range(n_observations):
                    neuron_value = prediction[idx][j][unit_idx]
                    neuron_values.append(neuron_value)
                neuron_values_over_layer.append(neuron_values)
                bounds[key]['upper'] = np.max(neuron_values)
                bounds[key]['lower'] = np.min(neuron_values)

            if plot:
                # https://matplotlib.org/3.1.1/gallery/pyplots/boxplot_demo_pyplot.html#sphx-glr-gallery-pyplots-boxplot-demo-pyplot-py
                plt.boxplot(neuron_values_over_layer, labels=labels, showfliers=True)
                plt.xlabel('unit')
                plt.ylabel('neuron value')
                plt.title(f'layer {self.get_model().layers[layer_index_in_model].name}')
                plt.show()
        return {'bound': bounds}

    def find_top_observations(self, percentage, neuron_layer_index, neuron_unit_index):
        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer
        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))  # (None, 1)
        # print(f"n_observations = {n_observations}")

        # find value of neurons given an observation and a neuron
        for idx in range(n_coverage_layers):
            layer_index_in_model = activation_layers[idx][1]
            if layer_index_in_model == neuron_layer_index:

                neuron_values = []
                for j in range(n_observations):
                    neuron_value = prediction[idx][j][neuron_unit_index]
                    neuron_values.append(neuron_value)

                # sort neuron values in decreasing order of neuron value
                sorted_neuron_indexes = numpy.argsort(neuron_values)
                begin = int((1 - percentage) * len(sorted_neuron_indexes))
                end = n_observations

                reversed_arr = (sorted_neuron_indexes[begin:end]).tolist()
                reversed_arr.reverse()
                #for index in reversed_arr:
                #    print(neuron_values[index])

                return reversed_arr


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
        testset_path="/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/test.csv")
    # testset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f2_attacked/original_test_plus_expansion.csv')
    print(model.summary())

    # compute neuron coverage
    statistics_computator = STATISTICS()
    statistics_computator.set_model(model_object.get_model())
    statistics_computator.set_X(model_object.get_Xtrain())

    # draw boxplot
    statistics_computator.draw_boxplot_of_units(plot=True)

    # get range of a neuron
    '''
    bounds, _ = statistics_computator.get_range_of_neurons()
    print(f"bound = {bounds}")

    bounds = statistics_computator.get_range_of_a_neuron(neuron_layer_index=1, neuron_unit_index=0)
    print(f"bound = {bounds}")

    # find top observations
    top_obervation_indexes = statistics_computator.find_top_observations(percentage=0.01, neuron_layer_index=1,
                                                                         neuron_unit_index=0)
    print(f"top_observation_indexes of neuron = {top_obervation_indexes}")

    # get range of a neuron: 7140, 6530
    for top_obervation_index in top_obervation_indexes:
        range_ = statistics_computator.get_range_of_a_neuron_given_an_observation(
            neuron_layer_index=1, neuron_unit_index=0, observation_index=top_obervation_index)
        print(f"top_obervation_index = {top_obervation_index}: range = {range_}")
    '''