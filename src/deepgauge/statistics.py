from src.deepgauge.abstract_coverage_computation import *

from src.saved_models.fashion_mnist_ann_keras import *
from src.utils import keras_layer
import matplotlib.pyplot as plt

class STATISTICS(abstract_coverage_computation):
    def __init__(self):
        super(STATISTICS, self).__init__()


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
        print(f"n_observations = {n_observations}")

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

    def draw_boxplot_of_units(self, plot = False):
        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer
        n_observations = len(self.get_X().reshape(len(self.get_X()), -1)) # (None, 1)
        print(f"n_observations = {n_observations}")
        bounds = dict() # store upper value and lower value of a unit

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
        #testset_path="/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/test.csv")
        testset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f2_attacked/original_test_plus_expansion.csv')
    print(model.summary())

    # compute neuron coverage
    statistics_computator = STATISTICS()
    statistics_computator.set_model(model_object.get_model())
    statistics_computator.set_X(model_object.get_Xtest())

    bounds = statistics_computator.draw_boxplot_of_units(plot=True)['bound']
    print(bounds)