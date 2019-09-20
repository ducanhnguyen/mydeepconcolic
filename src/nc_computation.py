import matplotlib.pyplot as plt
from keras.models import Model

from src.saved_models.fashion_mnist_ann_keras import *
from src.utils import keras_layer
from src.utils import keras_model
from scipy.stats import norm

class NC_COMPUTATION:
    def __init__(self):
        self.__X = None  # (None, ?)
        self.__y = None  # 1-D
        self.__model = None

    def load_model(self):
        assert (isinstance(self.get_model(), keras.engine.sequential.Sequential))
        activation_layers = keras_model.get_activation_layers(self.get_model())
        # ignore the last activation layer
        models = Model(inputs=self.get_model().input,
                       outputs=[item[0].output for item in activation_layers if
                                item[1] != len(self.get_model().layers) - 1])

        assert (len(self.get_X().shape) == 2 and len(self.get_y().shape) == 1)
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

    def compute_nc_coverage(self, thresholds, plot=True):
        assert (len(thresholds) >= 1)
        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer
        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))
        n_neurons = self.compute_the_number_of_neurons(activation_layers)

        # compute the coverage
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

    def draw_boxplot_of_layers_over_inputs(self):
        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer
        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))

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

    def draw_boxplot_of_units(self):
        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer
        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))

        # compute the layer distribution
        layer_distributions = dict()

        for idx in range(n_coverage_layers):
            layer_index_in_model = activation_layers[idx][1]
            layer_distributions[layer_index_in_model] = dict()
            n_units = keras_layer.get_number_of_units(self.get_model(), layer_index_in_model)

            labels = []
            neuron_values_over_layer = []
            for unit_idx in range(n_units):
                labels.append(f'{unit_idx}')

                neuron_values = []
                for j in range(n_observations):
                    neuron_value = prediction[idx][j][unit_idx]
                    neuron_values.append(neuron_value)
                neuron_values_over_layer.append(neuron_values)

            # https://matplotlib.org/3.1.1/gallery/pyplots/boxplot_demo_pyplot.html#sphx-glr-gallery-pyplots-boxplot-demo-pyplot-py
            plt.boxplot(neuron_values_over_layer, labels=labels, showfliers=False)
            plt.xlabel('unit')
            plt.ylabel('neuron value')
            plt.title(f'layer {self.get_model().layers[layer_index_in_model].name}')
            plt.show()

    def get_X(self):
        return self.__X

    def set_X(self, X):
        self.__X = X

    def get_y(self):
        return self.__y

    def set_y(self, y):
        self.__y = y

    def get_model(self):
        return self.__model

    def set_model(self, model):
        assert (isinstance(model, keras.engine.sequential.Sequential))
        self.__model = model


if __name__ == '__main__':
    # construct model
    model_object = FASHION_MNIST()
    model_object.set_num_classes(10)
    model = model_object.load_model(
        weight_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/fashion_mnist_ann_keras_f1_original.h5',
        structure_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/fashion_mnist_ann_keras_f1_original.json',
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f1/original_train_plus_expansion.csv')
    model_object.read_data(
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f1/original_train_plus_expansion.csv',
        testset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/test.csv')
    print(model.summary())

    # compute neuron coverage
    nc_computator = NC_COMPUTATION()
    nc_computator.set_model(model_object.get_model())
    nc_computator.set_X(model_object.get_Xtrain())
    nc_computator.set_y(model_object.get_ytrain())
    #nc_computator.draw_boxplot_of_units()

    thresholds = np.arange(start=0, stop = 20, step = 1)
    coverage_1 = nc_computator.compute_nc_coverage(thresholds=thresholds, plot = True)
    print(coverage_1)