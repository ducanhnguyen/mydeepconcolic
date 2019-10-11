from keras import Model
from tensorflow.python import keras

from src.saved_models.fashion_mnist_ann_keras import *
from src.utils import keras_model, keras_layer


class abstract_coverage_computation:
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

    def compute_output_values_of_neuron(self, prediction, activation_layers, n_coverage_layers, n_observations):
        output_values = []

        for layer_idx in range(n_coverage_layers):
            layer_index_in_model = activation_layers[layer_idx][1]
            n_units = keras_layer.get_number_of_units(self.get_model(), layer_index_in_model)

            layers = []
            for unit_idx in range(n_units):
                unit_value_over_observations = []
                for i in range(n_observations):
                    unit_value_over_observations.append(prediction[layer_idx][i][unit_idx])
                layers.append(unit_value_over_observations)

            output_values.append(layers)
        return output_values  # (#layer, #units, #inputs)

    def set_X(self, X):
        self.__X = X

    def get_X(self):
        return self.__X

    def get_model(self):
        return self.__model

    def set_model(self, model):
        assert (isinstance(model, keras.engine.sequential.Sequential))
        self.__model = model