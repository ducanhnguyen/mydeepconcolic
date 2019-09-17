import matplotlib.pyplot as plt
from keras.models import Model

from src.saved_models.mnist_ann_keras import *
from src.utils import keras_layer
from src.utils import keras_model


class NC_COMPUTATION:
    def __init__(self):
        self.__X = None  # (None, ?)
        self.__y = None  # 1-D
        self.__model = None

    def compute_nc_coverage(self, threshold=0.1):
        assert (len(self.get_X().shape) == 2 and len(self.get_y().shape) == 1)
        assert (isinstance(self.get_model(), keras.engine.sequential.Sequential))

        activation_layers = keras_model.get_activation_layers(self.get_model())
        # ignore the last activsation layer
        models = Model(inputs=self.get_model().input,
                       outputs=[item[0].output for item in activation_layers if item[1] != len(self.get_model().layers)-1])
        input = self.get_X().reshape(len(self.get_X()), -1)
        prediction = models.predict(input)  # (#activation_layers, #inputs, #hidden_units)

        # count the number of active neurons
        N = len(activation_layers) - 1 # -1: ignore the last activation layer
        M = len(input)
        total = 0
        n_active_neuron = 0
        for i in range(N):
            for j in range(M):
                layer_index = activation_layers[i][1]
                units = keras_layer.get_number_of_units(self.get_model(), layer_index)
                for unit_idx in range(units):
                    total += 1
                    if prediction[i][j][unit_idx] >= threshold:
                        n_active_neuron += 1

        return n_active_neuron, total

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
    model_object = MNIST()
    model_object.set_num_classes(10)
    model = model_object.load_model(
        weight_path='/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/src/saved_models/mnist_ann_keras_expansion.h5',
        structure_path='/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/src/saved_models/mnist_ann_keras_expansion.json',
        trainset_path='/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/digit-recognizer/train_expansion.csv')
    model_object.read_data(
        trainset_path='/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/digit-recognizer/train_expansion.csv',
        testset_path='/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/digit-recognizer/test.csv')
    print(model.summary())

    # compute neuron coverage
    com = NC_COMPUTATION()
    com.set_model(model_object.get_model())
    com.set_X(model_object.get_Xtrain())
    com.set_y(model_object.get_ytrain())
    thresholds = [0, 0.25, 0.5, 0.75, 1]
    covs = []
    for threshold in thresholds:
        n_active_neuron, total = com.compute_nc_coverage(threshold=threshold)
        cov = n_active_neuron / total
        print(f'nc coverage = {cov}')
        covs.append(cov)

    # plot
    plt.plot(thresholds, covs)
    plt.xlabel('threshold')
    plt.ylabel('nc coverage')
    plt.ylim(0, 1)
    plt.show()

    # result
    # mnist: delta = 50: expansion = [1.0, 0.6474857001972386, 0.6293555226824458, 0.6111558185404339, 0.5929704142011835]
    # original = [1.0, 0.6088883928571428, 0.5928174603174603, 0.5766800595238095, 0.5603546626984127]
    #
