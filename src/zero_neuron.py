from src.model_loader import initialize_dnn_model
from src.saved_models.mnist_deepcheck import MNIST_DEEPCHECK
from keras.models import Model
import numpy as np

from src.utils import keras_layer

if __name__ == '__main__':
    model_object = initialize_dnn_model()
    print(model_object.get_model().summary())

    n_zero_arr = []
    for idx in range(1):
        input_image = model_object.get_Xtrain()[idx]
        Y_pred = model_object.get_model().predict(input_image.reshape(-1, 784))  # for MNIST: shape = (N, 10)

        n_zero = 0
        n_total = 0
        presoftmax = -2
        for layer in model_object.get_model().layers[:presoftmax]:
            if keras_layer.is_activation(layer):
                intermediate_layer_model = Model(inputs=model_object.get_model().inputs,
                                                 outputs=layer.output)
                neurons = intermediate_layer_model.predict(input_image.reshape(-1, 784))

                for neuron in neurons[0]:
                    n_total += 1
                    if neuron == 0:
                        n_zero += 1
                print(f"layer {layer.name}: #n zeros = {n_zero}, #neuron = {neurons.shape[1]}")
            else:
                print(f'Ignore layer {layer.name}')

        n_zero_arr.append(n_zero)
        print(f'idx = {idx}, # zero neurons = {n_zero}, # total neurons = {n_total}')

    print(np.mean(n_zero_arr))
