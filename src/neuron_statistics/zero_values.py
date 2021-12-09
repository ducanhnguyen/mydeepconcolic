"""
Print neuron values
"""
from src.model_loader import initialize_dnn_model, initialize_dnn_model_from_name
from keras.models import Model
from src.utils import keras_layer


def get_name_neuron(layer, idx):
    return f"{layer}-idex_{idx}"


if __name__ == '__main__':
    SEED_IDX = 0
    model_object = initialize_dnn_model_from_name("mnist_simple")
    print(model_object.get_model().summary())

    testpath = ""
    print(f"Handle seed {SEED_IDX}")
    input_image = model_object.get_Xtrain()[SEED_IDX]
    # Y_pred = model_object.get_model().predict(input_image.reshape(-1, 784))  # for MNIST: shape = (N, 10)

    for layer in model_object.get_model().layers:
        print(f'\nlayer {layer}')
        # if keras_layer.is_activation(layer):
        testpath += f"[{layer.name}]"
        intermediate_layer_model = Model(inputs=model_object.get_model().inputs,
                                         outputs=layer.output)
        neurons = intermediate_layer_model.predict(input_image.reshape(-1, 784))

        index = 1
        for neuron in neurons[0]:
            print(f"neuron {index}: " + str(neuron))
            index += 1