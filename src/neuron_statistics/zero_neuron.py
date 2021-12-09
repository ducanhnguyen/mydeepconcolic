"""
1. Count the number of zero neurons under a set of inputs
2. Print a test path when executing an input
"""
from src.model_loader import initialize_dnn_model
from keras.models import Model
from src.utils import keras_layer


def get_name_neuron(layer, idx):
    return f"{layer}-idex_{idx}"


if __name__ == '__main__':
    model_object = initialize_dnn_model()  # mnist_ann_keras
    print(model_object.get_model().summary())

    non_zeros_dict = dict()
    n_zero_arr = []
    n_seeds = 1  # len(model_object.get_Xtrain())
    tp_statistics = dict()

    for idx in range(n_seeds):
        testpath = ""
        print(f"Handle seed {idx}")
        input_image = model_object.get_Xtrain()[idx]
        Y_pred = model_object.get_model().predict(input_image.reshape(-1, 784))  # for MNIST: shape = (N, 10)

        n_zero_neurons = 0
        n_neurons = 0
        presoftmax = -2

        for layer in model_object.get_model().layers[:presoftmax]:
            if keras_layer.is_activation(layer):
                testpath += f"[{layer.name}]"
                intermediate_layer_model = Model(inputs=model_object.get_model().inputs,
                                                 outputs=layer.output)
                neurons = intermediate_layer_model.predict(input_image.reshape(-1, 784))

                index = -1
                print("\n")
                for neuron in neurons[0]:
                    print("value " + str(neuron))
                    index += 1
                    n_neurons += 1
                    testpath += f".{index}"

                    if neuron == 0:
                        testpath += f"(zero)"
                        key = get_name_neuron(layer=layer.name, idx=index)
                        if key in non_zeros_dict:
                            non_zeros_dict[key] += 1
                        else:
                            non_zeros_dict[key] = 1
                        n_zero_neurons += 1
                print(f"layer {layer.name}: #n zeros = {n_zero_neurons}, #neuron = {neurons.shape[1]}")
            else:
                print(f'IGNORE layer {layer.name}')
        if testpath in tp_statistics:
            tp_statistics[testpath] += 1
        else:
            tp_statistics[testpath] = 1

        n_zero_arr.append(n_zero_neurons)

    # print(np.mean(n_zero_arr))
    # for key, num_of_zeros in non_zeros_dict.items():
    #     print(f"{key}:\t{num_of_zeros}")
    #
    # for key, num_of_executions in tp_statistics.items():
    #     print(f"{num_of_executions}: {key}")

    intermediate_layer_model = Model(inputs=model_object.get_model().inputs,
                                     outputs=model_object.get_model().layers[-2].output)
    neurons = intermediate_layer_model.predict(input_image.reshape(-1, 784))
    print(neurons)