from src.c_code.convert_c_code import get_neuron_name
from src.model_loader import initialize_dnn_model
from keras.models import Model
from src.utils import keras_layer, keras_activation
import numpy as np
import csv


def get_name_neuron(layer, idx):
    return f"{layer}-idex_{idx}"


if __name__ == '__main__':
    model_object = initialize_dnn_model()  # mnist_ann_keras
    model = model_object.get_model()
    model.summary()
    num_relu_neurons = keras_layer.get_num_relu_neurons(model)
    print(f"num_relu_neurons = {num_relu_neurons}")

    # the first one to the last: attr_as_n, attr_af_n, attr_ns_n, attr_nf_n
    attr_as_n = 0
    attr_af_n = 1
    attr_ns_n = 2
    attr_nf_n = 3
    statistics = np.zeros(shape=(num_relu_neurons, 4))

    n_seeds = None
    Xtrain = model_object.get_Xtrain()[:n_seeds].reshape(-1, 784)
    Ypred = model.predict(Xtrain)
    ypred = np.argmax(Ypred, axis=1)
    ytrue = model_object.get_ytrain()[:n_seeds]
    correctly_classified = (ytrue == ypred)
    print(np.sum(correctly_classified))
    presoftmax = -2

    neuron_dict = dict()
    neuron_idx = -1
    layer_idx = -2 # input layer has index -1
    for layer in model_object.get_model().layers[:presoftmax]:
        layer_idx += 1
        if keras_layer.is_activation(layer) and keras_activation.is_relu(layer):
            intermediate_layer_model = Model(inputs=model_object.get_model().inputs,
                                             outputs=layer.output)
            neurons = intermediate_layer_model.predict(Xtrain)
            num_neuron = keras_layer.get_number_of_units(model, layer_idx)

            for jdx in range(num_neuron):
                neuron_idx += 1

                # save name of neuron
                name_neuron = get_neuron_name(layer_idx, neuron_idx)
                if neuron_idx not in neuron_dict:
                    neuron_dict[neuron_idx] = name_neuron

                for seed_idx in range(neurons.shape[0]):
                    # compute score
                    neuron = neurons[seed_idx][jdx]
                    if neuron == 0:
                        if correctly_classified[seed_idx]:
                            statistics[neuron_idx][ attr_ns_n] += 1
                        else:
                            statistics[neuron_idx][ attr_nf_n] += 1
                    else:
                        if correctly_classified[seed_idx]:
                            statistics[neuron_idx][attr_as_n] += 1
                        else:
                            statistics[neuron_idx][attr_af_n] += 1

    suspicious_file = "/Users/ducanhnguyen/Documents/akautauto_2020/datatest/resultoftest/suspicious_neuron.csv"
    with open(suspicious_file, mode='w') as f:
        seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        seed.writerow(["neuron", "name", "attr_as_n", "attr_af_n", "attr_ns_n", "attr_nf_n", "Tarantula score", "Ochiai score", "D3 score"])

        for idx in range(len(statistics)):
            af_n = statistics[idx][attr_af_n]
            nf_n = statistics[idx][attr_nf_n]
            as_n = statistics[idx][attr_as_n]
            ns_n = statistics[idx][attr_ns_n]
            tarantula_score = af_n / (af_n + nf_n) / (af_n/(af_n+nf_n) + as_n/(as_n + ns_n) )
            ochiai_score = af_n / np.sqrt((af_n + nf_n)*(af_n + as_n))
            d3_score = af_n * af_n * af_n / (as_n + nf_n)

            seed.writerow([idx, neuron_dict[idx], statistics[idx][attr_as_n], statistics[idx][attr_af_n], statistics[idx][attr_ns_n],
                          statistics[idx][attr_nf_n], tarantula_score, ochiai_score, d3_score])
