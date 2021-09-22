"""
Just support FFNN
"""
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential

from src import utils
from src.model_loader import initialize_dnn_model
from src.utils import keras_layer, keras_activation, utilities
import numpy as np

nameVars = set()

def weight(layer_idx, starting_neuron, target_neuron):
    if layer_idx < 0:
        return f"w_f{starting_neuron}_to_n{target_neuron}"
    else:
        return f"w_l{layer_idx}_n{starting_neuron}_to_n{target_neuron}"

def get_feature_name(feature_idx):
    return f"f{feature_idx}"

def get_neuron_name(layer_idex, neuron_idx, model):
    presoftmax_index = get_presoftmax_index(model)
    if layer_idex < 0:
        return get_feature_name(neuron_idx)
    elif layer_idex == 0 and keras_layer.is_inputlayer(model.layers[0]):
        return get_feature_name(neuron_idx)
    elif layer_idex == presoftmax_index:
        return f"n_presoftmax_n{neuron_idx}"
    else:
        return f"n_l{layer_idex}_n{neuron_idx}"


def generate_weights(model: Sequential):
    output = ""
    for idx, current_layer in enumerate(model.layers):
        print("")
        print(f"layer {current_layer.name}")
        print(f"layer {current_layer}")
        weights = current_layer.get_weights()  # the second are biases, the first are weights between two layers
        if keras_layer.is_dense(current_layer):
            kernel = weights[0]
            biases = weights[1]

            print(f"#neuron in pre layer = {kernel.shape[0]}")
            print(f"#neuron in next layer = {kernel.shape[1]}")
            print(f"#neuron in bias = {len(biases)}")
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    output += f"double  {weight(idx - 1, i, j)} = {kernel[i][j]};"
                    output += "\n"

    return output


def generate_prototype(inputlayer):
    output = "/*Feature must be an integer.*/\n"
    output += "void predict("
    weights = inputlayer.get_weights()  # the second are biases, the first are weights between two layers
    n_features = len(weights[0])
    for i in range(n_features):
        output += f"double {get_feature_name(i)}"
        if i != n_features - 1:
            output += ", "
    output += ")"
    return output


def handle_relu_layer(layer, layer_idx, model):
    output = ""
    n_neurons = layer.input_shape[1]
    for j in range(n_neurons):
        name = get_neuron_name(layer_idex=layer_idx, neuron_idx=j, model=model)
        nameVars.add(name)
        output += f"double {name};\t"
        output += f"if ({get_neuron_name(layer_idex=layer_idx - 1, neuron_idx=j, model=model)} < 0)\t"
        output += "{"
        output += f"{name} = 0;"
        output += "} else {"
        output += f"{name} = {get_neuron_name(layer_idex=layer_idx - 1, neuron_idx=j, model=model)};"
        output += "}\n"
    return output


def handle_softmax_layer(layer, i, model):  # just print pre-softmax value
    output = ""
    n_neurons = layer.input_shape[1]
    # output += "double max;\n";
    #
    # output += f"max = {get_neuron_name(layer_idex=i - 1, neuron_idx=0, ispreSoftmaxLayer=True)};"
    # output += "\n"
    # for j in range(1, n_neurons):
    #     output += f"if (max < {get_neuron_name(layer_idex=i - 1, neuron_idx=j, ispreSoftmaxLayer=True)})";
    #     output += "{"
    #     output += f" max = {get_neuron_name(layer_idex=i - 1, neuron_idx=j, ispreSoftmaxLayer=True)}; "
    #     output += "}\n"

    output += "\n"
    for j in range(n_neurons - 1):
        name = get_neuron_name(layer_idex=i-1, neuron_idx=j, model=model)
        nameVars.add(name)
        # output += f"printf(\"prob of class {j} = %.6f  \\n \", {name});\n"
    return output

def print_all_vars():
    output = ""
    for nameVar in nameVars:
        output += f"printf(\"{nameVar} = %.8f  \\n \", {nameVar});\n"
    return output

def handle_dense_layer(layer, layer_idx, model):
    output = ""
    weights = layer.get_weights()
    kernel = weights[0]
    biases = weights[1]

    for j in range(kernel.shape[1]):
        name = get_neuron_name(layer_idex=layer_idx, neuron_idx=j, model=model)
        nameVars.add(name)
        output += f"double {name}; " \
                  f"{name} = "
        for s in range(kernel.shape[0]):
            output += f"{get_neuron_name(layer_idex=layer_idx - 1, neuron_idx=s, model=model)} * " \
                      f"{kernel[s][j]} + "
        output += f"{biases[j]};"
        output += "\n"
    return output


def normalize_features(inputlayer):
    output = ""
    weights = inputlayer.get_weights()  # the second are biases, the first are weights between two layers
    n_features = len(weights[0])
    for i in range(n_features):
        output += f"{get_feature_name(i)} = {get_feature_name(i)} * {1 / 255} + 0; "
    return output


def assume(inputlayer, input_image, delta, model):
    output = ""
    weights = inputlayer.get_weights()  # the second are biases, the first are weights between two layers
    n_features = len(weights[0])
    for i in range(n_features):
        lower = 0 if input_image[i] - delta < 0 else input_image[i] - delta
        upper = 255 if input_image[i] + delta > 255 else input_image[i] + delta
        output += f"klee_assume({get_neuron_name(0, i, model=model)} >= {int(lower)}); "  # not &&
        output += f"klee_assume({get_neuron_name(0, i, model=model)} <= {int(upper)}); "  # not &&

        # output += f"klee_prefer_cex({neuron(-1, i)}, {lower} <= {neuron(-1, i)} && {neuron(-1, i)} <= {upper}); "
    return output


def makesymbolic(input_image, model):
    output = ""
    for idx in range(len(input_image)):
        output += f"klee_make_symbolic( &{get_neuron_name(0, idx, model=model)}, sizeof({get_neuron_name(0, idx, model=model)}), \"{get_neuron_name(0, idx, model=model)}\"); "
    return output


def get_presoftmax_index(model):
    for idx, current_layer in enumerate(model.layers):
        if keras_activation.is_softmax(current_layer):
            return idx - 1


if __name__ == '__main__':
    model_object = initialize_dnn_model()  # mnist_ann_keras
    print(model_object.get_model().summary())
    model = model_object.get_model()
    inputlayer = model.layers[0]

    # y_pred = np.argmax(model.predict(model_object.get_Xtrain()[:100]), axis=1);
    # np.where(np.argmax(model.predict(model_object.get_Xtrain()), axis=1) != model_object.get_ytrain())
    '''
    add header
    '''
    code = "#include \"stdio.h\""
    code += "\n"
    # print(generate_weights(model))
    '''
    generate predict()
    '''
    code += generate_prototype(inputlayer) + "\n"
    code += "{\n"
    code += normalize_features(inputlayer) + "\n"

    for idx, current_layer in enumerate(model.layers):
        if keras_layer.is_inputlayer(current_layer):
            continue;

        elif keras_layer.is_dense(current_layer):
            code += handle_dense_layer(model.layers[idx], idx, model)
            code += "\n"
        elif keras_layer.is_activation(current_layer):
            if keras_activation.is_relu(current_layer):
                code += handle_relu_layer(model.layers[idx], idx, model)
                code += "\n"
            elif keras_activation.is_softmax(current_layer):
                code += handle_softmax_layer(model.layers[idx], idx, model)
                code += "\n"

    code += print_all_vars();
    code += "}\n\n"

    '''
    declaration
    '''
    code += "int main(){\n"
    N_FEATURES = 784
    for idx in range(N_FEATURES):
        code += f"double {get_feature_name(idx)}; "
    code += "\n";

    '''
    ifdef
    '''
    num_analyzed_seeds = 100  # len(model_object.get_Xtrain())
    code += f"#ifdef seed0\n"
    for seed_idx in range(0, num_analyzed_seeds):
        input_image = model_object.get_Xtrain()[seed_idx].reshape(-1)
        for idx in range(len(input_image)):
            code += f"\t{get_feature_name(idx)} = {np.round(255 * input_image[idx], 0)}; "

        # just for logging
        code += f"\n/*true label: {model_object.get_ytrain()[seed_idx]}*/\n"
        presoftmax = -2
        intermediate_layer_model = Model(inputs=model_object.get_model().inputs,
                                         outputs=model_object.get_model().layers[presoftmax].output)
        presoftmax_str = intermediate_layer_model.predict(input_image.reshape(-1, 784))
        code += f"/*presoftmax: {presoftmax_str}\n*/"

        if (seed_idx != num_analyzed_seeds - 1):
            code += f"\n#elif seed{seed_idx + 1}\n"
        else:
            code += "\n#endif\n"

    '''
    make symbolic
    '''
    # code += makesymbolic(input_image) + "\n"
    # code += assume(inputlayer, np.round(input_image * 255), delta=10) + "\n"

    '''
    call predict()
    '''
    code += "\n"
    code += "predict("
    for idx in range(len(input_image)):
        code += f"{get_feature_name(idx)}"
        if idx != len(input_image) - 1:
            code += ", "
    code += ");\n"  # end of predict call
    code += "return 0;\n"
    code += "}"

    '''
    export to file
    '''
    f = open("/Users/ducanhnguyen/Documents/ImprovedDeepFault/test/ccodeexam.c", "w")
    f.write(code)
    f.close()

    # utilities.show_two_images(input_image.reshape(28, 28), input_image.reshape(28, 28), display=True)

    # for checking
    # intermediate_layer_model = Model(inputs=model.inputs,
    #                                  outputs=model.layers[-2].output)
    # neurons = intermediate_layer_model.predict(input_image.reshape(-1, 784))
    # print(neurons)
