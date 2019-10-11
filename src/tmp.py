# TODO: Neu model khong co input layer, ma co Dense, ta nen them rang buoc cho input layer nuaf
import keras
import numpy  as np

from src.utils import keras_layer, keras_model
from src.utils.covered_layer import covered_layer

MINUS_INF = -10000000
INF = 10000000

def create_bound_of_neuron_constraints(acts, idxes):
    for act, layer_index in zip(acts, idxes):
        shape = act.shape
        CONVOLUTION_SHAPE = 3
        DENSE_SHAPE = 1

        my_ub = []
        my_lb = []
        my_colnames = []
        neuron_constraints = []

        if len(act.shape) == CONVOLUTION_SHAPE:
            WIDTH_INDEX = 0
            HEIGHT_INDEX = 1
            CHANNEL_OUT_INDEX = 2

            for i in range(shape[WIDTH_INDEX]):
                for j in range(shape[HEIGHT_INDEX]):
                    for k in range(shape[CHANNEL_OUT_INDEX]):
                        new_var = 'x_{0}_{1}_{2}_{3}'.format(layer_index, i, j, k)
                        my_colnames.append(new_var)
                        my_lb.append(MINUS_INF)
                        my_ub.append(INF)
                        neuron_constraint = f'{MINUS_INF} <= {new_var} <= {INF}'
                        neuron_constraints.append(neuron_constraint)

        elif len(act.shape) == DENSE_SHAPE:
            N_HIDDEN_UNITS_INDEX = 0
            for i in range(shape[N_HIDDEN_UNITS_INDEX]):
                new_var = 'x_{0}_{1}'.format(layer_index, i)
                my_colnames.append(new_var)
                my_lb.append(MINUS_INF)
                my_ub.append(INF)
                neuron_constraint = f'{MINUS_INF} <= {new_var} <= {INF}'
                neuron_constraints.append(neuron_constraint)

    return neuron_constraints

def get_shape_of_considered_layers(model):
    acts = []
    idxes = []

    if isinstance(model, keras.engine.sequential.Sequential):
        # TODO: ignore the input layer and output layer
        considered_layers = keras_model.get_considered_layers(model)
        print(f'considered_layers = {considered_layers}')

        for considered_layer in considered_layers:
            if isinstance(considered_layer, covered_layer):
                idx = considered_layer.get_index()
                idxes.append(idx)
                layer = considered_layer.get_layer()

                if keras_layer.is_2dconv(layer):
                    # instance of keras.layers.convolutional.Conv2D
                    output = layer.output_shape
                    WIDTH_INDEX = 1
                    HEIGHT_INDEX = 2
                    CHANNEL_OUT_INDEX = 3
                    act = np.zeros(shape=(output[WIDTH_INDEX], output[HEIGHT_INDEX], output[CHANNEL_OUT_INDEX]))
                    acts.append(act)

                elif keras_layer.is_dense(layer):
                    # instance of keras.layers.core.Dense
                    output_shape = layer.get_config()['units']
                    act = np.zeros(shape=(output_shape,))
                    acts.append(act)

                else:
                    print(f'Do not support {considered_layer.get_layer()}')
    else:
        print(f'model must be instance of keras.engine.sequential.Sequential')

    return acts, idxes
