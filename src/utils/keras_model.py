from tensorflow.python.keras.models import Sequential

from src.utils import keras_layer
from src.utils.covered_layer import covered_layer


def is_ANN(model):
    '''
    Check the model is ANN
    :param model:
    :return:
    '''
    assert (isinstance(model, Sequential))
    DENSE_SHAPE = 2  # (None, n_hidden_units)
    if len(model.input_shape) == DENSE_SHAPE:
        return True
    else:
        return False


def is_CNN(model):
    '''
    Check the model is CNN
    :param model:
    :return:
    '''
    assert (isinstance(model, Sequential))
    CONVOLUTION_SHAPE = 4  # (None, width, height, filter)
    if len(model.input_shape) == CONVOLUTION_SHAPE:
        return True
    else:
        return False


def get_dense_layers(model):
    dense_layers = []
    indexes = []

    if isinstance(model, Sequential):
        for idx, layer in enumerate(model.layers):
            if keras_layer.is_dense(layer):
                dense_layers.append(layer)
                indexes.append(idx)

    return zip(dense_layers, indexes)


def get_convolution_layers(model):
    assert (isinstance(model, Sequential))
    convolution_layers = []
    indexes = []

    if isinstance(model, Sequential):
        for idx, layer in enumerate(model.layers):
            if keras_layer.is_2dconv(layer):
                convolution_layers.append(layer)
                indexes.append(idx)

    return zip(convolution_layers, indexes)


def get_considered_layers(model):
    assert (isinstance(model, Sequential))
    considered_layers = []

    if isinstance(model, Sequential):
        dense_layers = get_dense_layers(model)
        convolution_layers = get_convolution_layers(model)

        for layer_pair in dense_layers:
            c = covered_layer()
            c.set_index(layer_pair[1])
            c.set_layer(layer_pair[0])
            considered_layers.append(c)

        for layer_pair in convolution_layers:
            c = covered_layer()
            c.set_index(layer_pair[1])
            c.set_layer(layer_pair[0])
            considered_layers.append(c)
    else:
        print(f'model must be instance of keras.engine.sequential.Sequential')

    return considered_layers
