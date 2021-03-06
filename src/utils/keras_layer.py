import logging

import keras

from src.utils import keras_model

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def is_2dconv(layer_object):
    assert (layer_object!=None)
    if isinstance(layer_object, keras.layers.convolutional.Conv2D):
        return True
    else:
        return False


def is_activation(layer_object):
    assert (layer_object!=None)
    if isinstance(layer_object, keras.layers.core.Activation):
        return True
    else:
        return False


def is_max_pooling(layer_object):
    assert (layer_object != None)
    if isinstance(layer_object, keras.layers.pooling.MaxPooling2D):
        return True
    else:
        return False


def is_dropout(layer_object):
    assert (layer_object != None)
    if isinstance(layer_object, keras.layers.core.Dropout):
        return True
    else:
        return False


def get_number_of_units(model, layer_idx):
    assert (isinstance(model, keras.engine.sequential.Sequential))
    assert (layer_idx>=0)
    units = None

    if keras_model.is_ANN(model):
        layer = model.layers[layer_idx]

        if is_dropout(layer):
            if layer_idx - 1 < 0:
                logger.debug('The first layer can not be dropout')
            elif is_dense(model.layers[layer_idx - 1]):
                units = model.layers[layer_idx - 1].get_config()['units']
            elif is_dense(model.layers[layer_idx - 2]) and \
                    not is_dense(model.layers[layer_idx - 1]):
                units = model.layers[layer_idx - 2].get_config()['units']

        elif is_dense(layer):
            units = layer.get_config()['units']

        elif is_activation(layer) and not is_dense(layer):
            if is_dense(model.layers[layer_idx - 1]):
                units = model.layers[layer_idx - 1].get_config()['units']
            else:
                logger.debug(
                    f'If the current layer is not dense, the previous layer before this layer must be dense!')

        else:
            logger.debug(f'Does not support this kind of last layer')

    elif keras_model.is_CNN(model):
        logger.debug(f'Model is not CNN. Does not support!')

    else:
        logger.debug(f'Unable to detect the type of neural network')
    return units


def is_dense(layer_object):
    assert (layer_object != None)
    if isinstance(layer_object, keras.layers.core.Dense):
        return True
    else:
        return False
