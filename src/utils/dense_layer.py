from src.utils import keras_activation, keras_layer


def is_dense_and_activity(layer):
    assert (layer != None)
    if keras_layer.is_dense(layer) and keras_activation.is_activation(layer):
        return True

def get_n_hidden_units(layer):
    assert (layer != None)
    if keras_layer.is_dense(layer):
        weights = layer.get_weights()  # the second are biases, the first are weights between two layers
        kernel = weights[0]
        hidden_units_pre_layer = kernel.shape[0]
        hidden_units_curr_layer = kernel.shape[1]
        return hidden_units_pre_layer, hidden_units_curr_layer
    else:
        return None
