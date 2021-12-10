
from tensorflow.python.keras.layers import Activation


def is_activation(obj):
    assert (obj != None)
    return isinstance(obj, Activation)

def is_relu(obj):
    assert (obj != None)
    if isinstance(obj, Activation):
        if str(obj.get_config()['activation']).lower().find('relu') != -1:
            return True
        else:
            return False
    else:
        # should not be this case
        return False

def is_tanh(obj):
    assert (obj != None)
    if isinstance(obj, Activation):
        if str(obj.get_config()['activation']).lower().find('tanh') != -1:
            return True
        else:
            return False
    else:
        # should not be this case
        return False

def is_softmax(obj):
    assert (obj != None)
    if isinstance(obj, Activation):
        if str(obj.get_config()['activation']).lower().find('softmax') != -1:
            return True
        else:
            return False
    else:
        # should not be this case
        return False

# TODO: add more activations