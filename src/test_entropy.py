import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import clip_ops

def _constant_to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.

  This is slightly faster than the _to_tensor function, at the cost of
  handling fewer cases.

  Arguments:
      x: An object to be converted (numpy arrays, floats, ints and lists of
        them).
      dtype: The destination type.

  Returns:
      A tensor.
  """
  return constant_op.constant(x, dtype=dtype)


a = tf.constant([1., 0., 0., 0., 1., 0., 0., 0., 1.], shape=[3,3])
print("a: ", a)
b = tf.constant([.9, .05, .05, .5, .89, .6, .05, .01, .94], shape=[3,3])
print("b: ", b)
loss = K.categorical_crossentropy(a, b)
print('Loss: ', loss) #Loss: tf.Tensor([0.10536055 0.8046684  0.06187541], shape=(3,), dtype=float32)

# normalize into [0..1]
from tensorflow.python.ops import math_ops
output = b / math_ops.reduce_sum(b, axis = -1, keepdims = True)

# get value of epsilon
epsilon = backend_config.epsilon
epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype) # epsilon = 1e-07

output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)

v = -math_ops.reduce_sum(a * math_ops.log(output), axis = -1)

with tf.Session() as sess:
    print(math_ops.log(2.7).eval())

