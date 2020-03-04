'''
Use gradient in keras
'''
import keras.backend as K
import tensorflow as tf

# step 1: construct the formula
x = tf.placeholder(dtype=tf.float32)
out = 2*x*x+1

# step 2: construct gradient
grad = K.gradients(loss = out, variables=x) #Gradient of output wrt the input of the model (Tensor)

# step 3: compute gradient under an input
with tf.Session() as sess:
    tf.global_variables_initializer()
    evaluated_gradients_1 = sess.run(grad, feed_dict={x: 2})
    print(f"evaluated_gradients_1 = {evaluated_gradients_1}")
