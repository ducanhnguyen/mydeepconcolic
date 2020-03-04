'''
Example of categorical_crossentropy
'''
import keras.backend as K
import tensorflow as tf

target = tf.placeholder(shape=(3, ), dtype=float)
output = tf.placeholder(shape=(3, ), dtype=float)
ce = K.mean(K.categorical_crossentropy(target, output))
print(ce)

with tf.Session() as session:
    entropy = session.run(ce, feed_dict={target:[1, 0, 0], output:[0.3, 0.4, 0.3]})
    print(entropy)