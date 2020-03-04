import tensorflow as tf
import numpy as np


def example_1():
    var = tf.Variable(1)
    add = tf.assign(ref=var, value=2)

    with tf.Session() as sess:
        sess.run(add)
        print(f"{var} = {var.eval()}")


def example_2():
    var = tf.Variable(shape=(10, 1), initial_value=np.zeros(shape=(10, 1)))

    with tf.Session() as sess:
        add = tf.assign(ref=var, value=np.ones(shape=(10, 1)))
        sess.run(add)
        print(f"{var} = {var.eval()}")

if __name__ == '__main__':
    example_2()