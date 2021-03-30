import enum
import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.model_loader import initialize_dnn_model
from src.utils.feature_ranker_2d import RANKING_ALGORITHM

global logger
logger = logging.getLogger()
from src.utils.mylogger import MyLogger

logger = MyLogger.getLog()


class feature_ranker1d:
    def __init__(self):
        return

    @staticmethod
    def compute_gradient_wrt_features(input: tf.Tensor,
                                      target_neuron: int,
                                      classifier: tf.keras.Sequential):
        """Compute gradient wrt features.
        Args:
            input: a tensor
            target_neuron: the index of the neuron on the output layer needed to be differentiated
            classifier: a sequential model
            n_pixels: for mnist = 784
        Returns:
            gradient: ndarray
        """
        with tf.GradientTape() as tape:
            tape.watch(input)
            prediction_at_target_neuron = classifier(input)[0][target_neuron]
        gradient = tape.gradient(prediction_at_target_neuron, input)
        gradient = gradient.numpy()[0]
        return gradient

    @staticmethod
    def find_important_features_of_a_sample(input_image: np.ndarray,
                                            n_important_features: int,
                                            algorithm: enum.Enum,
                                            gradient_label: int,
                                            classifier: keras.Sequential):
        """Apply ranking algorithm to find the most important features with 1-D input vector.
        Args:
            input_image: 2-D input vector. For MNIST: (-1, 784)
            n_important_features: a positive number
            algorithm:
            classifier: a deed-forward neural network model
            gradient_label: any label
        Returns:
            positions: 1-D array. Each element is the position of an important pixel.
        """
        input_image = input_image.copy()  # avoid modifying on the original one
        n_pixels = input_image.shape[1]

        gradient = feature_ranker1d.compute_gradient_wrt_features(
            input=tf.convert_to_tensor(input_image),
            target_neuron=gradient_label,
            classifier=classifier)

        # find the position of the highest value in the gradient
        index_arr = np.arange(0, n_pixels)  # 1-D array
        gradient = gradient.reshape(n_pixels)  # 1-D array
        if algorithm == RANKING_ALGORITHM.ABS:
            gradient, index_arr = zip(*sorted(zip(np.abs(gradient), index_arr), reverse=True))
            index_arr = index_arr[:n_important_features]

        elif algorithm == RANKING_ALGORITHM.CO:
            gradient, index_arr = zip(*sorted(zip(gradient, index_arr), reverse=True))
            index_arr = index_arr[:n_important_features]

        elif algorithm == RANKING_ALGORITHM.COI:
            input_image = input_image.reshape(n_pixels)
            gradient, index_arr = zip(*sorted(zip(gradient * input_image, index_arr), reverse=True))
            index_arr = index_arr[:n_important_features]

        else:
            index_arr = None

        return index_arr

    @staticmethod
    def highlight_important_features(important_features: np.ndarray, input_image: np.ndarray, shape):
        """Highlight important features
            :param important_features:
            :param input_image: for Mnist: (1, 784)
            :param num_pixel: The number of pixels. For MNIST: 784
        :return: None
        """
        input_image = input_image.reshape(input_image.shape[1]).copy()
        max = np.max(input_image)
        for important_feature in important_features:
            input_image[important_feature] = max + 2
        plt.imshow(input_image.reshape(shape), cmap='gray')
        plt.title("Most important features are highlighted")
        plt.show()


if __name__ == '__main__':
    logging.basicConfig()
    logging.root.setLevel(logging.DEBUG)

    model_object = initialize_dnn_model()

    input_image = model_object.get_Xtrain()[0].reshape(-1, 784)
    important_features = feature_ranker1d.find_important_features_of_a_sample(
        input_image=input_image,
        n_important_features=100,
        algorithm=RANKING_ALGORITHM.COI,
        gradient_label=5,
        classifier=model_object.get_model())

    feature_ranker1d.highlight_important_features(
        important_features=important_features,
        input_image=input_image,
        shape=(28, 28))
