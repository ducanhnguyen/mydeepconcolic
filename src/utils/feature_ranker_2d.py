import enum
import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

global logger
logger = logging.getLogger()
from src.utils.mylogger import MyLogger

logger = MyLogger.getLog()


class RANKING_ALGORITHM(enum.Enum):
    ABS = 1
    COI = 2
    CO = 3


class feature_ranker:
    def __init__(self):
        return

    @staticmethod
    def compute_gradient_wrt_features(input: tf.Tensor,
                                      target_neuron: int,
                                      classifier: tf.keras.Sequential):
        """Compute gradient wrt features.
        Args:
            input: a tensor (shape = `[1, height, width, channel`])
            target_neuron: the index of the neuron on the output layer needed to be differentiated
            classifier: a sequential model
        Returns:
            gradient: ndarray (shape = `[height, width, channel`])
        """
        with tf.GradientTape() as tape:
            tape.watch(input)
            prediction_at_target_neuron = classifier(input)[0][target_neuron]
        gradient = tape.gradient(prediction_at_target_neuron, input)
        gradient = gradient.numpy()[0]
        return gradient

    @staticmethod
    def find_important_features_of_a_sample(input_image: np.ndarray,
                                            n_rows: int,
                                            n_cols: int, n_channels: int, n_important_features: int,
                                            algorithm: enum.Enum,
                                            gradient_label: int,
                                            classifier: keras.Sequential):
        """Apply algorithm to find the most important features.
        Args:
            n_rows: a positive number
            n_cols: a positive number
            n_channels: a positive number
            n_important_features: a positive number
            input_image: shape = `[height, width, channel`]
            algorithm:
            classifier:
            gradient_label: any label
        Returns:
            positions: ndarray (shape=`[row, col, channel`])
        """
        input_image = input_image.copy()  # avoid modifying on the original one
        gradient = feature_ranker.compute_gradient_wrt_features(
            input=tf.convert_to_tensor([input_image]),
            target_neuron=gradient_label,
            classifier=classifier)
        feature_ranking = []

        # find the position of the highest value in the gradient
        score = []
        for rdx in range(0, n_rows):
            for cdx in range(0, n_cols):
                for chdx in range(0, n_channels):
                    feature_ranking.append([rdx, cdx, chdx])

                    if algorithm == RANKING_ALGORITHM.ABS:
                        grad = np.abs(gradient[rdx, cdx, chdx])
                        score.append(grad)

                    elif algorithm == RANKING_ALGORITHM.CO:
                        grad = gradient[rdx, cdx, chdx]
                        score.append(grad)

                    elif algorithm == RANKING_ALGORITHM.COI:
                        feature_value = input_image[rdx, cdx, chdx]
                        grad = gradient[rdx, cdx, chdx]
                        score.append(grad * feature_value)

        feature_ranking = np.asarray(feature_ranking)
        score = np.asarray(score)

        index_arr = np.arange(0, n_rows * n_rows * n_channels, 1)
        score_sort, index_arr = zip(*sorted(zip(score, index_arr), reverse=True))
        feature_ranking_sorted = []
        for idx in index_arr:
            feature_ranking_sorted.append(feature_ranking[idx])
        feature_ranking_sorted = np.asarray(feature_ranking_sorted)

        if n_important_features is not None:
            return np.asarray(feature_ranking_sorted[:n_important_features])
        else:
            return feature_ranking_sorted

    @staticmethod
    def find_important_features_of_samples(input_images: np.ndarray,
                                           n_rows: int, n_cols: int, n_channels: int, n_important_features: int,
                                           algorithm: enum.Enum,
                                           gradient_label: int,
                                           classifier: keras.Sequential):
        """Apply ranking algorithm to find the most important features.
        Args:
            input_images: an array of samples, `[size, height, width, channel`]
            n_rows: a positive number
            n_cols: a positive number
            n_channels: a positive number
            n_important_features: a positive number
            algorithm:
        Returns:
            positions: ndarray (shape=`[row, col, channel`])
        """
        final_important_features = np.ndarray(shape=(1, 3), dtype=int)
        for index in range(0, len(input_images)):
            input_image = input_images[index]
            important_features = feature_ranker.find_important_features_of_a_sample(
                n_rows=n_rows,
                n_cols=n_cols,
                n_channels=n_channels,
                input_image=input_image,
                n_important_features=n_important_features,
                algorithm=algorithm,
                gradient_label=gradient_label,
                classifier=classifier)
            final_important_features = np.concatenate(
                (final_important_features, important_features),
                axis=0)

        final_important_features = np.delete(
            arr=final_important_features,
            obj=0,
            axis=0)
        final_important_features = np.unique(final_important_features, axis=0)
        return final_important_features

    @staticmethod
    def highlight_important_features(important_features: np.ndarray, input_image: np.ndarray):
        """Highlight important features
            :param important_features: shape = '[ row, col, channel']. Each row stores the position of its feature on input image
            :param input_image: shape = `[ height, width, channel`]
        :return: None
        """
        input_image = input_image.copy()
        max = np.max(input_image)
        ROW_IDX = 0
        COL_IDX = 1
        CHANNEL_INDEX = 2
        for idx in range(0, important_features.shape[0]):
            row = important_features[idx, ROW_IDX]
            col = important_features[idx, COL_IDX]
            channel = important_features[idx, CHANNEL_INDEX]
            input_image[row, col, channel] = max + 2
        plt.imshow(input_image, cmap='gray')
        plt.title("Most important features are highlighted")
        plt.show()


if __name__ == '__main__':
    ATTACKED_MODEL_H5 = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/model/Alexnet.h5"
    dnn = keras.models.load_model(filepath=ATTACKED_MODEL_H5, compile=False)

    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    X_train = X_train / 255

    input_image = X_train[0].reshape(28, 28, 1)
    most_important_features = feature_ranker.find_important_features_of_a_sample(
        input_image=input_image,
        n_rows=28,
        n_cols=28,
        n_channels=1,
        n_important_features=100,
        algorithm=RANKING_ALGORITHM.COI,
        gradient_label=1,
        classifier=dnn)
    feature_ranker.highlight_important_features(
        np.asarray(most_important_features),
        input_image)
