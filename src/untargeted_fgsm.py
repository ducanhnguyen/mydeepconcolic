from __future__ import absolute_import
import platform
import tensorflow as tf
import numpy as np
import csv

from src.config_parser import get_config
from src.model_loader import initialize_dnn_model_from_name, initialize_dnn_model
from src.utils import utilities
from src.utils.feature_ranker1d import feature_ranker1d
from src.utils.mylogger import MyLogger
import os

logger = MyLogger.getLog()

MNIST_NUM_CLASSES = 10


class UNTARGETED_FGSM:

    @staticmethod
    def create_adversaries(x_1D: np.ndarray,
                           true_label: int,
                           classifier: tf.keras.Sequential,
                           epsilon: float):
        gradient_1D = feature_ranker1d.compute_gradient_wrt_features(
            tf.convert_to_tensor(x_1D.reshape(-1, 784)),
            true_label,
            classifier
        )
        sign_1D = np.sign(gradient_1D)
        adv_1D = x_1D + epsilon * sign_1D
        return adv_1D


if __name__ == '__main__':
    # if platform.system() == 'Darwin':  # macosx
    #     base = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result/"
    # elif platform.system() == 'Linux':  # hpc
    #     base = f"/home/anhnd/mydeepconcolic/result/"
    #
    # summary = f"{base}/summary.csv"
    # if os.path.exists(summary):
    #     os.remove(summary)
    #
    # model_object = initialize_dnn_model_from_name("mnist_ann_keras")
    # classifier = model_object.get_model()

    logger.debug("initialize_dnn_model")
    model_object = initialize_dnn_model()
    classifier = model_object.get_model()

    out = get_config(["output_folder"])
    if not os.path.exists(out):
        os.makedirs(out)
    summary = f"{out}/summary.csv"
    logger.debug(f"summary = {summary}")

    #########
    with open(summary, mode='w') as f:
        seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        seed.writerow(['seed', 'l0', 'l2', 'l_inf', 'minimum_change', 'true_label', 'adv_label', 'epsilon',
                       'position_adv_label_in_original_pred', 'first_largest',
                       'second_largest', 'third_largest', 'fourth_largest', 'fifth_largest',
                       'last_largest'])

    for SEED_IDX in range(0, 10000):
        logger.debug(f"Seed = {SEED_IDX}")
        x_1D = model_object.get_Xtrain()[SEED_IDX]
        true_label = model_object.get_ytrain()[SEED_IDX]
        pred_label = np.argmax(model_object.get_model().predict(x_1D.reshape(-1, 784)))
        if pred_label != true_label:  # wrong predicted samples
            logger.debug(f"Ignore seed {SEED_IDX}")
            continue

        for epsilon in np.arange(start=0, stop=0.03, step=0.001):
            # print(f'epsilon = {epsilon}')
            adv_1D = UNTARGETED_FGSM.create_adversaries(x_1D=x_1D,
                                                        classifier=classifier,
                                                        epsilon=epsilon,
                                                        true_label=true_label)
            adv_1D = np.clip(adv_1D, 0, 1)
            pred = model_object.get_model().predict(adv_1D.reshape(-1, 784))
            # print(f'pred = {pred}')

            adv_label = np.argmax(pred)
            if true_label != adv_label:
                logger.debug("Found")
                l0 = utilities.compute_l0(x_1D, adv_1D)
                l2 = utilities.compute_l2(x_1D, adv_1D)
                linf = utilities.compute_linf(x_1D, adv_1D)
                minimum_change = utilities.compute_minimum_change(x_1D, adv_1D)

                utilities.show_two_images(x_1D.reshape(28, 28),
                                          adv_1D.reshape(28, 28),
                                          left_title=f"idx = {SEED_IDX}: true label = {true_label}",
                                          right_title=f"pred label = {adv_label}, epsilon = {epsilon},\nl0 = {l0}\nl2 = {l2}",
                                          path=f"{out}/{SEED_IDX}.png",
                                          display=False)

                with open(summary, mode='a') as f:
                    seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    seed.writerow(
                        [SEED_IDX, l0, l2, linf, minimum_change, true_label, adv_label, epsilon, "", "", "", "", "", "",
                         ""])
                break
