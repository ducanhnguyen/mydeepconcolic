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
import pandas as pd
logger = MyLogger.getLog()

MNIST_NUM_CLASSES = 10


class UNTARGETED_IGS:

    @staticmethod
    def create_adversaries(x_1D: np.ndarray,
                           true_label: int,
                           classifier: tf.keras.Sequential,
                           alpha: float):
        gradient_1D = feature_ranker1d.compute_gradient_wrt_features(
            tf.convert_to_tensor(x_1D.reshape(-1, 784)),
            true_label,
            classifier
        )
        sign_1D = np.sign(gradient_1D)
        adv_1D = x_1D.copy()
        for iter in range(0, 20):
            adv_1D = np.clip(adv_1D + alpha * sign_1D, 0, 1)
            pred = model_object.get_model().predict(adv_1D.reshape(-1, 784))
            adv_label = np.argmax(pred)
            if true_label != adv_label:
                return adv_1D, iter
        return None, None


if __name__ == '__main__':
    # INITIALIZATION
    START_SEED = 0  # modified
    END_SEED = 10000  # modified
    name_model = "mnist_ann_keras"  # modified

    if platform.system() == 'Darwin':  # macosx
        base = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result"
    elif platform.system() == 'Linux':  # hpc
        base = f"/home/anhnd/mydeepconcolic/result"
    output_folder = f"{base}/bis/{name_model}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    summary = f"{output_folder}/summary.csv"
    logger.debug(f"summary = {summary}")

    analyzed_seed_file = f"{output_folder}/analyzed_seeds.csv"
    logger.debug(f"summary = {analyzed_seed_file}")

    analyzed_seed_indexes = []
    if os.path.exists(analyzed_seed_file):
        analyzed_seed_indexes = pd.read_csv(analyzed_seed_file, header=None)
        analyzed_seed_indexes = analyzed_seed_indexes.to_numpy()


    # MODEL
    logger.debug("initialize_dnn_model")
    model_object = initialize_dnn_model_from_name(name_model)
    classifier = model_object.get_model()

    # ATTACK
    with open(summary, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['seed', 'l0', 'l2', 'l_inf', 'minimum_change', 'true_label', 'adv_label', 'alpha',
                       'iter'])

    for seed_idx in range(START_SEED, END_SEED):
        if seed_idx in analyzed_seed_indexes:
            logger.debug(f'Visited seed {seed_idx}. Ignore!')
            continue

        # save
        with open(analyzed_seed_file, mode='a') as f:
            f.write(str(seed_idx) + ',')

        logger.debug(f"Seed = {seed_idx}")
        x_1D = model_object.get_Xtrain()[seed_idx]
        true_label = model_object.get_ytrain()[seed_idx]
        pred_label = np.argmax(model_object.get_model().predict(x_1D.reshape(-1, 784)))
        if pred_label != true_label:  # wrong predicted samples
            logger.debug(f"Ignore seed {seed_idx}")
            continue

        adv_1D, iter = UNTARGETED_IGS.create_adversaries(x_1D=x_1D,
                                                   classifier=classifier,
                                                   alpha=1/255,
                                                   true_label=true_label)
        if adv_1D is None:
            continue
        adv_1D = np.clip(adv_1D, 0, 1)
        pred = model_object.get_model().predict(adv_1D.reshape(-1, 784))

        adv_label = np.argmax(pred)
        if true_label != adv_label:
            logger.debug("Found")
            l0 = utilities.compute_l0(x_1D, adv_1D)
            l2 = utilities.compute_l2(x_1D, adv_1D)
            linf = utilities.compute_linf(x_1D, adv_1D)
            minimum_change = utilities.compute_minimum_change(x_1D, adv_1D)

            utilities.show_two_images(x_1D.reshape(28, 28),
                                      adv_1D.reshape(28, 28),
                                      left_title=f"idx = {seed_idx}: true label = {true_label}",
                                      right_title=f"pred label = {adv_label}, alpha = 1/255,\nl0 = {l0}\nl2 = {l2}",
                                      path=f"{output_folder}/{seed_idx}.png",
                                      display=False)

            with open(summary, mode='a') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(
                    [seed_idx, l0, l2, linf, minimum_change, true_label, adv_label, 1 / 255, iter])


