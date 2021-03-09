import csv
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def analyze_smt_output(solution_path):
    line = open(solution_path, "r").readline()
    if 'error' in line or 'unknown' in line or 'unsat' in line:
        return []
    else:
        pairs = line.split(';')
        indexes = []
        values = []
        for pair in pairs:
            key = pair.split("=")[0]
            if key.startswith('feature_'):
                index = key.split("_")[1]
                indexes.append(index)

                value = pair.split("=")[1]
                values.append(value)

            else:
                continue

        img = np.zeros(shape=(len(indexes)))
        for index, value in zip(indexes, values):
            img[int(index)] = value

        logger.debug(img.shape)
        assert (len(img.shape) == 1)
        if len(img) == 0:
            img = None
        return img


def is_valid_adv(model_object, config, csv_new_image_path: str):
    with open(config.true_label_seed_file, 'r') as f:
        true_label = int(f.read())
    logger.debug(f'{config.thread_name}: True label = {true_label}')

    # Compare the prediction
    seed = pd.read_csv(config.seed_file, header=None).to_numpy().reshape(1, -1)

    # with config.graph.as_default():
    original_prediction = np.argmax(model_object.get_model().predict(seed))
    logger.debug(f'{config.thread_name}: The prediction of the original seed = {original_prediction}')

    if true_label != original_prediction:
        logger.debug("Original prediction != true label")

    new_image = pd.read_csv(csv_new_image_path, header=None)  # [0..255]
    new_image = new_image / 255
    new_image = new_image.to_numpy()
    new_image = new_image.reshape(1, -1)
    modified_prediction = np.argmax(model_object.get_model().predict(new_image))
    logger.debug(f'{config.thread_name}: The prediction of the modified seed = {modified_prediction}')

    if modified_prediction != original_prediction and original_prediction == true_label:
        success = True
    else:
        success = False

    return success


def draw_figure(model_object, seed, original_prediction, modified_prediction, png_comparison_image_path,
                png_new_image_path, new_image_path,
                l2_dist: float,
                l0_dist: int):
    new_image = pd.read_csv(new_image_path, header=None).to_numpy().reshape(model_object.get_image_shape())

    # if the input model is in range of [0..1] and the value of pixel in image is in [0..255], we need to scale the image
    new_image /= 255

    matplotlib.image.imsave(png_new_image_path, new_image)
    '''
    if 'ubuntu' in platform.platform().lower():
        # scipy.misc.imsave works on macosx but does not work on ubuntu 17 + python 3.6
        # solution: on ubuntu 17, use matplotlib.image.imsave instead
        matplotlib.image.imsave(png_new_image_path, new_image)
    else:
        scipy.misc.imsave(new_image, cmin=0.0, cmax=1).save(png_new_image_path)
    '''

    fig = plt.figure()
    nrow = 1
    ncol = 2

    if len(model_object.get_image_shape()) == 2:
        # the input is black-white image
        # logger.debug('image_shape = 2')
        seed = seed.reshape(model_object.get_image_shape())

        fig1 = fig.add_subplot(nrow, ncol, 1)
        fig1.title.set_text(f'original image\n(prediction = {original_prediction})')
        plt.imshow(seed, cmap="gray")

        new_image = new_image.reshape(model_object.get_image_shape())
        fig2 = fig.add_subplot(nrow, ncol, 2)
        fig2.title.set_text(
            f'modified image\n(prediction = {modified_prediction})\n l0 = ' + str(l0_dist) + ', l2 = ' + str(l2_dist))
        plt.imshow(new_image, cmap="gray")

    elif len(model_object.get_image_shape()) == 3:
        # the input is rgb image
        # logger.debug('image_shape = 3')

        seed = seed.reshape(model_object.get_image_shape())
        fig1 = fig.add_subplot(nrow, ncol, 1)
        fig1.title.set_text(f'The original image\n(prediction = {original_prediction})')
        plt.imshow(seed)

        new_image = new_image.reshape(model_object.get_image_shape())
        fig2 = fig.add_subplot(nrow, ncol, 2)
        fig2.title.set_text(f'The modified image\n(prediction = {modified_prediction})')
        plt.imshow(new_image)

    l1_distance = compute_L1_distance(seed, new_image)
    logger.debug(f'l1_distance between two image= {l1_distance}')
    diff_pixels = compute_the_different_pixels(seed, new_image)

    plt.savefig(png_comparison_image_path, pad_inches=0, bbox_inches='tight')
    logger.debug('Saved image')
    # plt.show()

    return diff_pixels

def compute_the_different_pixels(img1, img2):
    diff = img1 - img2
    diff = diff.reshape(-1)
    diff_pixels = 0
    for item in diff:
        if item != 0:
            diff_pixels = diff_pixels + 1

    logger.debug(f'The different points in two image = {diff_pixels}')
    return diff_pixels


def compute_L1_distance(img1, img2):
    distance = np.sum(np.abs(img1 - img2))
    return distance


if __name__ == '__main__':
    seed_file = f'../data/seed.csv'
    true_label_seed_file = f'../data/true_label.txt'
    constraint_file = f'../data/constraint.txt'

    img = analyze_smt_output(solution_path="../data/norm_solution.txt")

    # export new image to file for later usage
    new_image_path = f'../data/new_image.csv'
    with open(new_image_path, mode='w') as f:
        seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        seed.writerow(img)

    # plot the seed and the new image
    seed, new_image, similar = is_valid_adv(seed_path=seed_file,
                                            csv_new_image_path=new_image_path)
