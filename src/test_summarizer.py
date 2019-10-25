import csv
import logging
import os
import threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as LA

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

lock = threading.Lock()

def get_new_image(solution_path):
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

        # if the input model is in range of [0..1] and the value of pixel in image is in [0..255], we need to scale the image
        img /= 255
        assert (len(img.shape) == 1)
        return img


def is_valid_modified_image(model_object, threadconfig, csv_original_image_path, csv_new_image_path):
    assert (os.path.exists(csv_new_image_path))
    assert (os.path.exists(csv_original_image_path))
    assert (model_object != None and threadconfig != None and threadconfig.graph != None)
    assert (os.path.exists(threadconfig.true_label_seed_file))

    # get the true label
    with open(threadconfig.true_label_seed_file, 'r') as f:
        true_label = int(f.read())
    logger.debug(f'{threadconfig.thread_name}: True label = {true_label}')

    # Make prediction on the original sample

    original_image = pd.read_csv(csv_original_image_path, header=None).to_numpy().reshape(1, -1)
    lock.acquire()
    with threadconfig.graph.as_default():
        original_prediction = np.argmax(model_object.get_model().predict(original_image))
        logger.debug(f'{threadconfig.thread_name}: The prediction of the original seed = {original_prediction}')
    lock.release()

    # Make prediction on the modified sample
    modified_image = pd.read_csv(csv_new_image_path, header=None).to_numpy().reshape(1, -1)

    lock.acquire()
    with threadconfig.graph.as_default():
        modified_prediction = np.argmax(model_object.get_model().predict(modified_image))
        logger.debug(f'{threadconfig.thread_name}: The prediction of the modified seed = {modified_prediction}')
    lock.release()

    assert (len(original_image) == len(modified_image))

    # compare
    if modified_prediction != original_prediction and original_prediction == true_label:
        is_valid = True
    else:
        is_valid = False

    return is_valid, original_prediction, modified_prediction


def create_figure_comparison(model_object, original_image, modified_image, original_prediction, modified_prediction, png_comparison_image_path):
    assert (len(original_image.shape) == 1 and len(modified_image.shape) == 1 and len(original_image) == len(modified_image))
    assert (original_prediction >= 0 and original_prediction >= 0)
    assert (png_comparison_image_path != None)
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

    # add the original image to the plot
    original_image = original_image.reshape(model_object.get_image_shape())
    fig1 = fig.add_subplot(nrow, ncol, 1)
    fig1.title.set_text(f'The original image\n(prediction = {original_prediction})')
    plt.imshow(original_image, cmap="gray")

    # add the modified image to the plot
    modified_image = modified_image.reshape(model_object.get_image_shape())
    fig2 = fig.add_subplot(nrow, ncol, 2)
    fig2.title.set_text(f'The modified image\n(prediction = {modified_prediction})')
    plt.imshow(modified_image, cmap="gray")

    # save to disk
    plt.savefig(png_comparison_image_path, pad_inches=0, bbox_inches='tight')
    logger.debug('Saved image')
    #plt.show()

    return png_comparison_image_path

def compute_the_different_pixels(img1, img2):
    diff = img1 - img2
    diff = diff.reshape(-1)
    diff_pixels = 0
    for item in diff:
        if item != 0:
            diff_pixels = diff_pixels + 1

    logger.debug(f'The different points in two image = {diff_pixels}')
    return diff_pixels


def compute_L0_distance(img1, img2):
    # Frobenius norm
    return LA.norm(img1 - img2, 0)

def compute_L1_distance(img1, img2):
    # Frobenius norm
    return LA.norm(img1 - img2, 1)

def compute_L2_distance(img1, img2):
    # Frobenius norm
    return LA.norm(img1 - img2, 2)

def compute_inf_distance(img1, img2):
    return LA.norm(img1 - img2, np.inf)

if __name__ == '__main__':
    seed_file = f'../data/seed.csv'
    true_label_seed_file = f'../data/true_label.txt'
    constraint_file = f'../data/constraint.txt'

    img = get_new_image(solution_path="../data/norm_solution.txt")

    # export new image to file for later usage
    new_image_path = f'../data/new_image.csv'
    with open(new_image_path, mode='w') as f:
        seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        seed.writerow(img)

    # plot the seed and the new image
    seed, new_image, similar = is_valid_modified_image(seed_path=seed_file,
                                                       csv_new_image_path=new_image_path)
