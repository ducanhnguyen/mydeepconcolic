import csv
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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

        # img = img.reshape(image_shape)
        return img


def plot_seed_and_new_image(model, seed_path, new_image_path, png_comparison_image_path, image_shape, png_new_image_path):
    with open('../data/true_label.txt', 'r') as f:
        true_label = int(f.read())
    logger.debug(f'True label = {true_label}')

    # Compare the prediction
    seed = pd.read_csv(seed_path, header=None)
    seed = seed.to_numpy()
    seed = seed.reshape(1, -1)
    original_prediction = np.argmax(model.predict(seed))
    logger.debug(f'The prediction of the original seed = {original_prediction}')

    new_image = pd.read_csv(new_image_path, header=None)
    new_image = new_image.to_numpy()
    new_image = new_image.reshape(1, -1)
    modified_prediction = np.argmax(model.predict(new_image))
    logger.debug(f'The prediction of the modified seed = {modified_prediction}')

    if modified_prediction != original_prediction and original_prediction == true_label:
        keep = True
    else:
        keep = False

    # Export new image to file
    new_image = pd.read_csv(new_image_path, header=None)
    new_image = new_image.to_numpy()
    new_image = new_image.reshape(image_shape)
    new_image /= 255
    import scipy.misc
    scipy.misc.toimage(new_image, cmin=0.0, cmax=1).save(png_new_image_path)

    # draw figure
    fig = plt.figure()
    nrow = 1
    ncol = 2

    if len(image_shape) == 2:
        logger.debug('image_shape = 2')
        seed = seed.reshape(image_shape)
        #seed /= 255 # for shvn

        fig1 = fig.add_subplot(nrow, ncol, 1)
        fig1.title.set_text(f'The original image\n(prediction = {original_prediction})')
        plt.imshow(seed, cmap="gray")

        new_image = new_image.reshape(image_shape)
        #new_image /= 255  # for shvn
        fig2 = fig.add_subplot(nrow, ncol, 2)
        fig2.title.set_text(f'The modified image\n(prediction = {modified_prediction})')
        plt.imshow(new_image, cmap="gray")

    elif len(image_shape) == 3:
        logger.debug('image_shape = 3')

        seed = seed.reshape(image_shape)
        #seed /= 255  # for shvn
        fig1 = fig.add_subplot(nrow, ncol, 1)
        fig1.title.set_text(f'The original image\n(prediction = {original_prediction})')
        plt.imshow(seed)

        new_image = new_image.reshape(image_shape)
        new_image /= 255  # for shvn
        fig2 = fig.add_subplot(nrow, ncol, 2)
        fig2.title.set_text(f'The modified image\n(prediction = {modified_prediction})')
        plt.imshow(new_image)

    l1_distance = compute_L1_distance(seed, new_image)
    logger.debug(f'l1_distance between two image= {l1_distance}')
    compute_the_different_pixels(seed, new_image)

    plt.savefig(png_comparison_image_path)

    #plt.show()

    return keep


def compute_the_different_pixels(img1, img2):
    diff = img1 - img2
    diff = diff.reshape(-1)
    count = 0
    for item in diff:
        if item != 0:
            count = count + 1

    logger.debug(f'The different points in two image = {count}')
    return count


def compute_L1_distance(img1, img2):
    distance = np.sum(np.abs(img1 - img2))
    return distance


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
    seed, new_image, similar = plot_seed_and_new_image(seed_path=seed_file,
                                                       new_image_path=new_image_path)
