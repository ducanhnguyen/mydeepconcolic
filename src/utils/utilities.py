import logging
import random

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from tensorflow.python.keras.models import model_from_json, Model

from src.mnist_dataset import mnist_dataset

# matplotlib.use('TkAgg')

global logger
logger = logging.getLogger()


def compute_l2s(advs: np.ndarray,
                oris: np.ndarray,
                n_features: int):
    if not (np.min(advs) >= 0 and np.max(advs) <= 1):
        advs = advs / 255
    if not (np.min(oris) >= 0 and np.max(oris) <= 1):
        oris = oris / 255
    l2_dist = np.linalg.norm(advs.reshape(-1, n_features) - oris.reshape(-1, n_features), axis=1)
    return l2_dist


def compute_l2(adv: np.ndarray,
               ori: np.ndarray):
    if not (np.min(adv) >= 0 and np.max(adv) <= 1):
        adv = adv / 255
    if not (np.min(ori) >= 0 and np.max(ori) <= 1):
        ori = ori / 255
    return np.linalg.norm(adv.reshape(-1) - ori.reshape(-1))


def compute_l0s(advs: np.ndarray,
                oris: np.ndarray,
                n_features: int,
                normalized=False):
    if not normalized:
        advs = np.round(advs * 255)
        oris = np.round(oris * 255)
    advs = advs.reshape(-1, n_features)
    oris = oris.reshape(-1, n_features)
    l0_dist = np.sum(advs != oris, axis=1)
    return l0_dist


def compute_l0(adv: np.ndarray,
               ori: np.ndarray,
               normalized=False):  # 1d array, value in range of [0 .. 1]
    if not normalized:
        adv = np.round(adv * 255)
        ori = np.round(ori * 255)
    adv = adv.reshape(-1)
    ori = ori.reshape(-1)
    l0_dist = 0
    for idx in range(len(adv)):
        if adv[idx] != ori[idx]:
            l0_dist += 1
    return l0_dist


def compute_linf(adv: np.ndarray,
                 ori: np.ndarray):  # 1d array, value in range of [0 .. 1]
    linf_dist = 0
    adv = adv.reshape(-1)
    ori = ori.reshape(-1)
    for idx in range(len(adv)):
        if np.abs(adv[idx] - ori[idx]) > linf_dist:
            linf_dist = np.abs(adv[idx] - ori[idx])
    return linf_dist


def compute_minimum_change(adv: np.ndarray,
                           ori: np.ndarray):  # 1d array, value in range of [0 .. 1]
    minimum_change = 99999999999
    for idx in range(len(adv)):
        delta = np.abs(adv[idx] - ori[idx])
        if 0 < delta < minimum_change:
            minimum_change = delta
    return minimum_change


def show_four_images(x_28_28_first, x_28_28_second, x_28_28_third, x_28_28_fourth,
                     x_28_28_first_title="", x_28_28_second_title="", x_28_28_third_title="", x_28_28_fourth_title="",
                     path=None, display=False):
    matplotlib.rcParams.update({'font.size': 8})
    if path is None and not display:
        return
    fig = plt.figure()
    fig1 = fig.add_subplot(2, 2, 1)
    fig1.title.set_text(x_28_28_first_title)
    plt.imshow(x_28_28_first, cmap="gray")

    fig2 = fig.add_subplot(2, 2, 2)
    fig2.title.set_text(x_28_28_second_title)
    plt.imshow(x_28_28_second, cmap='gray')

    fig3 = fig.add_subplot(2, 2, 3)
    fig3.title.set_text(x_28_28_third_title)
    plt.imshow(x_28_28_third, cmap='gray')

    fig4 = fig.add_subplot(2, 2, 4)
    fig4.title.set_text(x_28_28_fourth_title)
    plt.imshow(x_28_28_fourth, cmap='gray')

    plt.tight_layout(h_pad=0.5, w_pad=0.2)
    if path is not None:
        # plt.savefig(path, pad_inches=0.3, bbox_inches='tight')
        # plt.savefig(path, pad_inches=0, bbox_inches='tight', format='eps')
        plt.savefig(path, pad_inches=0, bbox_inches='tight', format='png')

    if display:
        plt.show()


def show_ori_adv_optmizedadv(ori_28_28: np.ndarray, adv_28_28: np.ndarray, optimizedadv_28_28: np.ndarray, highlight: np.ndarray,
                             path=None, display=False):
    matplotlib.rcParams.update({'font.size': 10})
    if path is None and not display:
        return
    fig = plt.figure()

    n_col = len(ori_28_28)
    n_row = 4

    for idx in range(n_col):
        # ori
        ax = fig.add_subplot(n_row, n_col, idx + 1)
        if idx == 0:
            plt.text(-0.3, 0.5, 'Origin',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes,
                     rotation=90)
        plt.imshow(ori_28_28[idx], cmap="gray")
        plt.gca().axes.get_xaxis().set_visible(False)  # remove axis label
        plt.gca().axes.get_yaxis().set_visible(False)

        # adv
        ax = fig.add_subplot(n_row, n_col, n_col + idx + 1)
        # L0_after = int(np.round(compute_l0(oris[idx], advs[idx], normalized=True)))
        # L2_after = np.round(compute_l2(oris[idx], advs[idx]), 1)
        # ax.title.set_text(f'L0: {L0_after}\nL2: {L2_after}')
        if idx == 0:
            plt.text(-0.3, 0.5, 'Adversary',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes,
                     rotation=90)
        plt.imshow(adv_28_28[idx], cmap="gray")
        plt.gca().axes.get_xaxis().set_visible(False)  # remove axis label
        plt.gca().axes.get_yaxis().set_visible(False)

        # optimized adv
        ax = fig.add_subplot(n_row, n_col, 2 * n_col + idx + 1)
        # L0_after = int(np.round(compute_l0(oris[idx], optimizedadv_28_28[idx], normalized=True)))
        # L2_after = np.round(compute_l2(oris[idx], optimizedadv_28_28[idx]), 1)
        # ax.title.set_text(f'L0: {L0_after}\nL2: {L2_after}')
        if idx == 0:
            plt.text(-0.3, 0.5, 'Optimized\nAdversary',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes,
                     rotation=90)
        plt.imshow(optimizedadv_28_28[idx], cmap="gray")
        plt.gca().axes.get_xaxis().set_visible(False)  # remove axis label
        plt.gca().axes.get_yaxis().set_visible(False)

        # highlight
        ax = fig.add_subplot(n_row, n_col, 3 * n_col + idx + 1)
        if idx == 0:
            plt.text(-0.3, 0.5, 'Difference',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes,
                     rotation=90)
        plt.imshow(highlight[idx], cmap="gray")
        plt.gca().axes.get_xaxis().set_visible(False)  # remove axis label
        plt.gca().axes.get_yaxis().set_visible(False)


    plt.tight_layout(h_pad=0.5, w_pad=0.2)

    if path is not None:
        # plt.savefig(path, pad_inches=0.3, bbox_inches='tight')
        # plt.savefig(path, pad_inches=0, bbox_inches='tight', format='eps')
        plt.savefig(path, pad_inches=0, bbox_inches='tight', format='png')

    if display:
        plt.show()


def show_two_images(x_28_28_left, x_28_28_right, left_title="", right_title="", path=None, display=False):
    fig = plt.figure()
    fig1 = fig.add_subplot(1, 2, 1)
    fig1.title.set_text(left_title)
    plt.imshow(x_28_28_left, cmap="gray")

    fig2 = fig.add_subplot(1, 2, 2)
    fig2.title.set_text(right_title)
    plt.imshow(x_28_28_right, cmap='gray')

    if path is not None:
        plt.savefig(path, pad_inches=0, bbox_inches='tight')

    if display:
        plt.show()


def plot_line_chart(x, y, x_title=None, y_title=None, title=None):
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    if x_title is not None:
        plt.xlabel(x_title)
    if y_title is not None:
        plt.ylabel(y_title)
    if title is not None:
        plt.title(title)
    plt.show()


def load_model(weight_path: str,
               structure_path: str,
               trainset_path: str):
    # load structure of model
    json_file = open(structure_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load weight from file
    model.load_weights(weight_path)
    return model


def visualize_cnn(x_image_4D: np.ndarray,
                  model: keras.engine.sequential.Sequential,
                  specified_layer: str):
    for layer in model.layers:
        # logger.debug(f"Layer {layer.name}")
        if specified_layer is None or \
                (specified_layer is not None and layer.name == specified_layer):

            # redefine model to output right after the first hidden layer
            model = Model(inputs=model.inputs, outputs=layer.output)

            logger.debug("Partial model")
            model.summary()

            # get feature map for first hidden layer
            feature_maps = model.predict(x_image_4D)

            logger.debug(f"Input shape = {model.input.shape}")  # MNIST: (None, 28, 28, 1)
            logger.debug(f"Output shape = {model.output.shape}")  # MNIST: (None, 24, 24, 32)
            logger.debug(f"Feature maps shape = {feature_maps.shape}")  # MNIST: (1, 24, 24, 32)

            if len(feature_maps.shape) < 4:  # feature map must be 4d
                continue
            # plot all 64 maps in an 8x8 squares
            n_feature_maps = feature_maps.shape[3]
            n_col = 8
            n_row = np.floor(n_feature_maps / n_col).astype(np.int)
            ix = 1

            pyplot.figure().suptitle(f"name layer = {layer.name}, shape = {layer.output.shape}")
            for _ in range(n_row):
                for _ in range(n_col):
                    if ix > n_feature_maps:
                        break

                    # specify subplot and turn of axis
                    ax = pyplot.subplot(n_row, n_col, ix)  # index starts with 1
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot filter channel in grayscale
                    feature_map = feature_maps[0, :, :, ix - 1]
                    pyplot.imshow(feature_map, cmap='gray')
                    ix += 1

            # show the figure
            pyplot.show()

            if specified_layer is not None:
                break


def category2indicator(y, nclass=10):
    Y = np.zeros(shape=(y.shape[0], nclass))

    for idx, item in enumerate(y):
        Y[idx][item] = 1

    return Y


def highlight_diff(img1_0_255, img2_0_255):
    shape = img2_0_255.shape
    img1_0_255 = img1_0_255.reshape(-1)
    img2_0_255 = img2_0_255.reshape(-1)
    highlight = []
    for idx in range(len(img1_0_255)):
        if img1_0_255[idx] != img2_0_255[idx]:
            highlight.append(1)
        else:
            highlight.append(0)
    highlight = np.asarray(highlight)
    return highlight.reshape(shape)


if __name__ == '__main__':
    # logger.debug("initialize_dnn_model")
    # model = keras.models.load_model(
    #     filepath="/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/rivf/autoencoder_mnist.h5",
    #     compile=False)
    # model.summary()
    #
    # mnist_loader = mnist_dataset()
    # Xtrain, ytrain, Xtest, ytest = mnist_loader.read_data(
    #     trainset_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/train.csv',
    #     testset_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/test.csv')
    # x_image_4D = Xtrain[1].reshape(-1, 28, 28, 1)
    #
    # visualize_cnn(x_image_4D=x_image_4D, model=model, specified_layer=None)

    advs = np.load(
        "/Users/ducanhnguyen/Documents/mydeepconcolic/optimization_batch3/Alexnet/trash_ae_border_Alexnet_9to7_rankingCOI_step6/adv.npy")
    advs = advs.reshape(-1, 28, 28)

    oris = np.load(
        "/Users/ducanhnguyen/Documents/mydeepconcolic/optimization_batch3/Alexnet/trash_ae_border_Alexnet_9to7_rankingCOI_step6/origin.npy")
    oris = oris.reshape(-1, 28, 28)

    optimized_advs = np.load(
        "/Users/ducanhnguyen/Documents/mydeepconcolic/optimization_batch3/Alexnet/trash_ae_border_Alexnet_9to7_rankingCOI_step6/optimized_adv.npy")
    optimized_advs = optimized_advs.reshape(-1, 28, 28)

    selected = []
    l0s = compute_l0s(optimized_advs.reshape(-1, 784), oris.reshape(-1, 784), n_features=784,  normalized=True)
    for idx in range(len(l0s)):
        if l0s[idx] <= 10:
            selected.append(idx)
    selected = selected[:7]
    highlight = optimized_advs != advs
    show_ori_adv_optmizedadv(oris[selected], advs[selected], optimized_advs[selected], highlight[selected], display=True, path=None)
