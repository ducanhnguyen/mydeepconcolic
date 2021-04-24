import keras
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from tensorflow.python.keras.models import model_from_json, Model

from src.deepconcolic import logger
from src.mnist_dataset import mnist_dataset


def compute_l2(adv: np.ndarray,
               ori: np.ndarray):  # 1d array, value in range of [0 .. 1]
    return np.linalg.norm(adv - ori)


def compute_l0(adv: np.ndarray,
               ori: np.ndarray,
               normalized = False):  # 1d array, value in range of [0 .. 1]
    if not normalized:
        adv = np.round(adv*255)
        ori = np.round(ori*255)

    l0_dist = 0
    for idx in range(len(adv)):
        if adv[idx] != ori[idx]:
            l0_dist += 1
    return l0_dist


def compute_linf(adv: np.ndarray,
                 ori: np.ndarray):  # 1d array, value in range of [0 .. 1]
    linf_dist = 0
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


def category2indicator(y, nclass = 10):
    Y = np.zeros(shape=(y.shape[0], nclass))

    for idx, item in enumerate(y):
        Y[idx][item] = 1

    return Y

if __name__ == '__main__':
    logger.debug("initialize_dnn_model")
    model = keras.models.load_model(
        filepath="/Users/ducanhnguyen/Documents/mydeepconcolic/src/saved_models/rivf/pretrained_mnist_cnn1.h5")
    model.summary()

    mnist_loader = mnist_dataset()
    Xtrain, ytrain, Xtest, ytest = mnist_loader.read_data(
        trainset_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/train.csv',
        testset_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/test.csv')
    x_image_4D = Xtrain[1].reshape(-1, 28, 28, 1)

    visualize_cnn(x_image_4D=x_image_4D, model=model, specified_layer=None)
