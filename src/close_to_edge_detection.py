from src.saved_models.mnist_ann_keras import MNIST_ANN_KERAS
import numpy as np
import matplotlib.pyplot as plt


def close_to_edge(row_idx, col_idx, x_28_28):
    isEdge = False
    avg = 0
    min = 0
    if x_28_28[row_idx, col_idx] > 0:
        isEdge = False
    elif col_idx - 1 >= 0 and \
            row_idx + 1 <= 27 and \
            x_28_28[row_idx, col_idx - 1] > 0 and x_28_28[row_idx + 1, col_idx - 1] > 0 \
            and x_28_28[row_idx + 1, col_idx] > 0:
        avg = (x_28_28[row_idx, col_idx - 1] + x_28_28[row_idx + 1, col_idx - 1] + x_28_28[
            row_idx + 1, col_idx]) / 3
        min = np.min([x_28_28[row_idx, col_idx - 1], x_28_28[row_idx + 1, col_idx - 1], x_28_28[
            row_idx + 1, col_idx]])
        isEdge = True
    elif col_idx - 1 >= 0 and \
            row_idx - 1 >= 0 and \
            x_28_28[row_idx, col_idx - 1] > 0 and x_28_28[row_idx - 1, col_idx - 1] > 0 \
            and x_28_28[row_idx - 1, col_idx] > 0:
        avg = (x_28_28[row_idx, col_idx - 1] + x_28_28[row_idx - 1, col_idx - 1] + x_28_28[
            row_idx - 1, col_idx]) / 3
        min = np.min([x_28_28[row_idx, col_idx - 1], x_28_28[row_idx - 1, col_idx - 1], x_28_28[
            row_idx - 1, col_idx]])
        isEdge = True
    elif col_idx + 1 <= 27 and \
            row_idx - 1 >= 0 and \
            x_28_28[row_idx, col_idx + 1] > 0 and x_28_28[row_idx - 1, col_idx] > 0 \
            and x_28_28[row_idx - 1, col_idx + 1] > 0:
        avg = (x_28_28[row_idx, col_idx + 1] + x_28_28[row_idx - 1, col_idx] + x_28_28[
            row_idx - 1, col_idx + 1]) / 3
        min = np.min([x_28_28[row_idx, col_idx + 1], x_28_28[row_idx - 1, col_idx], x_28_28[
            row_idx - 1, col_idx + 1]])
        isEdge = True
    elif col_idx + 1 <= 27 and \
            row_idx + 1 <= 27 and \
            x_28_28[row_idx, col_idx + 1] > 0 and x_28_28[row_idx + 1, col_idx + 1] > 0 \
            and x_28_28[row_idx + 1, col_idx] > 0:
        avg = (x_28_28[row_idx, col_idx + 1] + x_28_28[row_idx + 1, col_idx + 1] + x_28_28[row_idx + 1, col_idx]) / 3
        min = np.min([x_28_28[row_idx, col_idx + 1], x_28_28[row_idx + 1, col_idx + 1], x_28_28[row_idx + 1, col_idx]])
        isEdge = True
    return isEdge, avg, min


if __name__ == '__main__':
    model_object = MNIST_ANN_KERAS()
    model_object.set_num_classes(10)
    model_object.read_data(
        trainset_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/train.csv',
        testset_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/test.csv')

    base_path = "/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_ann_keras_1_pixel_attack/"

    x_28_28 = model_object.get_Xtrain()[5638].reshape(28, 28)  # 0..1
    x_clone = x_28_28.copy()
    max = np.max(x_28_28)

    for feature_idx in range(784):
        row_idx = int(np.floor(feature_idx / 28))
        col_idx = feature_idx - row_idx * 28
        close, avg, min = close_to_edge(row_idx, col_idx, x_28_28)
        if close:
            x_clone[row_idx, col_idx] = 1
        elif x_28_28[row_idx, col_idx] > 0.9:
            x_clone[row_idx, col_idx] = 0.9
        else:
            x_clone[row_idx, col_idx] = x_28_28[row_idx, col_idx]

    # for row_idx in range(28):
    #     for col_idx in range(28):
    #         if close_to_edge(row_idx, col_idx, x_28_28):
    #             x_clone[row_idx, col_idx] = 1
    #         elif x_28_28[row_idx, col_idx] > 0:
    #             x_clone[row_idx, col_idx] = 0.5
    #         else:
    #             x_clone[row_idx, col_idx] = 0
    #
    fig = plt.figure()
    fig1 = fig.add_subplot(1, 2, 1)
    fig1.title.set_text(
        f'origin')
    plt.imshow(x_28_28, cmap="gray")

    fig2 = fig.add_subplot(1, 2, 2)
    fig2.title.set_text("Most close-edge features are highlighted")
    plt.imshow(x_clone, cmap='gray')

    # plt.title()
    plt.show()
