from src.saved_models.mnist_ann_keras import MNIST_ANN_KERAS
import numpy as np
import matplotlib.pyplot as plt


def is_edge(row_idx, col_idx, x_28_28):
    if x_28_28[row_idx, col_idx] == 0:
        return False
    elif row_idx == 0 or col_idx == 0 or col_idx == 28 or row_idx == 28:
        return True
    elif x_28_28[row_idx - 1, col_idx] > 0 \
            and x_28_28[row_idx + 1, col_idx] > 0 \
            and x_28_28[row_idx, col_idx - 1] > 0 \
            and x_28_28[row_idx, col_idx + 1] > 0 \
            and x_28_28[row_idx, col_idx] > 0:
        return False
    else:
        return True


if __name__ == '__main__':
    model_object = MNIST_ANN_KERAS()
    model_object.set_num_classes(10)
    model_object.read_data(
        trainset_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/train.csv',
        testset_path='/Users/ducanhnguyen/Documents/mydeepconcolic/dataset/digit-recognizer/test.csv')

    base_path = "/Users/ducanhnguyen/Documents/mydeepconcolic/result/mnist_ann_keras_1_pixel_attack/"

    x_28_28 = model_object.get_Xtrain()[1].reshape(28, 28)  # 0..1
    x_clone = x_28_28.copy()
    max = np.max(x_28_28)
    for row_idx in range(28):
        for col_idx in range(28):
            if is_edge(row_idx, col_idx, x_28_28):
                x_clone[row_idx, col_idx] = max + 1
            else:
                x_clone[row_idx, col_idx] = x_28_28[row_idx, col_idx]

    plt.imshow(x_clone, cmap='gray')
    plt.title("Most edge features are highlighted")
    plt.show()
