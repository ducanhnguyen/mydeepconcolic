from src.saved_models.mnist_ann_keras import MNIST_ANN_KERAS
import numpy as np
import matplotlib.pyplot as plt

from src.utils import utilities


def is_edge(row_idx, col_idx, x_28_28):
    if x_28_28[row_idx, col_idx] == 0:
        return False
    elif row_idx == 0 or col_idx == 0 or col_idx == 27 or row_idx == 27:
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

    idx = 36
    x_28_28 = model_object.get_Xtrain()[idx].reshape(28, 28)  # 0..1
    x_clone = x_28_28.copy()
    max = np.max(x_28_28)
    for row_idx in range(28):
        for col_idx in range(28):
            if is_edge(row_idx, col_idx, x_28_28):
                x_clone[row_idx, col_idx] = 1
            else:
                x_clone[row_idx, col_idx] = 0

    utilities.show_two_images(left_title=f"ori (idx = {idx})",
                              x_28_28_left=x_28_28.reshape(28, 28),
                              right_title="border_origin_images",
                              x_28_28_right=x_clone.reshape(28, 28),
                              display=True
                              )
