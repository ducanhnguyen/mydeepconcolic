import tensorflow as tf
import numpy as np
"""
AlexNet
    acc train set = 0.9903166666666666
    acc test set = 0.9875
    
LeNet_v2
    acc train set =  0.9986166666666667
    acc test set =  0.9882
    wrongseeds_LeNet_v2 = [80, 494, 635, 902, 2426, 2427, 2488, 5298, 6315, 6848, 6885, 8200]

LeNet
    acc train set = 0.9960833333333333
    acc test set = 0.9851
    wrongseeds_LeNet = [80, 132, 494, 500, 788, 1201, 1244, 1604, 2426, 2554, 2622, 2676, 3210, 4014, 4164, 4402, 4438, 4476, 4715, 5065, 5174, 5482, 5723, 5798, 5821, 5855, 6197, 6202, 6246, 6315, 6818, 6885, 7006, 7080, 7264, 7606, 7972, 7994, 8200, 8202, 8270, 8480, 8729, 8772, 8849, 9256, 9266]
vgg13
    acc train set = 0.99795
    acc test set = 0.9918
vgg16
    acc train set = 0.9812833333333333
    acc test set = 0.981
"""
import keras

if __name__ == '__main__':
    ATTACKED_MODEL_H5 = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/model/Lenet_v2.h5"
    print(ATTACKED_MODEL_H5)
    dnn = keras.models.load_model(filepath=ATTACKED_MODEL_H5, compile=False)
    dnn.summary()

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    X_train = X_train / 255
    X_test = X_test / 255

    Y_pred = dnn.predict(X_train.reshape(-1, 28, 28, 1))
    y_pred = np.argmax(Y_pred, axis=1)
    print(f"acc train set = {(len(y_train) - np.sum(y_pred != y_train)) / len(y_train)}")

    wrong_seeds = []
    for idx in range(len(y_test)):
        if y_pred[idx] != y_train[idx]:
            wrong_seeds.append(idx)
    print(wrong_seeds)

    Y_pred = dnn.predict(X_test.reshape(-1, 28, 28, 1))
    y_pred = np.argmax(Y_pred, axis=1)
    print(f"acc test set = {(len(y_test) - np.sum(y_pred != y_test)) / len(y_test)}")