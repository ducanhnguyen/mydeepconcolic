import csv
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.decomposition import PCA
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
def create_model():
    model = Sequential()
    # model.add(Dense(1024, activation='relu', input_dim=9 * 785))
    model.add(Dense(512, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    '''
    read csv file
    '''
    df = pd.read_csv(f"/Users/ducanhnguyen/Documents/akautauto_2020/datatest/resultoftest/trainingset_0to500.csv").to_numpy()

    X_train = df[:,1:].reshape(-1, 785 * 9)
    # X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)
    print(X_train[0,:20])
    print(f"x_train shape = {X_train.shape}")

    y_train = df[:,0].reshape(-1)
    y_train = to_categorical(y_train, num_classes=2)
    print(f"y_train shape = {y_train.shape}")
    print(y_train[:10])

    '''
    train
    '''
    model = create_model()
    history = model.fit(X_train, y_train, epochs=20)

    '''
    Get testset
    '''
    df = pd.read_csv(
        f"/Users/ducanhnguyen/Documents/akautauto_2020/datatest/resultoftest/trainingset_501to1000.csv").to_numpy()

    X_test = df[:, 1:].reshape(-1, 785 * 9)
    # X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
    print(X_test[0, :20])
    print(f"X_test shape = {X_test.shape}")


    y_test = df[:, 0].reshape(-1)
    print(f"y_test shape = {y_test.shape}")
    print(y_test[:10])

    ypred = np.argmax(model.predict(X_test))
    print(f"acc = {np.sum(ypred == y_train)}/{len(y_train)}")
    print(np.where(ypred != y_train))