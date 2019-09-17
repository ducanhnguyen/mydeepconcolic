from keras.layers import Dense, Activation
from keras.models import Sequential

from src.abstract_dataset import *
import pandas as pd

'''
Data set: https://www.kaggle.com/c/digit-recognizer
Train on 42000 samples.

Test the model on kaggle:
(1) 784 units (input layer), 32 units, 16 units, 10 units 
    accuracy on original test set = 0.95957 -> 0.95957 (original test set + adversarial samples)
'''
class MNIST(abstract_dataset):

    def __init__(self):
        super(MNIST, self).__init__()

    def read_data(self, trainset_path, testset_path, divided_by_255 = True):
        X = pd.read_csv(trainset_path).to_numpy()
        if divided_by_255:
            self.set_Xtrain(X[:, 1:] / 255)
        else:
            self.set_Xtrain(X[:, 1:])
        self.set_ytrain(X[:, 0].reshape(-1).astype(int))

        # we do not know the label of samples in test set
        # to compute the accuracy of the model, we will upload the prediction on kaggle
        X = pd.read_csv(testset_path).to_numpy()
        if divided_by_255:
            self.set_Xtest(X / 255)
        else:
            self.set_Xtest(X)
        return self.get_Xtrain(), self.get_ytrain(), self.get_Xtest(), self.get_ytest()

    def create_model(self, input_shape):
        model = Sequential()

        model.add(Dense(32, name='dense_1', input_dim=input_shape))
        model.add(Activation('relu', name='relu_1'))

        model.add(Dense(16, name='dense_2'))
        model.add(Activation('relu', name='relu_2'))

        model.add(Dense(self.get_num_classes(), name='dense_n'))
        model.add(Activation('softmax', name='softmax'))

        self.set_model(model)
        return model

    def predict_on_testset_kaggle(self, path):
        model = self.get_model()
        Ytest = np.argmax(model.predict(self.get_Xtest()), axis=1)

        import csv
        with open(path, mode='w') as f:
            seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            seed.writerow(['ImageId', 'Label'])
            for idx, item in enumerate(Ytest):
                seed.writerow([idx + 1, item]) # index starts from 1
        print(Ytest)

if __name__ == '__main__':
    # train model
    mnist = MNIST()
    mnist.set_num_classes(10)

    '''
    mnist.train_model(train=True, kernel_path='../saved_models/mnist_ann_keras_expansion.h5',  model_path='../saved_models/mnist_ann_keras_expansion.json',
                      training_path='/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/digit-recognizer/train_expansion.csv',
                      testing_path='/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/digit-recognizer/test.csv')
    '''
    # load model
    model = mnist.load_model(weight_path='../saved_models/mnist_ann_keras_expansion.h5', structure_path='../saved_models/mnist_ann_keras_expansion.json',
                             trainset_path='/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/digit-recognizer/train_expansion.csv')
    mnist.read_data(trainset_path='/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/digit-recognizer/train_expansion.csv',
                    testset_path='/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/digit-recognizer/test.csv')
    print(f'model.layers = {model.layers}')
    mnist.predict_on_testset_kaggle(path = '/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/src/saved_models/mnist_test_result.csv')

    # plot an observation
    import matplotlib.pyplot as plt
    x_train, y_train = mnist.get_an_observation(index=516)
    img = x_train.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f'A sample')
    plt.show()

