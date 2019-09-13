from keras.layers import Dense, Activation
from keras.models import Sequential

from src.abstract_dataset import *
import pandas as pd

'''
Data set: https://www.kaggle.com/zalando-research/fashionmnist
Train on 60000 samples.

model 1:
    Accuracy on test set = 0.8776
    Accuracy on train set = 0.9316
    
model 2:
    Accuracy on test set = 0.8823
    Accuracy on train set = 0.9683
    
'''
class FASHION_MNIST(abstract_dataset):

    def __init__(self):
        super(FASHION_MNIST, self).__init__()

    def read_data(self, trainset_path, testset_path):
        X = pd.read_csv(trainset_path).to_numpy()
        self.set_Xtrain(X[:, 1:]/255)
        self.set_ytrain(X[:, 0].reshape(-1))

        # we do not know the label of samples in test set
        # to compute the accuracy of the model, we will upload the prediction on kaggle
        X = pd.read_csv(testset_path).to_numpy()
        self.set_Xtest(X[:, 1:]/255)
        self.set_ytest(X[:, 0].reshape(-1))

        return self.get_Xtrain(), self.get_ytrain(), self.get_Xtest(), self.get_ytest()

    def create_model(self, input_shape):
        model = Sequential()

        model.add(Dense(64, name='dense_0', input_dim=input_shape))
        model.add(Activation('relu', name='relu_0'))

        model.add(Dense(32, name='dense_1', input_dim=input_shape))
        model.add(Activation('relu', name='relu_1'))

        model.add(Dense(16, name='dense_2'))
        model.add(Activation('relu', name='relu_2'))

        model.add(Dense(self.get_num_classes(), name='dense_n'))
        model.add(Activation('softmax', name='softmax'))

        self.set_model(model)
        return model

if __name__ == '__main__':
    # train model
    mnist = FASHION_MNIST()
    mnist.set_num_classes(10)

    '''
    mnist.train_model(train=True, kernel_path='../saved_models/fashion_mnist_ann_keras.h5',  model_path='../saved_models/fashion_mnist_ann_keras.json',
                      training_path='../../fashion_mnist/train.csv', testing_path='../../fashion_mnist/test.csv')
    '''
    # load model
    model = mnist.load_model(weight_path='../saved_models/fashion_mnist_ann_keras.h5', structure_path='../saved_models/fashion_mnist_ann_keras.json',
                             trainset_path='../../fashion_mnist/train.csv')
    print(model.summary())
    mnist.read_data(trainset_path='../../fashion_mnist/train.csv', testset_path='../../fashion_mnist/test.csv')
    print(f'model.layers = {model.layers}')

    # plot an observation
    import matplotlib.pyplot as plt
    x_train, y_train = mnist.get_an_observation(index=516)
    img = x_train.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f'A sample')
    plt.show()


