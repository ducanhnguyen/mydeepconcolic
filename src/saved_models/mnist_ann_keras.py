from keras.layers import Dense, Activation
from keras.models import Sequential

from src.abstract_dataset import *
import pandas as pd

'''
Data set: https://www.kaggle.com/c/digit-recognizer
Train on 42000 samples.
Test the model on kaggle: accuracy on test set = 0.95957
'''
class MNIST(abstract_dataset):

    def __init__(self):
        super(MNIST, self).__init__()

    def read_data(self, trainset_path, testset_path):
        X = pd.read_csv(trainset_path).to_numpy()
        self.set_Xtrain(X[:, 1:]/255)
        self.set_ytrain(X[:, 0].reshape(-1))

        # we do not know the label of samples in test set
        # to compute the accuracy of the model, we will upload the prediction on kaggle
        X = pd.read_csv(testset_path).to_numpy()
        self.set_Xtest(X/255)
        #self.set_ytest(None)

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

    def predict_on_testset_kaggle(self):
        model = self.get_model()
        Ytest = np.argmax(model.predict(self.get_Xtest()), axis=1)

        import csv
        with open('/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/src/saved_models/mnist_test_result.csv', mode='w') as f:
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
    mnist.train_model(train=True, kernel_path='../saved_models/mnist_ann_keras.h5',  model_path='../saved_models/mnist_ann_keras.json',
                      training_path='../../digit-recognizer/train.csv', testing_path='../../digit-recognizer/test.csv')
    '''
    # load model
    model = mnist.load_model(weight_path='../saved_models/mnist_ann_keras.h5', structure_path='../saved_models/mnist_ann_keras.json',
                             trainset_path='../../digit-recognizer/train.csv')
    mnist.read_data(trainset_path='../../digit-recognizer/train.csv', testset_path='../../digit-recognizer/test.csv')
    print(f'model.layers = {model.layers}')
    mnist.predict_on_testset_kaggle()

    # plot an observation
    import matplotlib.pyplot as plt
    x_train, y_train = mnist.get_an_observation(index=516)
    img = x_train.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f'A sample')
    plt.show()