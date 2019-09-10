import pickle

from keras.layers import Dense, Activation
from keras.models import Sequential
from skimage import color
from sklearn.utils import shuffle

from src.abstract_dataset import *


class CIFAR10(abstract_dataset):
    def __init__(self):
        super(CIFAR10, self).__init__()

    def read_data(self, training_path, testing_path):
        base_path = '/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/cifar-10'
        features_1, labels_1 = self.load_cfar10_batch(base_path, 1)
        features_2, labels_2 = self.load_cfar10_batch(base_path, 2)
        features_3, labels_3 = self.load_cfar10_batch(base_path, 3)
        features_4, labels_4 = self.load_cfar10_batch(base_path, 4)
        features_5, labels_5 = self.load_cfar10_batch(base_path, 5)

        # merge all
        features = np.concatenate((features_1, features_2, features_3, features_4, features_5), axis=0)
        labels = np.concatenate((labels_1, labels_2, labels_3, labels_4, labels_5), axis=0)
        print(f'features.shape = {features.shape}')
        print(f'labels.shape = {labels.shape}')

        full_data_set = np.concatenate((labels, features), axis=1)
        full_data_set = shuffle(full_data_set)
        print(f'full_data_set.shape = {full_data_set.shape}')

        # split
        limit = 45000
        train_set = full_data_set[:limit, :]
        print(f'train_set shape = {train_set.shape}')

        X_train = train_set[:, 1:]
        print(f'X_train shape = {X_train.shape}')
        self.set_Xtrain(X_train)

        y_train = train_set[:, 0]
        self.set_ytrain(y_train)

        test_set = full_data_set[limit:, :]
        print(f'test_set shape = {test_set.shape}')

        X_test = test_set[:, 1:]
        self.set_Xtest(X_test)

        y_test = test_set[:, 0]
        self.set_ytest(y_test)

        return self.get_Xtrain(), self.get_ytrain(), self.get_Xtest(), self.get_ytest()

    def load_cfar10_batch(self, cifar10_dataset_folder_path, batch_id):
        # https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c

        with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')

        width_image = 32
        height_image = 32
        channel_in = 3  # rgb
        features = batch['data']
        # .reshape((len(batch['data']), channel_in, width_image, height_image)) \
        # .transpose(0, 2, 3, 1)  # convert into the format: (channel_in, width, height, channel_out)
        labels = np.asarray(batch['labels']).reshape(-1, 1)  # convert list of lables into array, then reshape
        return features, labels

    def create_model(self, input_shape):
        assert (input_shape > 0)
        model = Sequential()

        #model.add(Dense(512, name='dense_0', input_dim=input_shape))
        #model.add(Activation('relu', name='relu_0'))

        #model.add(Dense(256, name='dense_1', input_dim=input_shape))
        #model.add(Activation('relu', name='relu_1'))

        model.add(Dense(128, name='dense_2'))
        model.add(Activation('relu', name='relu_2'))

        model.add(Dense(64, name='dense_3'))
        model.add(Activation('relu', name='relu_3'))

        model.add(Dense(32, name='dense_4'))
        model.add(Activation('relu', name='relu_4'))

        model.add(Dense(16, name='dense_5'))
        model.add(Activation('relu', name='relu_5'))

        model.add(Dense(self.get_num_classes(), name='dense_n'))
        model.add(Activation('softmax', name='softmax'))

        self.set_model(model)
        return model

    def flatten_bw(self, X):
        W, H, C, N = X.shape
        Xbw = np.zeros((N, W * H))

        for i in range(N):
            bw = np.array(color.rgb2gray(X[:, :, :, i]))
            Xbw[i] = bw.flatten()

        return Xbw

    def flatten_rgb(self, X):
        W, H, C, N = X.shape
        Xbw = np.zeros((N, W * H * C))

        for i in range(N):
            bw = np.array(X[:, :, :, i])
            Xbw[i] = np.reshape(a=bw, newshape=(W * H * C))

        Xbw /= 255

        return Xbw


if __name__ == '__main__':
    cifar = CIFAR10()

    '''
    train_set, test_set = cifar.create_dataset(
        '/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/cifar-10/train_set.csv',
        '/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/cifar-10/test_set.csv'
    )
    '''
    cifar.set_num_classes(10)
    cifar.train_model(train=True, kernel_path='../saved_models/cifar10_ann_keras.h5',
                      model_path='../saved_models/cifar10_ann_keras.json',
                      training_path='../../cifar-10/train_set.csv', testing_path='../../cifar-10/test_set.csv',
                      batch_size = 64, nb_epoch = 100, learning_rate=1e-3)