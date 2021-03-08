import keras
import numpy as np
from keras.models import model_from_json


class abstract_dataset:
    def __init__(self):
        self.__Xtrain = None  # 2-D array: (N_samples, N_features)
        self.__Xtest = None  # 2-D array: (N_samples, N_features)
        self.__ytrain = None  # 1-D array, where each element is the label of an observation on the training set
        self.__ytest = None  # 1-D array, where each element is the label of an observation on the training set
        self.__model = None  # Keras model
        self.__num_classes = None  # int
        self.__image_shape = None  # 2-D (black-white image) or 3-D array (rgb image)
        self.__name_dataset = None  # String
        self.__selected_seed_index_file_path = None  # String

    def create_model(self, input_shape):
        pass

    def read_data(self, trainset_path, testset_path):
        pass

    def get_an_observation(self, index):
        assert (index >= 0 and len(self.get_Xtrain().shape) == 2 and len(self.get_ytrain().shape) == 1)
        return self.get_Xtrain()[index].reshape(1, -1), self.get_ytrain()[index]

    def train_model(self, train, kernel_path, model_path, training_path, testing_path, batch_size=64, nb_epoch=30,
                    learning_rate=1e-3):
        assert (train == True or train == False)
        assert (kernel_path != None and training_path != None and testing_path != None)

        self.read_data(training_path, testing_path)
        model = self.create_model(input_shape=len(self.get_Xtrain()[0]))

        # train ANN model
        if train:
            # compiling
            model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=learning_rate),
                          metrics=['accuracy'])

            model.fit(self.get_Xtrain(), self.category2indicator(self.get_ytrain()),
                      batch_size=batch_size, epochs=nb_epoch, verbose=1)
            score = model.evaluate(self.get_Xtrain(), self.category2indicator(self.get_ytrain()), verbose=0)
            print('Overall training score:', score[0])
            print('Accuracy on train set:', score[1])

            if self.get_ytest() is not None and self.get_Xtest() is not None:
                score = model.evaluate(self.get_Xtest(), self.category2indicator(self.get_ytest()), verbose=0)
                print('Overall test score:', score[0])
                print('Accuracy on test set:', score[1])

            # save model
            model.save_weights(kernel_path)

            with open(model_path, "w") as json_file:
                json_file.write(model.to_json())

        else:
            model = self.load_model(kernel_path, model_path, training_path)

        self.set_model(model)
        return model

    def score(Y, Yhat):
        y = np.argmax(Y, axis=1)
        yhat = np.argmax(Yhat, axis=1)
        return np.mean(y == yhat)

    def load_model(self, weight_path, structure_path, trainset_path):
        assert (weight_path != None and trainset_path != None)

        # load structure of model
        json_file = open(structure_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # Load weight from file
        model.load_weights(weight_path)

        self.set_model(model)
        return model

    def category2indicator(self, y):
        Y = np.zeros(shape=(y.shape[0], self.get_num_classes()))

        for idx, item in enumerate(y):
            if item == 10:
                Y[idx][0] = 1
            else:
                Y[idx][item] = 1

        return Y

    def get_num_classes(self):
        return self.__num_classes

    def set_num_classes(self, num_class):
        self.__num_classes = num_class

    def set_Xtrain(self, Xtrain):
        assert (len(Xtrain.shape) == 2)
        self.__Xtrain = Xtrain

    def get_Xtrain(self):
        return self.__Xtrain

    def set_ytrain(self, y_train):
        assert (len(y_train.shape) == 1)
        self.__ytrain = y_train

    def get_ytrain(self):
        return self.__ytrain

    def get_Xtest(self):
        return self.__Xtest

    def set_Xtest(self, Xtest):
        assert (len(Xtest.shape) == 2)
        self.__Xtest = Xtest

    def get_ytest(self):
        return self.__ytest

    def set_ytest(self, ytest):
        assert (len(ytest.shape) == 1)
        self.__ytest = ytest

    def set_model(self, model):
        self.__model = model

    def get_model(self):
        return self.__model

    def set_image_shape(self, image_shape):
        self.__image_shape = image_shape

    def get_image_shape(self):
        return self.__image_shape

    def get_name_dataset(self):
        return self.__name_dataset

    def set_name_dataset(self, dataset):
        self.__name_dataset = dataset

    def get_selected_seed_index_file_path(self):
        return self.__selected_seed_index_file_path

    def set_selected_seed_index_file_path(self, selected_seed_index_file_path):
        self.__selected_seed_index_file_path = selected_seed_index_file_path
