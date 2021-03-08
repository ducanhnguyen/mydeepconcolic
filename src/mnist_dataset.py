import pandas as pd

from src.abstract_dataset import abstract_dataset


class mnist_dataset(abstract_dataset):

    def read_data(self, trainset_path, testset_path):
        X = pd.read_csv(trainset_path).to_numpy()
        NORMALIZATION_FACTOR = 255

        self.set_Xtrain(X[:, 1:] / NORMALIZATION_FACTOR)
        self.set_ytrain(X[:, 0].reshape(-1))
        print(f'training set: {len(X)} samples')

        X = pd.read_csv(testset_path).to_numpy()
        self.set_Xtest(X[:, 1:] / NORMALIZATION_FACTOR)
        self.set_ytest(X[:, 0].reshape(-1))
        print(f'test set: {len(X)} samples')

        return self.get_Xtrain(), self.get_ytrain(), self.get_Xtest(), self.get_ytest()