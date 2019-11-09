from src.deepgauge.abstract_coverage_computation import *
from tensorflow.python import keras

from src.saved_models.fashion_mnist_ann_keras import *
from src.saved_models.mnist_ann_keras import MNIST
from src.utils import keras_layer
from src.deepgauge.statistics import *

class K_MULTISECTION_NEURON_COVERAGE(abstract_coverage_computation):
    def __init__(self):
        super(K_MULTISECTION_NEURON_COVERAGE, self).__init__()
        self.X_train = None

    def compute_bound_from_train_set(self):
        statistics_computator = STATISTICS()
        statistics_computator.set_model(self.get_model())
        statistics_computator.set_X(self.get_Xtrain())
        bounds = statistics_computator.draw_boxplot_of_units()['bound']
        return bounds

    def compute_k_multisection_neuron_coverage(self, k):
        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer
        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))

        bounds = self.compute_bound_from_train_set()
        covs = []
        N = 0
        for idx in range(n_coverage_layers):
            layer_index_in_model = activation_layers[idx][1]
            n_units = keras_layer.get_number_of_units(self.get_model(), layer_index_in_model)

            for unit_idx in range(n_units):
                # get lower and upper
                #print(len(values[idx][unit_idx]))
                bound = bounds[STATISTICS().create_key(layer_idx=layer_index_in_model, unit_idx=unit_idx)]
                u = bound['upper']
                l = bound['lower']
                fragment = np.abs(u - l) / k
                print(f'layer {idx}, unit {unit_idx}: [{l}, {u})')

                # run over
                N += 1
                n_active_times = 0

                for threshold in range(k):
                    # compute the fragment
                    start = l + threshold * fragment
                    if threshold == k - 1:
                        end = u
                    else:
                        end = start + fragment
                    #
                    is_active = False
                    for j in range(n_observations):
                        if prediction[idx][j][unit_idx] >= start and prediction[idx][j][unit_idx] < end:
                            is_active = True
                            break
                    if is_active:
                        n_active_times += 1
                assert (n_active_times<=k)

                # compute coverage
                covs.append(n_active_times)

        # final deepgauge
        final_coverage = np.sum(covs) / (k*N)
        print(final_coverage)
        return final_coverage

    def set_Xtrain(self, X_train):
        self.X_train = X_train

    def get_Xtrain(self):
        return self.X_train

if __name__ == '__main__':
    # construct model 1
    model_object = MNIST()
    model_object.set_num_classes(10)
    model = model_object.load_model(
        weight_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/mnist_ann_keras_f1_original.h5',
        structure_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/mnist_ann_keras_f1_original.json',
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/digit-recognizer/train.csv')
    model_object.read_data(
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/digit-recognizer/train.csv',
        testset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/result/mnist fgsm start=0, stop=0.05, step=1: 250/original_test_plus_expansion.csv')
    model = model_object.get_model()
    assert (isinstance(model, Sequential))
    print(model.summary())

    # compute neuron coverage
    nc_computator = K_MULTISECTION_NEURON_COVERAGE()
    nc_computator.set_model(model_object.get_model())
    nc_computator.set_X(model_object.get_Xtest())
    nc_computator.set_Xtrain(model_object.get_Xtrain())
    nc_computator.compute_k_multisection_neuron_coverage(k=1000)
