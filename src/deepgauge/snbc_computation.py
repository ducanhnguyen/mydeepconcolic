from src.deepgauge.statistics import *
from src.saved_models.mnist_ann_keras import MNIST


class SNAC_COVERAGE(abstract_coverage_computation):
    def __init__(self):
        super(SNAC_COVERAGE, self).__init__()
        self.X_train = None

    def compute_bound_from_train_set(self):
        statistics_computator = STATISTICS()
        statistics_computator.set_model(self.get_model())
        statistics_computator.set_X(self.get_Xtrain())
        bounds = statistics_computator.draw_boxplot_of_units()['bound']
        return bounds

    def compute_snac_neuron_coverage(self):
        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer
        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))
        bounds = self.compute_bound_from_train_set()

        num_upper_corner_neuron = 0
        num_units = 0

        for idx in range(n_coverage_layers):
            layer_index_in_model = activation_layers[idx][1]
            n_units = keras_layer.get_number_of_units(self.get_model(), layer_index_in_model)
            num_units+= n_units

            for unit_idx in range(n_units):
                # get lower and upper
                bound = bounds[STATISTICS().create_key(layer_idx=layer_index_in_model, unit_idx=unit_idx)]
                for j in range(n_observations):
                    if prediction[idx][j][unit_idx] > bound['upper']:
                        num_upper_corner_neuron += 1
                        break

        # final coverage computation
        print(f'num_upper_corner_neuron = {num_upper_corner_neuron}')
        print(f'num_units = {num_units}')
        final_coverage = (num_upper_corner_neuron) / (num_units)
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
        testset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/result/mnist fgsm e=1:255 *(1 to 9)/original_test_plus_expansion.csv')
    model = model_object.get_model()

    # compute neuron coverage
    computator = SNAC_COVERAGE()
    computator.set_model(model_object.get_model())
    computator.set_X(model_object.get_Xtest())
    computator.set_Xtrain(model_object.get_Xtrain())
    coverage = computator.compute_snac_neuron_coverage()
    print(f'coverage = {coverage}')