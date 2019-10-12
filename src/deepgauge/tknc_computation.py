from src.deepgauge.statistics import *


class TKNC_COVERAGE(abstract_coverage_computation):
    def __init__(self):
        super(TKNC_COVERAGE, self).__init__()

    def create_key(self, layer_idx, unit_idx):
        return "layer_idx:" + str(layer_idx) + ",unit_idx:" + str(unit_idx)

    def compute_tknc_neuron_coverage(self, k_arr):
        activation_layers, prediction = self.load_model()
        n_coverage_layers = len(activation_layers) - 1  # -1: ignore the last activation layer
        n_observations = len(self.get_X().reshape(len(self.get_X()), -1))

        active_neurons_dict = dict()
        for k in k_arr:
            active_neurons_dict[k] = set()
        num_units = 0

        # compute the number of units in the model
        for idx in range(n_coverage_layers):
            layer_index_in_model = activation_layers[idx][1]
            n_units = keras_layer.get_number_of_units(self.get_model(), layer_index_in_model)
            num_units += n_units

        #
        for j in range(n_observations):
            for idx in range(n_coverage_layers):
                layer_index_in_model = activation_layers[idx][1]
                n_units = keras_layer.get_number_of_units(self.get_model(), layer_index_in_model)
                neuron_values = []
                neuron_idxes = []
                for unit_idx in range(n_units):
                    neuron_values.append(prediction[idx][j][unit_idx])
                    neuron_idxes.append(unit_idx)

                # sort
                from operator import itemgetter
                # the first element: neuron values, the second element: neuron indexes
                sorted_lists = [list(x) for x in
                                zip(*sorted(zip(neuron_values, neuron_idxes), key=itemgetter(0), reverse=True))]

                for k in k_arr:
                    for element_idx in range(k):
                        neuron_value = sorted_lists[0][element_idx]
                        neuron_index = sorted_lists[1][element_idx]
                        key = self.create_key(layer_idx=layer_index_in_model, unit_idx=neuron_index)
                        active_neurons_dict[k].add(key)

        # final coverage computation
        for key, value in active_neurons_dict.items():
            print(f'num of active_neurons = {len(value)}')
            print(f'active_neurons = {value}')
            print(f'num_units = {num_units}')
            coverage = len(value) / num_units
            print(f'key = {key}, value = {coverage}\n')


if __name__ == '__main__':
    # construct model 1
    model_object = FASHION_MNIST()
    model_object.set_num_classes(10)
    model = model_object.load_model(
        weight_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/fashion_mnist_ann_keras_f1_original.h5',
        structure_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/fashion_mnist_ann_keras_f1_original.json',
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/train.csv')
    model_object.read_data(
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/train.csv',
        #testset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/fashion_mnist/test.csv')
        testset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist_f1/original_test_plus_expansion.csv')
    print(model.summary())

    # compute coverage
    computator = TKNC_COVERAGE()
    computator.set_model(model_object.get_model())
    computator.set_X(model_object.get_Xtest())
    coverage = computator.compute_tknc_neuron_coverage(k_arr=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f'coverage = {coverage}')
