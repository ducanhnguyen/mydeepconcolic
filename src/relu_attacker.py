from src.deepconcolic import *
from src.deepgauge.statistics import *

global logger
logger = logging.getLogger()


class RELU_ATTACKER(SMT_DNN):
    def __init__(self):
        pass

    def create_output_constraints_from_an_observation(self, model_object, x_train, y_train, thread_config):
        hidden_unit_idx = thread_config.attacked_neuron['lower_layer_index']
        hidden_unit_layer = thread_config.attacked_neuron['lower_neuron_index']
        upper_bound = thread_config.attacked_neuron['upper_bound']
        lower_bound = thread_config.attacked_neuron['lower_bound']

        output_constraints = []
        smt_output_constraints = []
        if lower_bound != None:
            output_constraints.append(f"v_{hidden_unit_layer}_{hidden_unit_idx} >= {lower_bound}")
            smt_output_constraints.append(f"(assert (>= v_{hidden_unit_layer}_{hidden_unit_idx} {lower_bound}))")
        if upper_bound != None:
            output_constraints.append(f"v_{hidden_unit_layer}_{hidden_unit_idx} <= {upper_bound}")
            smt_output_constraints.append(f"(assert (<= v_{hidden_unit_layer}_{hidden_unit_idx} {upper_bound}))")
        return output_constraints, smt_output_constraints

    def get_bad_neurons(self):
        # compute bound from the train set
        statistics_computator = STATISTICS()
        statistics_computator.set_model(model_object.get_model())
        statistics_computator.set_X(model_object.get_Xtrain())
        neurons_from_train, _ = statistics_computator.get_range_of_neurons()

        # compute bound from the test set
        statistics_computator = STATISTICS()
        statistics_computator.set_model(model_object.get_model())
        statistics_computator.set_X(model_object.get_Xtest())
        neurons_from_test, _ = statistics_computator.get_range_of_neurons()

        # compare two bounds
        UPPER_IDX = 2
        LOWER_IDX = 3
        bad_neurons = []
        for idx in range(len(neurons_from_train)):
            if neurons_from_train[idx][UPPER_IDX] > neurons_from_test[idx][UPPER_IDX]:
                layer_idx = neurons_from_train[idx][0]
                unit_idx = neurons_from_train[idx][1]
                lower_bound = neurons_from_test[idx][UPPER_IDX]
                upper_bound = neurons_from_train[idx][UPPER_IDX]
                bad_neurons.append([layer_idx, unit_idx, lower_bound, upper_bound])
            if neurons_from_train[idx][LOWER_IDX] < neurons_from_test[idx][LOWER_IDX]:
                layer_idx = neurons_from_train[idx][0]
                unit_idx = neurons_from_train[idx][1]
                lower_bound = neurons_from_train[idx][LOWER_IDX]
                upper_bound = neurons_from_test[idx][LOWER_IDX]
                bad_neurons.append([layer_idx, unit_idx, lower_bound, upper_bound])
        print(f"bad_neurons = {bad_neurons}")
        return bad_neurons


if __name__ == '__main__':
    logging.basicConfig()
    logging.root.setLevel(logging.DEBUG)

    relu_attacker = RELU_ATTACKER()

    model_object = relu_attacker.initialize_dnn_model_from_configuration_file(FASHION_MNIST())
    print(model_object.get_model().summary())

    bad_neurons = relu_attacker.get_bad_neurons()
    print(f"bad_neurons = {bad_neurons}")
    # bad_neurons = [[1, 0, 6.9129443, 8.849387], [1, 1, 17.40716, 18.295204], [1, 3, 8.159637, 9.754293], [1, 4, 5.7981305, 5.9015365], [1, 5, 12.225392, 14.162071], [1, 6, 18.2284, 19.058233], [1, 7, 11.377147, 12.496146], [1, 8, 18.383759, 21.362309], [1, 11, 10.9524555, 12.547186], [1, 12, 10.673858, 12.12503], [1, 13, 14.482282, 16.2899], [1, 14, 13.285749, 17.717718], [1, 15, 11.300592, 11.714331], [1, 16, 11.077839, 13.226842], [1, 18, 13.385778, 14.800182], [1, 19, 12.828551, 13.415758], [1, 20, 13.386428, 15.288409], [1, 21, 13.599655, 14.514343], [1, 22, 8.862251, 10.218973], [1, 23, 10.97079, 13.481878], [1, 24, 15.342602, 18.875645], [1, 26, 9.209093, 9.694359], [1, 27, 10.208138, 10.976832], [1, 28, 7.8196254, 8.575809], [1, 29, 13.506631, 14.779074], [1, 30, 20.905083, 23.854065], [1, 31, 12.796826, 16.815573], [3, 0, 24.814348, 28.842], [3, 1, 27.504719, 34.405518], [3, 3, 25.11091, 26.058344], [3, 4, 24.898163, 25.273428], [3, 5, 26.0245, 26.199827], [3, 6, 10.46668, 11.261339], [3, 7, 29.382933, 32.290066], [3, 8, 22.94414, 23.740458], [3, 9, 21.225649, 23.01664], [3, 10, 15.249615, 16.55212], [3, 12, 20.488012, 25.363205], [3, 13, 21.012957, 26.77353], [3, 14, 13.584735, 16.355663], [3, 15, 38.383133, 41.266354]]bad_neurons = [[1, 0, 6.9129443, 8.849387], [1, 1, 17.40716, 18.295204], [1, 3, 8.159637, 9.754293], [1, 4, 5.7981305, 5.9015365], [1, 5, 12.225392, 14.162071], [1, 6, 18.2284, 19.058233], [1, 7, 11.377147, 12.496146], [1, 8, 18.383759, 21.362309], [1, 11, 10.9524555, 12.547186], [1, 12, 10.673858, 12.12503], [1, 13, 14.482282, 16.2899], [1, 14, 13.285749, 17.717718], [1, 15, 11.300592, 11.714331], [1, 16, 11.077839, 13.226842], [1, 18, 13.385778, 14.800182], [1, 19, 12.828551, 13.415758], [1, 20, 13.386428, 15.288409], [1, 21, 13.599655, 14.514343], [1, 22, 8.862251, 10.218973], [1, 23, 10.97079, 13.481878], [1, 24, 15.342602, 18.875645], [1, 26, 9.209093, 9.694359], [1, 27, 10.208138, 10.976832], [1, 28, 7.8196254, 8.575809], [1, 29, 13.506631, 14.779074], [1, 30, 20.905083, 23.854065], [1, 31, 12.796826, 16.815573], [3, 0, 24.814348, 28.842], [3, 1, 27.504719, 34.405518], [3, 3, 25.11091, 26.058344], [3, 4, 24.898163, 25.273428], [3, 5, 26.0245, 26.199827], [3, 6, 10.46668, 11.261339], [3, 7, 29.382933, 32.290066], [3, 8, 22.94414, 23.740458], [3, 9, 21.225649, 23.01664], [3, 10, 15.249615, 16.55212], [3, 12, 20.488012, 25.363205], [3, 13, 21.012957, 26.77353], [3, 14, 13.584735, 16.355663], [3, 15, 38.383133, 41.266354]]

    # for bad_neuron in bad_neurons:
    #     attacked_neuron = dict()
    #     attacked_neuron['upper_layer_index'] = bad_neuron[0]
    #     attacked_neuron['lower_layer_index'] = bad_neuron[0]
    #     attacked_neuron['upper_unit_index'] = bad_neuron[1]
    #     attacked_neuron['lower_unit_index'] = bad_neuron[1]
    #     attacked_neuron['lower_bound'] = bad_neuron[2]
    #     attacked_neuron['upper_bound'] = bad_neuron[3]
    #
    n_observations = len(model_object.get_Xtrain())
    seeds = np.arange(0, n_observations)
    relu_attacker.generate_samples(model_object=model_object, seeds=seeds, n_threads=get_config(["n_threads"]))
