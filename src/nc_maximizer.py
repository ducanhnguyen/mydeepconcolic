from src.deepconcolic import *
from src.deepgauge.statistics import *
import logging.config
import logging

global logger

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)

'''
Given a threshold, let find all neurons which the upper bound is below this threshold (bad neurons).
For each bad neuron, let find adversarial images to make its upper bound over the threshold.
'''


class NC_MAXIMIZER(SMT_DNN):
    def __init__(self):
        pass

    def create_output_constraints_from_an_observation(self, model_object, x_train, y_train, thread_config):
        assert (isinstance(thread_config, ThreadConfig))
        assert (isinstance(model_object, ABSTRACT_DATASET))

        layer_idx = thread_config.attacked_neuron['layer_index']
        neuron_idx = thread_config.attacked_neuron['neuron_index']
        upper_bound = thread_config.attacked_neuron['upper_bound']
        lower_bound = thread_config.attacked_neuron['lower_bound']

        output_constraints = []
        smt_output_constraints = []
        if lower_bound != None:
            output_constraints.append(f"v_{layer_idx}_{neuron_idx} >= {lower_bound}")
            smt_output_constraints.append(f"(assert (>= v_{layer_idx}_{neuron_idx} {lower_bound}))")

        if upper_bound != None:
            output_constraints.append(f"v_{layer_idx}_{neuron_idx} <= {upper_bound}")
            smt_output_constraints.append(f"(assert (<= v_{layer_idx}_{neuron_idx} {upper_bound}))")

        logger.debug(f"output constraints: {smt_output_constraints}")
        return output_constraints, smt_output_constraints

    def get_neurons_under_a_threshold_from_Xtest(self, model_object, threshold):
        '''
        Find bad neurons
        :param model_object:
        :param threshold:
        :return:
        '''
        assert (isinstance(model_object, ABSTRACT_DATASET))

        # compute bound from the test set
        statistics_computator = STATISTICS()
        statistics_computator.set_model(model_object.get_model())
        statistics_computator.set_X(model_object.get_Xtest())
        neurons_from_test, _ = statistics_computator.get_range_of_neurons()

        # compare two bounds
        UPPER_IDX = 2
        bad_neurons = []
        for idx in range(len(neurons_from_test)):
            if neurons_from_test[idx][UPPER_IDX] < threshold:
                layer_idx = neurons_from_test[idx][0]
                unit_idx = neurons_from_test[idx][1]
                bad_neurons.append([layer_idx, unit_idx])
        return bad_neurons


if __name__ == '__main__':
    nc_maximizer = NC_MAXIMIZER()

    # construct a model
    model_object = nc_maximizer.initialize_dnn_model_from_configuration_file(FASHION_MNIST())
    print(model_object.get_model().summary())

    # find all bad neurons below a given threshold
    threshold = 100  # all neurons are bad neurons
    bad_neurons = nc_maximizer.get_neurons_under_a_threshold_from_Xtest(model_object, threshold=threshold)
    logger.debug(f"bad_neurons under threshold {threshold} from Xtest = {bad_neurons}")

    # save threshold to configuration file
    config_controller = ConfigController()

    # For each bad neuron, let generate adversarial images
    for bad_neuron in bad_neurons:
        logger.debug(f"Updating bad neuron {bad_neuron} to the configuration file")
        config_controller.update_and_write(key="attacked_neuron.layer_index", value=bad_neuron[0])
        config_controller.update_and_write(key="attacked_neuron.neuron_index", value=bad_neuron[1])

        # initialize statistics computator
        statistics_computator = STATISTICS()
        statistics_computator.set_model(model_object.get_model())
        statistics_computator.set_X(model_object.get_Xtest())

        # set threshold
        option = 2
        if option == 1:
            config_controller.update_and_write(key="attacked_neuron.lower_bound", value=threshold)
            config_controller.update_and_write(key="attacked_neuron.upper_bound", value=threshold * 5)
        elif option == 2:
            min, max = statistics_computator.get_range_of_a_neuron(neuron_layer_index=bad_neuron[0],
                                                                   neuron_unit_index=bad_neuron[1])
            delta = 0.1
            threshold = max + delta
            config_controller.update_and_write(key="attacked_neuron.lower_bound", value=threshold)
            config_controller.update_and_write(key="attacked_neuron.upper_bound", value=threshold * 5)

        # find the best seeds
        logger.debug(f"Finding the best seeds for the current neuron")
        top_obervation_indexes = statistics_computator.find_top_observations(percentage=0.01,
                                                                             neuron_layer_index=bad_neuron[0],
                                                                             neuron_unit_index=bad_neuron[1])
        logger.debug(f"Top observation indexes in order of neuron value = {top_obervation_indexes}")

        # should run on a thread rather than multi-thread
        nc_maximizer.generate_samples(
            model_object=model_object,
            seeds=top_obervation_indexes,
            n_threads=ConfigController().get_config(["n_threads"]),
            just_find_one_seed=False)

        logger.debug("Deleting analyzed_seed_index_file_path")
        if os.path.exists(ConfigController().get_config(['files', 'analyzed_seed_index_file_path'])):
            os.remove(ConfigController().get_config(['files', 'analyzed_seed_index_file_path']))

        logger.debug("Deleting selected_seed_index_file_path")
        if os.path.exists(ConfigController().get_config(['files', 'selected_seed_index_file_path'])):
            os.remove(ConfigController().get_config(['files', 'selected_seed_index_file_path']))