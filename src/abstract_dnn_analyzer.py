import threading
from abc import ABC, abstractmethod
from threading import Thread

from src.ConfigController import ConfigController
from src.ThreadConfig import ThreadConfig
from src.saved_models.fashion_mnist_ann_keras import *
from src.test_summarizer import *

global logger
logger = logging.getLogger()

global graph

class ABSTRACT_DNN_ANALYZER(ABC):
    MINUS_INF = -10000000
    INF = 10000000

    def __init__(self):
        pass

    @abstractmethod
    def create_constraint_between_layers(self, model_object, upper_layer_index=INF):
        pass

    @abstractmethod
    def create_activation_constraints(self, model_object, upper_layer_index=INF, upper_unit_index=INF,
                                      lower_unit_index=MINUS_INF):
        pass

    @abstractmethod
    def create_variable_declarations(self, model_object,type_feature):
        pass

    @abstractmethod
    def create_feature_constraints_from_an_observation(self, model_object, x_train, delta):
        pass

    @abstractmethod
    def create_output_constraints_from_an_observation(self, model_object, x_train, y_train, thread_config):
        pass

    @abstractmethod
    def create_bound_of_feature_constraints(self, model_object,
                                            feature_lower_bound, feature_upper_bound,
                                            delta_lower_bound, delta_upper_bound, delta_prefix):
        pass

    @abstractmethod
    def adversarial_image_generation(self, seeds, thread_config, model_object, just_find_one_seed = False):
        pass

    @abstractmethod
    def initialize_dnn_model_from_configuration_file(self, model_object):
        pass

    def read_seeds_from_config(self, model_object):
        start_seed = ConfigController().get_config([model_object.get_name_dataset(), "start_seed"])
        end_seed = ConfigController().get_config([model_object.get_name_dataset(), "end_seed"])
        seeds = np.arange(start_seed, end_seed)
        return seeds

    def read_seeds_from_file(self, csv_file):
        selected_seed_indexes = pd.read_csv(csv_file, header=None).to_numpy()
        selected_seed_indexes = selected_seed_indexes.reshape(-1)
        return selected_seed_indexes

    def define_mathematical_function(self):
        exp = ['(define-fun exp ((x Real)) Real\n\t(^ 2.718281828459045 x))']
        return exp

    def create_constraints_file(self, model_object, seed_index, thread_config):
        assert (isinstance(model_object, ABSTRACT_DATASET))
        assert (seed_index >= 0)
        assert (isinstance(thread_config, ThreadConfig))

        # get an observation from test set
        x_train, y_train = model_object.get_an_observation_from_test_set(seed_index)
        with open(thread_config.seed_file, mode='w') as f:
            seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            seed.writerow(x_train[0])
        with open(thread_config.true_label_seed_file, 'w') as f:
            f.write(str(y_train))

        # generate constraints
        smt_expression = self.define_mathematical_function()

        variable_types = self.create_variable_declarations(model_object, thread_config.feature_input_type)

        layers_constraints, smt_layers_constraints = self.create_constraint_between_layers(
            model_object,
            upper_layer_index=thread_config.attacked_neuron['layer_index']
        )

        activation_constraints, smt_activation_constraints = self.create_activation_constraints(
            model_object,
            upper_layer_index=thread_config.attacked_neuron['layer_index'],
            upper_unit_index=thread_config.attacked_neuron['neuron_index'],
            lower_unit_index=thread_config.attacked_neuron['neuron_index'])

        input_constraints, smt_input_constraints = self.create_feature_constraints_from_an_observation(
            model_object,
            x_train,
            thread_config.delta_prefix_name)

        output_constraints, smt_output_constraints = self.create_output_constraints_from_an_observation(
            model_object,
            x_train,
            y_train,
            thread_config)

        smt_bound_input_constraints = self.create_bound_of_feature_constraints(
            model_object=model_object,
            delta_prefix=thread_config.delta_prefix_name,
            feature_lower_bound=thread_config.feature_lower_bound,
            feature_upper_bound=thread_config.feature_upper_bound,
            delta_lower_bound=thread_config.delta_lower_bound,
            delta_upper_bound=thread_config.delta_upper_bound)

        # create constraint file
        with open(thread_config.constraints_file, 'w') as f:
            f.write(f'(set-option :timeout {thread_config.z3_time_out})\n')
            # f.write(f'(using-params smt :random-seed {ran.randint(1, 101)})\n')

            for constraint in smt_expression:
                f.write(constraint + '\n')

            for constraint in variable_types:
                f.write(constraint + '\n')

            for constraint in smt_layers_constraints:
                f.write(constraint + '\n')

            for constraint in smt_activation_constraints:
                f.write(constraint + '\n')

            for constraint in smt_input_constraints:
                f.write(constraint + '\n')

            for constraint in smt_bound_input_constraints:
                f.write(constraint + '\n')

            for constraint in smt_output_constraints:
                f.write(constraint + '\n')

            f.write('(check-sat)\n')
            f.write('(get-model)\n')

    def set_up_config(self, thread_idx):
        thread_config = ThreadConfig()
        thread_config.setId(thread_idx)
        thread_config.setConfigFile(config_path='./config_ubuntu.json')
        return thread_config

    def generate_samples(self, model_object, seeds, n_threads, just_find_one_seed):
        """
        Generate adversarial samples
        :return:
        """
        # remove the analyzed seeds
        analyzed_seed_indexes = []
        analyzed_seed_file = ConfigController().get_config(['files', 'analyzed_seed_index_file_path'])
        if os.path.exists(analyzed_seed_file):
            analyzed_seed_indexes = pd.read_csv(analyzed_seed_file, header=None).to_numpy()
        not_analyzed_seeds = []
        for seed in seeds:
            if seed not in analyzed_seed_indexes:
                not_analyzed_seeds.append(seed)
        not_analyzed_seeds = np.asarray(not_analyzed_seeds)
        logger.debug(f'Putting {len(not_analyzed_seeds)} seeds into {n_threads} threads')

        # run in multithread
        if n_threads >= 2:
            # prone to error
            n_single_thread_seeds = int(np.floor((len(not_analyzed_seeds)) / n_threads))
            logger.debug(f'n_single_thread_seeds = {n_single_thread_seeds}')
            threads = []

            for thread_idx in range(n_threads):
                # get range of seed in the current thread
                if thread_idx == n_threads - 1:
                    thread_seeds = np.arange(n_single_thread_seeds * (n_threads - 1), len(not_analyzed_seeds))
                else:
                    thread_seeds = np.arange(n_single_thread_seeds * thread_idx,
                                             n_single_thread_seeds * (thread_idx + 1))

                # read the configuration of a thread
                thread_config = self.set_up_config(thread_idx)
                thread_config.image_shape = model_object.get_image_shape()

                # create new thread
                t = Thread(target=self.adversarial_image_generation, args=(thread_seeds, thread_config, model_object, just_find_one_seed))
                threads.append(t)

            # start all threads
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        elif n_threads == 1:
            # read the configuration of a thread
            main_thread_config = self.set_up_config(0)
            main_thread_config.image_shape = model_object.get_image_shape()

            # run on the main thread
            self.adversarial_image_generation(not_analyzed_seeds, main_thread_config, model_object, just_find_one_seed)
