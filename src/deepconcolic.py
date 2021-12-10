'''
Command:
/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/lib/z3-4.8.5-x64-osx-10.14.2/bin/z3 -smt2 /Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/constraint.txt > /Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/solution.txt
'''
import subprocess
from threading import Thread
from types import SimpleNamespace
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import Sequential

from src.abstract_dataset import abstract_dataset
from src.close_to_edge_detection import close_to_edge
from src.config_parser import *
from src.log_analyzer import is_int
from src.model_loader import initialize_dnn_model
from src.test_summarizer import *
from src.utils import keras_activation, keras_layer, keras_model
from src.utils.edge_detection import is_edge
from src.utils.feature_ranker1d import feature_ranker1d
from src.utils.feature_ranker_2d import RANKING_ALGORITHM
from src.utils.utilities import compute_l0, compute_l2, compute_minimum_change, compute_linf
import time


MINUS_INF = -10000000
INF = 10000000

DELTA_PREFIX_NAME = get_config(["constraint_config", "delta_prefix_name"])

global logger
logger = logging.getLogger()

global graph

NORMALIZATION_FACTOR = 255
smttime  = []


def create_constraint_between_layers(model_object):
    assert (isinstance(model_object, abstract_dataset))
    smt_constraints = []

    model = model_object.get_model()
    if not keras_model.is_ANN(model):
        return

    for current_layer_idx, current_layer in enumerate(model.layers):

        if keras_layer.is_conv(current_layer):
            logger.debug(f'Layer {model.layers[current_layer_idx].name}: support conv later. Terminating...')
            return

        elif current_layer_idx == 0:  # 1: the first layer behind the input layer
            smt_constraints.append(f'\n; (input layer, input of {current_layer.name})')
            weights = current_layer.get_weights()  # the second are biases, the first are weights between two layers
            kernel = weights[0]
            biases = weights[1]
            n_features = kernel.shape[0]
            hidden_units_curr_layer = kernel.shape[1]

            for current_pos in range(hidden_units_curr_layer):
                var = f'u_{current_layer_idx}_{current_pos}'

                smt_constraint = ''
                for feature_idx in range(n_features):
                    previous_var = f'feature_{feature_idx}'
                    weight = kernel[feature_idx][current_pos]

                    weight = weight / NORMALIZATION_FACTOR  # rather than normalizing feature input
                    if feature_idx == 0:
                        smt_constraint = f'(* {previous_var} {weight:.25f}) '
                    else:
                        smt_constraint = f'(+ {smt_constraint} (* {previous_var} {weight:.25f})) '

                smt_constraint = f'(+ {smt_constraint} {biases[current_pos]:.25f}) '
                smt_constraint = f'(assert(= {var} {smt_constraint}))'
                smt_constraints.append(smt_constraint)

        elif keras_layer.is_dense(current_layer):
            pre_layer_idx = current_layer_idx - 1
            pre_layer = model.layers[pre_layer_idx]

            weights = current_layer.get_weights()  # the second are biases, the first are weights between two layers
            kernel = weights[0]
            biases = weights[1]
            hidden_units_pre_layer = kernel.shape[0]
            hidden_units_curr_layer = kernel.shape[1]

            smt_constraints.append(f'\n; ( output of {pre_layer.name}, input of {current_layer.name})')

            if keras_layer.is_dense(pre_layer):  # instance of keras.layers.core.Dense
                # the current layer and the previous layer is dense
                for current_pos in range(hidden_units_curr_layer):
                    # constraints of a hidden unit layer include (1) bias, and (2) weight
                    var = f'u_{current_layer_idx}_{current_pos}'

                    '''
                    type constraint 2
                    '''
                    smt_constraint = f'{var} = '
                    for feature_idx in range(hidden_units_pre_layer):
                        previous_var = f'v_{pre_layer_idx}_{feature_idx}'
                        weight = kernel[feature_idx][current_pos]
                        if feature_idx == 0:
                            smt_constraint = f'(* {previous_var} {weight:.25f}) '
                        else:
                            smt_constraint = f'(+ {smt_constraint} (* {previous_var} {weight:.25f})) '

                    smt_constraint = f'(+ {smt_constraint} {biases[current_layer_idx]:.25f}) '
                    smt_constraint = f'(assert {smt_constraint})'
                    smt_constraints.append(smt_constraint)

            elif keras_layer.is_activation(pre_layer) or keras_layer.is_dropout(
                    pre_layer):  # instance of keras.layers.core.Dense
                for current_pos in range(hidden_units_curr_layer):
                    var = f'u_{current_layer_idx}_{current_pos}'
                    previous_var = f'v_{current_layer_idx - 1}_{current_pos}'

                    smt_constraint = f'(assert(= {var} {previous_var}))'
                    smt_constraints.append(smt_constraint)

        elif keras_layer.is_activation(current_layer):
            pre_layer_idx = current_layer_idx - 1
            pre_layer = model.layers[pre_layer_idx]

            smt_constraints.append(f'\n; (output of {pre_layer.name}, input of {current_layer.name})')

            units = keras_layer.get_number_of_units(model, pre_layer_idx)
            for unit_idx in range(units):
                smt_constraint = f'(assert(= v_{pre_layer_idx}_{unit_idx} u_{current_layer_idx}_{unit_idx}))'
                smt_constraints.append(smt_constraint)

    return smt_constraints


def create_activation_constraints(model_object):
    assert (isinstance(model_object, abstract_dataset))
    smt_constraints = []

    model = model_object.get_model()
    if keras_model.is_ANN(model):
        for current_layer_idx, current_layer in enumerate(model.layers):
            units = keras_layer.get_number_of_units(model, current_layer_idx)

            smt_constraints.append("\n")
            if keras_activation.is_activation(current_layer):
                smt_constraints.append(f'; layer {current_layer.name}')

                # Create the formula based on the type of activation
                if keras_activation.is_relu(current_layer):
                    smt_constraints.append(f'; is relu layer')
                    for unit_idx in range(units):
                        '''
                        constraint type 2
                        v = u>0?x:0
                        '''
                        smt_constraint = f'(= v_{current_layer_idx}_{unit_idx} (ite (>= u_{current_layer_idx}_{unit_idx} 0) u_{current_layer_idx}_{unit_idx} 0))'
                        smt_constraint = f'(assert{smt_constraint})'
                        smt_constraints.append(smt_constraint)

                elif keras_activation.is_tanh(current_layer):
                    smt_constraints.append(f';is tanh layer')
                    for unit_idx in range(units):
                        '''
                        constraint type 2
                        1 - exp(2*a)
                        a = u_{current_layer_idx}_{unit_idx}
                        '''
                        smt_constraint = f'(= v_{current_layer_idx}_{unit_idx} (/ (- 1 (exp (* 2 u_{current_layer_idx}_{unit_idx}))) (+ 1 (exp (* 2 u_{current_layer_idx}_{unit_idx})))))'
                        smt_constraint = f'(assert{smt_constraint})'
                        smt_constraints.append(smt_constraint)

                elif keras_activation.is_softmax(current_layer):
                    smt_constraints.append(f'; is softmax layer')
                    if current_layer_idx == len(model.layers) - 1:
                        # no need to create constraints for the output layer
                        smt_constraints.append(f'; The last layer is softmax: no need to create constraints!\n')
                        continue
                    else:
                        smt_sum_name = f'sum_{current_layer_idx}_{unit_idx}'
                        smt_constraints.append(f'(declare-fun {smt_sum_name} () Real)')

                        smt_sum = ''
                        for unit_idx in range(units):
                            if unit_idx == 0:
                                smt_sum = f'(exp (- 0 u_{current_layer_idx}_{unit_idx}))'
                            else:
                                smt_sum = f'(+ {smt_sum} (exp (- 0 u_{current_layer_idx}_{unit_idx})))'

                        smt_constraint = f'(= {smt_sum_name} {smt_sum})'
                        smt_constraint = f'(assert{smt_constraint})'
                        smt_constraints.append(smt_constraint)

                        #
                        for unit_idx in range(units):
                            smt_constraint = f'(= v_{current_layer_idx}_{unit_idx} (/ (exp (- 0 u_{current_layer_idx}_{unit_idx})) {smt_sum_name}))'
                            smt_constraint = f'(assert{smt_constraint})'
                            smt_constraints.append(smt_constraint)

            elif keras_layer.is_dense(current_layer) and not keras_layer.is_activation(current_layer):
                # the current layer is dense and not an activation
                smt_constraints.append(f'; Layer {current_layer.name}\n; not activation layer')

                for unit_idx in range(units):
                    '''
                    constraint type 2
                    '''
                    smt_constraint = f'(assert(= u_{current_layer_idx}_{unit_idx} v_{current_layer_idx}_{unit_idx}))'
                    smt_constraints.append(smt_constraint)
            else:
                logger.debug(f'Does not support layer {current_layer}')

    elif keras_model.is_CNN(model):
        logger.debug(f'Model is not CNN. Does not support!')

    else:
        logger.debug(f'Unable to detect the type of neural network')

    return smt_constraints


def create_variable_declarations(model_object, type_feature=get_config(["constraint_config", "feature_input_type"])):
    assert (isinstance(model_object, abstract_dataset))
    constraints = []

    model = model_object.get_model()
    if keras_model.is_ANN(model):
        if isinstance(model, Sequential):
            # for input layer
            constraints.append(f'; feature declaration')
            input_shape = model.input_shape  # (None, n_features)
            n_features = input_shape[1]

            constraint = ''
            for feature_idx in range(n_features):
                if type_feature == 'float':
                    constraint += f'(declare-fun feature_{feature_idx} () Real)\t'

                elif type_feature == 'int':
                    constraint += f'(declare-fun feature_{feature_idx} () Int)\t'
                else:
                    logger.debug(f'Does not support the type {type_feature}')
            constraints.append(constraint)

            # for other layers
            for layer_idx, layer in enumerate(model.layers):
                # get number of hidden units
                units = keras_layer.get_number_of_units(model, layer_idx)
                constraints.append(
                    f'\n; Layer ' + layer.name + ' declaration (index = ' + str(layer_idx) + ', #neurons = ' + str(
                        units) + ')')

                # value of neuron in all layers except input layer must be real
                if units is not None:
                    for unit_idx in range(units):
                        before_activation = f'(declare-fun u_{layer_idx}_{unit_idx} () Real)'
                        after_activation = f'(declare-fun v_{layer_idx}_{unit_idx} () Real)'
                        constraints.append(before_activation + '\t\t\t' + after_activation)

    elif keras_model.is_CNN(model):
        logger.debug(f'Model is not CNN. Does not support!')

    else:
        logger.debug(f'Unable to detect the type of neural network')

    return constraints


def create_bound(feature_value, delta_lower_bound, delta_upper_bound, feature_lower_bound, feature_upper_bound,
                 feature_idx):
    tmp = feature_value - delta_lower_bound
    lower = feature_lower_bound if feature_lower_bound > tmp else tmp

    tmp = feature_value + delta_upper_bound
    upper = tmp if feature_upper_bound > tmp else feature_upper_bound

    smt_constraint = f'(assert(and (>= feature_{feature_idx} {lower}) (<= feature_{feature_idx} {upper})))'
    return smt_constraint


def create_equal(var, value):
    return f'(assert(and (>= {var} {value}) (<= {var} {value})))'


def create_feature_constraints_from_an_observation(model_object,
                                                   x_train_shape784_bound1,
                                                   y_train,
                                                   delta_lower_bound: int,
                                                   delta_upper_bound: int,
                                                   feature_lower_bound: int,
                                                   feature_upper_bound: int,
                                                   strategy=get_config(
                                                       ['constraint_config', 'strategy'])):
    assert (isinstance(model_object, abstract_dataset))

    smt_constraints = []
    model = model_object.get_model()
    input_shape = model.input_shape  # (None, n_features)
    smt_constraints.append('; Feature bound constraints')

    n_features = input_shape[1]
    x_train_shape784_bound255 = np.round(x_train_shape784_bound1.reshape(-1) * NORMALIZATION_FACTOR).astype(
        int)  # for MNIST, round to integer range

    if strategy == 'change_nonzero_pixels':
        for feature_idx in range(n_features):
            if x_train_shape784_bound255[feature_idx] == 0:  # fix value
                smt_constraint = create_equal(f'feature_{feature_idx}', 0)
                smt_constraints.append(smt_constraint)

            else:  # change value
                smt_constraint = create_bound(x_train_shape784_bound255[feature_idx], delta_lower_bound, delta_upper_bound,
                                              feature_lower_bound, feature_upper_bound, feature_idx)
                smt_constraints.append(smt_constraint)

    elif strategy == 'change_close_to_edge':
        x_28_28_bound255 = x_train_shape784_bound255.reshape(28, 28)
        for feature_idx in range(n_features):
            row_idx = int(np.floor(feature_idx / 28))
            col_idx = feature_idx - row_idx * 28
            if close_to_edge(row_idx, col_idx, x_28_28_bound255): # changed features
                smt_constraint = create_bound(x_train_shape784_bound255[feature_idx], delta_lower_bound,
                                              delta_upper_bound,
                                              feature_lower_bound, feature_upper_bound, feature_idx)
                smt_constraints.append(smt_constraint)
            else:  # fixed features
                smt_constraint = create_equal(f'feature_{feature_idx}', x_train_shape784_bound255[feature_idx])
                smt_constraints.append(smt_constraint)


    elif strategy == 'change_edge':
        x_28_28_bound255 = x_train_shape784_bound255.reshape(28, 28)
        for feature_idx in range(n_features):
            row_idx = int(np.floor(feature_idx / 28))
            col_idx = feature_idx - row_idx * 28
            if is_edge(row_idx, col_idx, x_28_28_bound255): # changed features
                smt_constraint = create_bound(x_train_shape784_bound255[feature_idx], delta_lower_bound,
                                              delta_upper_bound,
                                              feature_lower_bound, feature_upper_bound, feature_idx)
                smt_constraints.append(smt_constraint)
            else:  # fixed features
                smt_constraint = create_equal(f'feature_{feature_idx}', x_train_shape784_bound255[feature_idx])
                smt_constraints.append(smt_constraint)

    elif strategy.index('_most_important_pixels') > 0:
        n_importances = strategy.replace('_most_important_pixels', '').replace('change_', '')
        if is_int(n_importances):
            n_importances = int(n_importances)

            if n_importances > 0:
                important_features = feature_ranker1d.find_important_features_of_a_sample(
                    input_image=x_train_shape784_bound1.reshape(-1, 784),
                    n_important_features=n_importances,
                    algorithm=RANKING_ALGORITHM.COI,
                    gradient_label=y_train,
                    classifier=model_object.get_model())

                for feature_idx in range(n_features):
                    if feature_idx not in important_features:  # fix value
                        smt_constraint = create_equal(f'feature_{feature_idx}', x_train_shape784_bound255[feature_idx])
                        smt_constraints.append(smt_constraint)
                    else:  # change value
                        smt_constraint = create_bound(x_train_shape784_bound255[feature_idx], delta_lower_bound,
                                                      delta_upper_bound,
                                                      feature_lower_bound, feature_upper_bound, feature_idx)
                        smt_constraints.append(smt_constraint)


    elif strategy == 'default':
        for feature_idx in range(n_features):
            smt_constraint = create_bound(x_train_shape784_bound255[feature_idx], delta_lower_bound, delta_upper_bound,
                                          feature_lower_bound, feature_upper_bound, feature_idx)
            smt_constraints.append(smt_constraint)

    return smt_constraints


def create_output_constraints_from_an_observation(model_object, x_train, y_train, thread_config):
    assert (isinstance(model_object, abstract_dataset))
    constraints = []
    smt_constraints = []
    model = model_object.get_model()

    if keras_model.is_ANN(model):
        assert (x_train.shape[0] == 1 and len(x_train.shape) == 2)
        last_layer_idx = len(model.layers) - 1
        n_classes = keras_layer.get_number_of_units(model,
                                                    last_layer_idx)  # get the number of hidden units in the output layer

        tmp = x_train.reshape(1, -1)

        before_softmax = model.layers[-2]
        intermediate_layer_model = Model(inputs=model.inputs,
                                         outputs=before_softmax.output)

        # with thread_config.graph.as_default():
        # must use when using thread
        prediction = intermediate_layer_model.predict(tmp)
        largest_value = np.max(prediction[0])
        largest_idx = np.argmax(prediction[0])
        left = f'u_{last_layer_idx}_{largest_idx}'

        if n_classes is not None:
            smt_constraints.append(f'; output constraints')
            smt_constraints.append(f'; pre-sotmax = ' + str(prediction[0]).replace('\n', ''))

            smt_type_constraint = get_config(["constraint_config", "output_layer_type_constraint"])
            if smt_type_constraint == 'or':
                '''
                Neuron of true label is smaller than all other neurons
                '''
                smt_constraint = ''
                for class_idx in range(n_classes):
                    if class_idx != largest_idx:
                        if class_idx == 0:
                            smt_constraint = f'(< {left} u_{last_layer_idx}_{class_idx}) '
                        else:
                            smt_constraint = f'(or {smt_constraint} (< {left} u_{last_layer_idx}_{class_idx})) '
                smt_constraint = f'(assert {smt_constraint})'
                logger.debug(f'Output constraint = {smt_constraint}')
                smt_constraints.append(smt_constraint)

            elif smt_type_constraint == 'upper_bound':
                '''
                Neuron of true label is smaller than its value
                '''
                smt_constraint = f'(assert (< {left} {largest_value}) )'
                logger.debug(f'Output constraint = {smt_constraint}')
                smt_constraints.append(smt_constraint)

            elif smt_type_constraint == 'true_index_smaller_than_second_index':
                # get the position of the second largest value
                smallest_value = np.min(prediction[0])
                second_value = smallest_value
                second_idx = np.argmin(prediction[0])

                for idx in range(0, len(prediction[0])):
                    if second_value < prediction[0][idx] < largest_value:
                        second_value = prediction[0][idx]
                        second_idx = idx

                smt_constraint = f'(assert (< {left} u_{last_layer_idx}_{second_idx}) )'
                logger.debug(f'Output constraint = {smt_constraint}')
                smt_constraints.append(smt_constraint)

    elif keras_model.is_CNN(model):
        logger.debug(f'Does not support CNN')

    else:
        logger.debug(f'Unable to detect the type of neural network')

    return constraints, smt_constraints


def define_mathematical_function():
    exp = ['(define-fun exp ((x Real)) Real\n\t(^ 2.718281828459045 x))']
    return exp


def create_constraints_file(model_object, seed_index, thread_config):
    assert (isinstance(model_object, abstract_dataset))
    assert (seed_index >= 0)

    # get an observation
    x_train, y_train = model_object.get_an_observation(seed_index)
    with open(thread_config.seed_file, mode='w') as f:
        seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        seed.writerow(x_train[0])
    with open(thread_config.true_label_seed_file, 'w') as f:
        f.write(str(y_train))

    # generate constraints
    smt_exp = define_mathematical_function()

    variable_types = create_variable_declarations(model_object)

    smt_layers_constraints = create_constraint_between_layers(model_object)

    smt_activation_constraints = create_activation_constraints(model_object)

    delta_lower_bound = get_config(["constraint_config", "delta_lower_bound"])
    delta_upper_bound = get_config(["constraint_config", "delta_upper_bound"])
    feature_lower_bound = get_config(["constraint_config", "feature_lower_bound"])
    feature_upper_bound = get_config(["constraint_config", "feature_upper_bound"])
    smt_input_constraints_modification = create_feature_constraints_from_an_observation(model_object,
                                                                                        x_train,
                                                                                        y_train,
                                                                                        delta_lower_bound,
                                                                                        delta_upper_bound,
                                                                                        feature_lower_bound,
                                                                                        feature_upper_bound)

    output_constraints, smt_output_constraints = create_output_constraints_from_an_observation(model_object, x_train,
                                                                                               y_train, thread_config)

    # create constraint file
    with open(thread_config.constraints_file, 'w') as f:
        f.write(f'(set-option :timeout {get_config(["z3", "time_out"])})\n')
        # f.write(f'(using-params smt :random-seed {ran.randint(1, 101)})\n')

        for constraint in smt_exp:
            f.write(constraint + '\n')

        for constraint in variable_types:
            f.write(constraint + '\n')

        for constraint in smt_layers_constraints:
            f.write(constraint + '\n')

        for constraint in smt_activation_constraints:
            f.write(constraint + '\n')

        for constraint in smt_input_constraints_modification:
            f.write(constraint + '\n')

        f.write('\n')
        for constraint in smt_output_constraints:
            f.write(constraint + '\n')

        f.write('\n')
        f.write('(check-sat)\n')
        f.write('(get-model)\n')


def image_generation(seeds, thread_config, model_object):
    assert (isinstance(model_object, abstract_dataset))
    assert (len(seeds.shape) == 1)

    # append to analyzed seed file
    # just read one time
    # ranges of seed index in different threads are different
    analyzed_seed_indexes = []
    if os.path.exists(thread_config.analyzed_seed_index_file_path):
        analyzed_seed_indexes = pd.read_csv(thread_config.analyzed_seed_index_file_path, header=None)
        analyzed_seed_indexes = analyzed_seed_indexes.to_numpy()

    for seed_index in seeds:
        if seed_index in analyzed_seed_indexes:
            logger.debug(f'{thread_config.thread_name}: Visited seed {seed_index}. Ignore!')
            continue
        else:
            '''
            if the seed is never analyzed before
            '''
            # append the current seed index to the analyzed seed index file

            with open(thread_config.analyzed_seed_index_file_path, mode='a') as f:
                f.write(str(seed_index) + ',')

            # clean the environment
            if os.path.exists(thread_config.true_label_seed_file):
                os.remove(thread_config.true_label_seed_file)
            if os.path.exists(thread_config.seed_index_file):
                os.remove(thread_config.seed_index_file)
            if os.path.exists(thread_config.constraints_file):
                os.remove(thread_config.constraints_file)
            if os.path.exists(thread_config.z3_solution_file):
                os.remove(thread_config.z3_solution_file)
            if os.path.exists(thread_config.z3_normalized_output_file):
                os.remove(thread_config.z3_normalized_output_file)

            logger.debug(f'{thread_config.thread_name}: seed index = {seed_index}')
            with open(thread_config.seed_index_file, mode='w') as f:
                f.write(str(seed_index))

            # generate constraints
            logger.debug(f'{thread_config.thread_name}: generate constraints')
            create_constraints_file(model_object, seed_index, thread_config)

            # call SMT-Solver
            start_smt_time = time.time()
            logger.debug(f'{thread_config.thread_name}: call SMT-Solver to solve the constraints')
            command = f"{thread_config.z3_path} -smt2 {thread_config.constraints_file} > {thread_config.z3_solution_file}"
            logger.debug(f'\t{thread_config.thread_name}: command = {command}')
            os.system(command)
            smttime.append(time.time() - start_smt_time)

            # parse solver solution
            logger.debug(f'{thread_config.thread_name}: parse solver solution')
            tmp1 = thread_config.z3_solution_file
            tmp2 = thread_config.z3_normalized_output_file
            command = get_config(["z3", "z3_solution_parser_command"]) + f' {tmp1} ' + f'{tmp2}'

            # logger.debug(f'{thread_config.thread_name}: \t{command}')
            if platform.system() == 'Darwin' or platform.system() == 'Linux':  # macosx or hpc
                os.system(command)
            else: # windows
                subprocess.run(command, shell=True)

            # comparison
            try:
                candidate_adv = analyze_smt_output(solution_path=thread_config.z3_normalized_output_file)
            except:
                candidate_adv = None

            if candidate_adv is not None:
                # export candidate adv to file
                candidate_adv_csv_path = f'../result/{thread_config.dataset}/{seed_index}.csv'
                with open(candidate_adv_csv_path, mode='w') as f:
                    seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    seed.writerow(candidate_adv)

                # check
                is_valid = is_valid_adv(model_object=model_object, config=thread_config,
                                        csv_new_image_path=candidate_adv_csv_path)
                if is_valid:
                    with open(thread_config.selected_seed_index_file_path, mode='a') as f:
                        f.write(str(seed_index) + ',')
                else:
                    os.remove(candidate_adv_csv_path)
            else:
                logger.debug(f'The constraints have no solution')
            logger.debug('--------------------------------------------------')


def set_up_config(thread_idx):
    assert (thread_idx >= 0)
    OLD = '{thread_idx}'

    config = SimpleNamespace(**dict())
    config.dataset = get_config(["dataset"])
    config.seed_file = get_config(["files", "seed_file_path"]).replace(OLD, str(thread_idx))
    config.true_label_seed_file = get_config(["files", "true_label_seed_file_path"]). \
        replace(OLD, str(thread_idx))
    config.seed_index_file = get_config(["files", "seed_index_file_path"]).replace(OLD, str(thread_idx))
    config.constraints_file = get_config(["z3", "constraints_file_path"]).replace(OLD, str(thread_idx))
    config.z3_solution_file = get_config(["z3", "z3_solution_path"]).replace(OLD, str(thread_idx))
    config.z3_normalized_output_file = get_config(["z3", "z3_normalized_solution_path"]) \
        .replace(OLD, str(thread_idx))
    config.z3_path = get_config(["z3", "z3_solver_path"]).replace(OLD, str(thread_idx))
    config.graph = tf.compat.v1.get_default_graph()
    config.graph.finalize()  # make graph read-only
    config.analyzed_seed_index_file_path = get_config(["files", "analyzed_seed_index_file_path"]).replace(OLD, str(
        thread_idx))
    config.selected_seed_index_file_path = get_config(["files", "selected_seed_index_file_path"]).replace(OLD, str(
        thread_idx))
    config.thread_name = f'thread_{thread_idx}'
    config.should_plot = False  # should be False when running in multithread
    config.z3_solution_parser_command = get_config(["z3", "z3_solution_parser_command"])
    config.new_image_file_path = get_config(["files", "new_image_file_path"])
    config.comparison_file_path = get_config(["files", "comparison_file_path"])
    return config


def generate_samples(model_object):
    """
    Generate adversarial samples
    :return:
    """
    '''
    construct model from file
    '''
    start_seed = get_config([model_object.get_name_dataset(), "start_seed"])
    end_seed = get_config([model_object.get_name_dataset(), "end_seed"])

    seeds = np.arange(start_seed, end_seed)
    logger.debug("Before prioritization: len seeds = " + str(len(seeds)))
    seeds = priority_seeds(seeds, model_object)
    logger.debug("After prioritization: len seeds = " + str(len(seeds)))

    '''
    generate adversarial samples
    '''
    n_threads = get_config(["n_threads"])
    if n_threads >= 2:
        # prone to error
        n_single_thread_seeds = int(np.floor((len(seeds)) / n_threads))
        logger.debug(f'n_single_thread_seeds = {n_single_thread_seeds}')
        threads = []

        for thread_idx in range(n_threads):
            # get range of seed in the current thread
            if thread_idx == n_threads - 1:
                thread_seeds_idx = np.arange(n_single_thread_seeds * (n_threads - 1), len(seeds))
            else:
                thread_seeds_idx = np.arange(n_single_thread_seeds * thread_idx,
                                             n_single_thread_seeds * (thread_idx + 1))

            # read the configuration of a thread
            thread_config = set_up_config(thread_idx)
            thread_config.image_shape = model_object.get_image_shape()

            # create new thread
            t = Thread(target=image_generation, args=(seeds[thread_seeds_idx], thread_config, model_object))
            threads.append(t)

        # start all threads
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    elif n_threads == 1:
        # read the configuration of a thread
        main_thread_config = set_up_config(0)
        main_thread_config.image_shape = model_object.get_image_shape()
        image_generation(seeds, main_thread_config, model_object)


def priority_seeds(seeds, model_object, threshold=9999):
    '''
    Remove wrongly predicted samples and ranking the others
    :param seeds:
    :param model_object:
    :return:
    '''
    delta_arr = []
    # for seed in seeds:
    ori = model_object.get_Xtrain()[seeds]
    true_labels = model_object.get_ytrain()[seeds]
    pred = model_object.get_model().predict(ori.reshape(-1, 784))

    selected_seeds = []
    for idx in range(len(seeds)):

        # ignore wrongly predicted sample
        predicted_label = np.argmax(pred[idx])
        if predicted_label != true_labels[idx]:
            print(f'Seed {seeds[idx]}: Predict wrongly. Ignoring...')
            continue

        # print(idx)
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        pred_sort, labels = zip(*sorted(zip(pred[idx], labels)))
        last_idx = len(pred_sort) - 1
        delta = pred_sort[last_idx] - pred_sort[last_idx - 1]

        # if delta < 1: # to reduce the size of attacking set
        if delta < threshold:  # ignore
            delta_arr.append(delta)
            selected_seeds.append(seeds[idx])

    # for k, v in zip(delta, seeds):
    #     print(f'{k}, {v}')
    delta_arr, selected_seeds = zip(*sorted(zip(delta_arr, selected_seeds)))
    for idx in range(len(delta_arr)):
        print(f'{idx}: prob(true label) - prob(second largest label) = {delta_arr[idx]}')
    return np.asarray(selected_seeds)  # 1-D array


def create_summary(directory: str, model_object, all_seeds):
    # get path of all adv files
    adv_arr_path = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            adv_arr_path.append(os.path.join(directory, filename))
        else:
            continue

    adv_dict = {}  # key: seed index, value: array of pixels
    selected_seed_index_arr = []
    for idx in range(len(adv_arr_path)):
        seed_index = os.path.basename(adv_arr_path[idx]).replace(".csv", "")
        if not is_int(seed_index):
            continue
        seed_index = int(seed_index)
        selected_seed_index_arr.append(seed_index)

        with open(adv_arr_path[idx]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in csv_reader:
                if len(row) > 0:
                    adv_dict[seed_index] = np.asarray(row).astype(int)

    # load model
    X_train = model_object.get_Xtrain()  # for MNIST: shape = (60k, 784)
    Y_pred = model_object.get_model().predict(X_train.reshape(-1, 784))  # for MNIST: shape = (N, 10)

    before_softmax = model_object.get_model().layers[-2]
    intermediate_layer_model = Model(inputs=model_object.get_model().inputs,
                                     outputs=before_softmax.output)
    Y_pred_intermediate_layer_model = intermediate_layer_model.predict(X_train.reshape(-1, 784))

    y_true = model_object.get_ytrain()  # for MNIST: shape = (, N)
    l0_arr = []
    l2_arr = []
    linf_arr = []
    seed_arr = []
    minimum_change_arr = []
    pred_label_arr = []
    adv_label_arr = []
    position_adv_arr = []

    first_largest_prob = []
    second_largest_prob = []
    third_largest_prob = []
    fourth_largest_prob = []
    fifth_largest_prob = []
    last_largest_prob = []
    # delta_first_prod_vs_second_prob = []
    # delta_first_prod_vs_third_prob = []
    # delta_first_prod_vs_fourth_prob = []

    for seed_index in all_seeds:
        print(seed_index)
        predicted_prob = Y_pred_intermediate_layer_model[seed_index]
        seed_arr.append(seed_index)

        if seed_index not in selected_seed_index_arr:
            l0_arr.append(None)
            l2_arr.append(None)
            linf_arr.append(None)
            minimum_change_arr.append(None)
            adv_label_arr.append(None)
            position_adv_arr.append(None)
            pred_label_arr.append(None)

            true_pred_sorted = sorted(predicted_prob, reverse=True)
            first_largest_prob.append(true_pred_sorted[0])
            second_largest_prob.append(true_pred_sorted[1])
            third_largest_prob.append(true_pred_sorted[2])
            fourth_largest_prob.append(true_pred_sorted[3])
            fifth_largest_prob.append(true_pred_sorted[4])
            last_largest_prob.append(true_pred_sorted[-1])

            # delta_first_prod_vs_second_prob.append(np.abs(true_pred_sorted[0] - true_pred_sorted[1]))
            # delta_first_prod_vs_third_prob.append(np.abs(true_pred_sorted[0] - true_pred_sorted[2]))
            # delta_first_prod_vs_fourth_prob.append(np.abs(true_pred_sorted[0] - true_pred_sorted[3]))
        else:
            # the adv must be valid and the seed must be predicted correctly
            pred_label = np.argmax(predicted_prob)
            adv = adv_dict[seed_index]  # for MNIST: [0..255]
            adv_pred = model_object.get_model().predict((adv / NORMALIZATION_FACTOR).reshape(-1, 784))[0]
            adv_label = np.argmax(adv_pred)
            if pred_label == adv_label or pred_label != y_true[seed_index]:  # just confirm
                print(f"PROBLEM with seed {seed_index}!")

                l0_arr.append(None)
                l2_arr.append(None)
                linf_arr.append(None)
                minimum_change_arr.append(None)
                adv_label_arr.append(None)
                position_adv_arr.append(None)
                pred_label_arr.append(None)

                true_pred_sorted = sorted(predicted_prob, reverse=True)
                first_largest_prob.append(true_pred_sorted[0] )
                second_largest_prob.append(true_pred_sorted[1] )
                third_largest_prob.append(true_pred_sorted[2])
                fourth_largest_prob.append(true_pred_sorted[3])
                fifth_largest_prob.append(true_pred_sorted[4])
                last_largest_prob.append(true_pred_sorted[-1])
                # delta_first_prod_vs_second_prob.append(np.abs(true_pred_sorted[0] - true_pred_sorted[1]))
                # delta_first_prod_vs_third_prob.append(np.abs(true_pred_sorted[0] - true_pred_sorted[2]))
                # delta_first_prod_vs_fourth_prob.append(np.abs(true_pred_sorted[0] - true_pred_sorted[3]))

                continue

            adv_label_arr.append(adv_label)
            pred_label_arr.append(pred_label)

            # compute distance of adv and its ori
            ori = X_train[seed_index]  # [0..1]

            l0 = compute_l0((ori * NORMALIZATION_FACTOR).astype(int), adv)
            l0_arr.append(l0)

            l2 = compute_l2(ori, adv / NORMALIZATION_FACTOR)
            l2_arr.append(l2)

            linf = compute_linf(ori, adv / NORMALIZATION_FACTOR)
            linf_arr.append(linf)

            minimum_change = compute_minimum_change(ori, adv / NORMALIZATION_FACTOR)
            minimum_change_arr.append(minimum_change)

            # position of adv in the probability of original prediction
            position_adv = -9999999999999
            labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            true_pred_sorted, labels_sorted_by_prob = zip(*sorted(zip(predicted_prob, labels), reverse=True))
            for j in range(len(labels_sorted_by_prob)):
                if labels_sorted_by_prob[j] == adv_label:
                    position_adv = j + 1  # start from 1
                    break
            position_adv_arr.append(position_adv)

            first_largest_prob.append(true_pred_sorted[0] )
            second_largest_prob.append(true_pred_sorted[1] )
            third_largest_prob.append(true_pred_sorted[2])
            fourth_largest_prob.append(true_pred_sorted[3])
            fifth_largest_prob.append(true_pred_sorted[4])
            last_largest_prob.append(true_pred_sorted[-1])

            # delta_first_prod_vs_second_prob.append(true_pred_sorted[0] - true_pred_sorted[1])
            # delta_first_prod_vs_third_prob.append(true_pred_sorted[0] - true_pred_sorted[2])
            # delta_first_prod_vs_fourth_prob.append(true_pred_sorted[0] - true_pred_sorted[3])

            # export image comparison
            fig = plt.figure()
            nrow = 1
            ncol = 2
            ori = ori.reshape(28, 28)
            fig1 = fig.add_subplot(nrow, ncol, 1)
            fig1.title.set_text(
                f'origin \nindex = {seed_index},\nlabel {pred_label}, acc = {Y_pred[seed_index][pred_label]}')
            plt.imshow(ori, cmap="gray")

            adv = (adv / NORMALIZATION_FACTOR).reshape(28, 28)
            fig2 = fig.add_subplot(nrow, ncol, 2)
            fig2.title.set_text(
                f'adv\nlabel {adv_label}, acc = {adv_pred[adv_label]}\n l0 = {l0}, l2 = ~{np.round(l2, 2)}')
            plt.imshow(adv, cmap="gray")

            png_comparison_image_path = directory + f'/{seed_index}_comparison.png'
            plt.savefig(png_comparison_image_path, pad_inches=0, bbox_inches='tight')

    # export to csv
    summary_path = directory + '/summary.csv'
    with open(summary_path, mode='w') as f:
        seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # seed.writerow(['seed', 'l0', 'l2', 'l_inf', 'minimum_change', 'true_label', 'adv_label',
        #                'position_adv_label_in_original_pred', 'delta_first_prod_vs_second_prob',
        #                'delta_first_prod_vs_third_prob', 'delta_first_prod_vs_fourth_prob'])
        # for i in range(len(seed_arr)):
        #     seed.writerow([seed_arr[i], l0_arr[i], l2_arr[i], linf_arr[i], minimum_change_arr[i], pred_label_arr[i],
        #                    adv_label_arr[i], position_adv_arr[i], delta_first_prod_vs_second_prob[i],
        #                    delta_first_prod_vs_third_prob[i],
        #                    delta_first_prod_vs_fourth_prob[i]])
        seed.writerow(['seed', 'l0', 'l2', 'l_inf', 'minimum_change', 'true_label', 'adv_label',
                       'position_adv_label_in_original_pred', 'first_largest',
                       'second_largest', 'third_largest', 'fourth_largest', 'fifth_largest', 'last_largest'])
        for i in range(len(seed_arr)):
            seed.writerow([seed_arr[i], l0_arr[i], l2_arr[i], linf_arr[i], minimum_change_arr[i], pred_label_arr[i],
                           adv_label_arr[i], position_adv_arr[i], first_largest_prob[i],
                           second_largest_prob[i], third_largest_prob[i], fourth_largest_prob[i],fifth_largest_prob[i],
                           last_largest_prob[i]])
    return summary_path


def compute_prob(model_object):
    ori_arr = [44, 45, 46, 47, 48, 49, 50]
    for ori_idx in ori_arr:
        ori = model_object.get_Xtrain()[ori_idx]
        pred = model_object.get_model().predict(ori.reshape(1, 784))[0]
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        pred, labels = zip(*sorted(zip(pred, labels)))
        # print(ori_idx)
        # print(pred)

        last_idx = len(pred) - 1
        largest = labels[last_idx]
        second = labels[last_idx - 1]
        third = labels[last_idx - 2]
        print(
            f'{ori_idx}. true label = {largest} ({pred[last_idx]}), second = {second} ({pred[last_idx - 1]}), third = {third} ({pred[last_idx - 2]})')
        # print()


if __name__ == '__main__':
    logging.basicConfig()
    logging.root.setLevel(logging.DEBUG)

    model_object = initialize_dnn_model()

    start = time.time()
    generate_samples(model_object)
    end = time.time()
    TOTAL = end - start
    print(f"total time {TOTAL}")
    print(f"smt time {smttime}, avg = {np.average(smttime)}")

    start_seed = int(get_config([model_object.get_name_dataset(), "start_seed"]))
    end_seed = int(get_config([model_object.get_name_dataset(), "end_seed"]))
    create_summary(get_config(["output_folder"]), model_object,
                   all_seeds=range(start_seed, end_seed))
