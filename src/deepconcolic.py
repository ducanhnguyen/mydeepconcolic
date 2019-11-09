'''
Command:
/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/lib/z3-4.8.5-x64-osx-10.14.2/bin/z3 -smt2 /Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/constraint.txt > /Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/solution.txt

ALgorithm

Step 1. Run over the training set, for each sample on the training set:
Step 2.     Generate a SMT-Lib file to describe the relationship between the current sample and the prediction
Step 3.     Add new constraint to the SMT-Lib file
Step 4.     Call SMT-Solver
Step 5.     Save the solution of SMT-Solver if it has

Limitation
- Do not support CNN
- Support ANN with low complexity
- Just support linear activation function (i,e., relu)
- Poor performance (i.e., MNIST ~ 2 days)
- Low coverage: NC, NBC, KMNC
'''

import time

import matplotlib
from keras.models import Model

from src.abstract_dnn_analyzer import *
from src.deepgauge.statistics import STATISTICS
from src.saved_models.fashion_mnist_ann_keras import *
from src.test_summarizer import *
from src.utils import keras_activation, keras_layer, keras_model


class SMT_DNN(ABSTRACT_DNN_ANALYZER):
    def __init__(self):
        pass

    def create_constraint_between_layers(self, model_object, upper_layer_index=10000000):
        assert (isinstance(model_object, ABSTRACT_DATASET))
        constraints = []
        smt_constraints = []

        model = model_object.get_model()
        for current_layer_idx, current_layer in enumerate(model.layers):
            if current_layer_idx <= upper_layer_index:  # when we just want to consider some specific layers
                if current_layer_idx == 0:
                    '''
                    Case: if the current layer is the input layer.
                    '''
                    constraints.append(f'; (input layer, {current_layer.name})')
                    smt_constraints.append(f'; (input layer, {current_layer.name})')

                    if keras_model.is_ANN(model):
                        weights = current_layer.get_weights()  # the second are biases, the first are weights between two layers
                        kernel = weights[0]
                        biases = weights[1]
                        n_features = kernel.shape[0]
                        hidden_units_curr_layer = kernel.shape[1]

                        for current_pos in range(hidden_units_curr_layer):
                            var = f'u_{current_layer_idx}_{current_pos}'

                            '''
                            type constraint 1 (just for reading)
                            '''
                            constraint = f'{var} = '
                            for feature_idx in range(n_features):
                                previous_var = f'feature_{feature_idx}'
                                weight = kernel[feature_idx][current_pos]
                                weight = weight / 255  # because the feature input is in range of [0..255]
                                constraint += f'{previous_var} * {weight:.25f} + '

                            constraint += f'{biases[current_pos]:.25f}'
                            constraints.append(constraint)

                            '''
                            type constraint 2 (for SMT-solver)
                            '''
                            smt_constraint = ''
                            for feature_idx in range(n_features):
                                previous_var = f'feature_{feature_idx}'
                                weight = kernel[feature_idx][current_pos]

                                weight = weight / 255  # because the feature input is in range of [0..255]
                                if feature_idx == 0:
                                    smt_constraint = f'(* {previous_var} {weight:.25f}) '
                                else:
                                    smt_constraint = f'(+ {smt_constraint} (* {previous_var} {weight:.25f})) '

                            smt_constraint = f'(+ {smt_constraint} {biases[current_pos]:.25f}) '
                            smt_constraint = f'(assert(= {var} {smt_constraint}))'
                            smt_constraints.append(smt_constraint)

                    else:
                        logger.debug(f'Do not support this kind of DNN')
                        continue

                elif current_layer_idx > 0:
                    '''
                    Case: The current layer is not the input layer
                    '''
                    if keras_layer.is_2dconv(current_layer):
                        logger.debug(f'Layer {model.layers[current_layer_idx].name}: support conv later')

                    elif keras_layer.is_dense(current_layer):
                        pre_layer_idx = current_layer_idx - 1
                        pre_layer = model.layers[pre_layer_idx]

                        weights = current_layer.get_weights()  # the second are biases, the first are weights between two layers
                        kernel = weights[0]
                        biases = weights[1]
                        hidden_units_pre_layer = kernel.shape[0]
                        hidden_units_curr_layer = kernel.shape[1]

                        constraints.append(f'; ({pre_layer.name}, {current_layer.name})')
                        smt_constraints.append(f'; ({pre_layer.name}, {current_layer.name})')

                        if keras_layer.is_dense(pre_layer):  # instance of keras.layers.core.Dense
                            # the current layer and the previous layer is dense
                            for current_pos in range(hidden_units_curr_layer):
                                # constraints of a hidden unit layer include (1) bias, and (2) weight
                                var = f'u_{current_layer_idx}_{current_pos}'

                                '''
                                type constraint 1
                                '''
                                constraint = f'{var} = '
                                for feature_idx in range(hidden_units_pre_layer):
                                    previous_var = f'v_{current_layer_idx - 1}_{feature_idx}'
                                    weight = kernel[feature_idx][current_pos]
                                    constraint += f'{previous_var} * {weight:.25f} + '

                                constraint += f'{biases[current_pos]:.25f}'
                                constraints.append(constraint)

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

                                constraint = f'{var} = {previous_var}'
                                constraints.append(constraint)

                                smt_constraint = f'(assert(= {var} {previous_var}))'
                                smt_constraints.append(smt_constraint)

                    elif (not keras_layer.is_dense(current_layer) and keras_layer.is_activation(current_layer)) \
                            or keras_layer.is_dropout(current_layer):
                        pre_layer_idx = current_layer_idx - 1
                        pre_layer = model.layers[pre_layer_idx]

                        constraints.append(f'; ({pre_layer.name}, {current_layer.name})')
                        smt_constraints.append(f'; ({pre_layer.name}, {current_layer.name})')

                        units = keras_layer.get_number_of_units(model, pre_layer_idx)
                        for unit_idx in range(units):
                            '''
                            type constraint 1
                            '''
                            constraint = f'v_{pre_layer_idx}_{unit_idx} = u_{current_layer_idx}_{unit_idx}'
                            constraints.append(constraint)

                            '''
                            type constraint 2
                            '''
                            smt_constraint = f'(assert(= v_{pre_layer_idx}_{unit_idx} u_{current_layer_idx}_{unit_idx}))'
                            smt_constraints.append(smt_constraint)

        return constraints, smt_constraints

    def create_activation_constraints(self, model_object, upper_layer_index=10000000, upper_unit_index=10000000,
                                      lower_unit_index=-1):
        assert (isinstance(model_object, ABSTRACT_DATASET))
        constraints = []
        smt_constraints = []

        model = model_object.get_model()
        if keras_model.is_ANN(model):
            for current_layer_idx, current_layer in enumerate(model.layers):
                if current_layer_idx <= upper_layer_index:  # added
                    units = keras_layer.get_number_of_units(model, current_layer_idx)

                    if keras_activation.is_activation(current_layer):
                        constraints.append(f'; {current_layer.name}')
                        smt_constraints.append(f'; {current_layer.name}')

                        # Create the formula based on the type of activation
                        if keras_activation.is_relu(current_layer):
                            for unit_idx in range(units):
                                if unit_idx >= lower_unit_index and unit_idx <= upper_unit_index:  # added
                                    '''
                                    constraint type 1
                                    '''
                                    constraint = f'v_{current_layer_idx}_{unit_idx} = u_{current_layer_idx}_{unit_idx} >= 0 ? u_{current_layer_idx}_{unit_idx} : 0 '
                                    constraints.append(constraint)

                                    '''
                                    constraint type 2
                                    v = u>0?x:0
                                    '''
                                    smt_constraint = f'(= v_{current_layer_idx}_{unit_idx} (ite (>= u_{current_layer_idx}_{unit_idx} 0) u_{current_layer_idx}_{unit_idx} 0))'
                                    smt_constraint = f'(assert{smt_constraint})'
                                    smt_constraints.append(smt_constraint)

                        elif keras_activation.is_tanh(current_layer):
                            for unit_idx in range(units):
                                if unit_idx >= lower_unit_index and unit_idx <= upper_unit_index:  # added
                                    '''
                                    constraint type 1
                                    '''
                                    constraint = f'v_{current_layer_idx}_{unit_idx} = (1 - exp(2*u_{current_layer_idx}_{unit_idx})) / (1 + exp(2*u_{current_layer_idx}_{unit_idx}))'
                                    constraints.append(constraint)

                                    '''
                                    constraint type 2
                                    1 - exp(2*a)
                                    a = u_{current_layer_idx}_{unit_idx}
                                    '''
                                    smt_constraint = f'(= v_{current_layer_idx}_{unit_idx} (/ (- 1 (exp (* 2 u_{current_layer_idx}_{unit_idx}))) (+ 1 (exp (* 2 u_{current_layer_idx}_{unit_idx})))))'
                                    smt_constraint = f'(assert{smt_constraint})'
                                    smt_constraints.append(smt_constraint)

                        elif keras_activation.is_softmax(current_layer):
                            if current_layer_idx == len(model.layers) - 1:
                                # no need to create constraints for the output layer
                                smt_constraints.append(f'; The last layer is softmax: no need to create constraints!')
                                continue
                            else:
                                '''
                                constraint type 1
                                '''
                                # add denominator
                                sum = f'sum_{current_layer_idx}_{unit_idx}'
                                constraint = f'{sum} = '
                                for unit_idx in range(units):
                                    if unit_idx >= lower_unit_index and unit_idx <= upper_unit_index:  # added
                                        if unit_idx == units - 1:
                                            constraint += f'exp(-u_{current_layer_idx}_{unit_idx})'
                                        else:
                                            constraint += f'exp(-u_{current_layer_idx}_{unit_idx}) + '
                                constraints.append(constraint)

                                for unit_idx in range(units):
                                    if unit_idx >= lower_unit_index and unit_idx <= upper_unit_index:  # added
                                        constraint = f'v_{current_layer_idx}_{unit_idx} = exp(-u_{current_layer_idx}_{unit_idx}) / {sum}'
                                        constraints.append(constraint)
                                '''
                                constraint type 2
                                '''
                                smt_sum_name = f'sum_{current_layer_idx}_{unit_idx}'
                                smt_constraints.append(f'(declare-fun {smt_sum_name} () Real)')

                                smt_sum = ''
                                for unit_idx in range(units):
                                    if unit_idx >= lower_unit_index and unit_idx <= upper_unit_index:  # added
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

                    elif (keras_layer.is_dense(current_layer) and not keras_layer.is_activation(current_layer)) or \
                            keras_layer.is_dropout(current_layer):
                        # the current layer is dense and not an activation
                        constraints.append(f'; {current_layer.name}')
                        smt_constraints.append(f'; {current_layer.name}')

                        for unit_idx in range(units):
                            if unit_idx >= lower_unit_index and unit_idx <= upper_unit_index:  # added
                                '''
                                constraint type 1
                                '''
                                constraint = f'u_{current_layer_idx}_{unit_idx} = v_{current_layer_idx}_{unit_idx}'
                                constraints.append(constraint)

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

        return constraints, smt_constraints

    def create_variable_declarations(self, model_object, type_feature):
        assert (isinstance(model_object, ABSTRACT_DATASET))
        constraints = []
        constraints.append(f'; variable declaration')

        model = model_object.get_model()
        if keras_model.is_ANN(model):
            if isinstance(model, keras.engine.sequential.Sequential):
                # for input layer
                input_shape = model.input_shape  # (None, n_features)
                n_features = input_shape[1]

                for feature_idx in range(n_features):
                    if type_feature == 'float':
                        constraint = f'(declare-fun feature_{feature_idx} () Real)'
                        constraints.append(constraint)
                    elif type_feature == 'int':
                        constraint = f'(declare-fun feature_{feature_idx} () Int)'
                        constraints.append(constraint)
                    else:
                        logger.debug(f'Does not support the type {type_feature}')

                # for other layers
                for layer_idx, layer in enumerate(model.layers):
                    # get number of hidden units
                    units = keras_layer.get_number_of_units(model, layer_idx)

                    # value of neuron in all layers except input layer must be real
                    if units != None:
                        for unit_idx in range(units):
                            before_activation = f'(declare-fun u_{layer_idx}_{unit_idx} () Real)'
                            constraints.append(before_activation)

                            after_activation = f'(declare-fun v_{layer_idx}_{unit_idx} () Real)'
                            constraints.append(after_activation)

        elif keras_model.is_CNN(model):
            logger.debug(f'Model is not CNN. Does not support!')

        else:
            logger.debug(f'Unable to detect the type of neural network')

        return constraints

    def create_feature_constraints_from_an_observation(self, model_object, x_train, delta):
        assert (isinstance(model_object, ABSTRACT_DATASET))
        constraints = []
        smt_constraints = []

        model = model_object.get_model()
        if keras_model.is_ANN(model):

            if isinstance(model, keras.engine.sequential.Sequential):
                input_shape = model.input_shape  # (None, n_features)
                constraints.append('; input value')
                smt_constraints.append('; input value')

                n_features = input_shape[1]
                x_train = x_train.reshape(-1)
                if n_features == x_train.shape[0]:
                    for feature_idx in range(n_features):
                        '''
                        constraint type 1
                        '''
                        constraint = f'feature_{feature_idx}  >= {x_train[feature_idx] * 255}-{delta}_{feature_idx} ' \
                                     f' and ' \
                                     f' feature_{feature_idx} <= {x_train[feature_idx] * 255} + {delta}_{feature_idx}'
                        constraints.append(constraint)

                        '''
                        constraint type 2
                        '''
                        smt_constraints.append(f'(declare-fun {delta}_{feature_idx} () Int)')
                        smt_constraint = \
                            f'(assert(and ' \
                            + f'(>= feature_{feature_idx} (- {x_train[feature_idx] * 255} {delta}_{feature_idx})) ' \
                              f'(<= feature_{feature_idx} (+ {x_train[feature_idx] * 255} {delta}_{feature_idx}))' \
                            + f'))'
                        smt_constraints.append(smt_constraint)
                else:
                    logger.debug(f'The size of sample does not match!')
            else:
                logger.debug(f'The input model must be sequential')

        elif keras_model.is_CNN(model):
            logger.debug(f'Model is not CNN. Does not support!')

        else:
            logger.debug(f'Unable to detect the type of neural network')

        return constraints, smt_constraints

    def create_output_constraints_from_an_observation(self, model_object, x_train, y_train, thread_config):
        assert (isinstance(model_object, ABSTRACT_DATASET))
        assert (isinstance(thread_config, ThreadConfig))
        constraints = []
        smt_constraints = []
        model = model_object.get_model()

        if keras_model.is_ANN(model):
            assert (x_train.shape[0] == 1 and len(x_train.shape) == 2)

            last_layer_idx = len(model.layers) - 1
            last_layer = model.layers[last_layer_idx]
            true_label = y_train
            left = f'u_{last_layer_idx}_{true_label}'

            # get the number of hidden units in the output layer
            n_classes = keras_layer.get_number_of_units(model, last_layer_idx)

            if n_classes != None:
                '''
                constraint type 1
                '''
                right = ''
                for class_idx in range(n_classes):
                    if class_idx == n_classes - 1:
                        right += f'u_{last_layer_idx}_{class_idx}'
                    else:
                        right += f'u_{last_layer_idx}_{class_idx}, '
                right = f'max({right})'
                constraint = f'{left} = {right}'
                constraints.append(f'; output constraints')
                constraints.append(constraint)

                smt_type_constraint = ConfigController().get_config(
                    ["constraint_config", "output_layer_type_constraint"])

                if smt_type_constraint == 'or':
                    smt_constraint = ''
                    for class_idx in range(n_classes):
                        if class_idx != true_label:
                            if class_idx == 0:
                                smt_constraint = f'(< {left} u_{last_layer_idx}_{class_idx}) '
                            else:
                                smt_constraint = f'(or {smt_constraint} (< {left} u_{last_layer_idx}_{class_idx})) '
                    smt_constraints.append(f'; output constraints')
                    smt_constraint = f'(assert {smt_constraint})'
                    smt_constraints.append(smt_constraint)

                elif smt_type_constraint == 'upper_bound':
                    '''
                    Add a constraint related to the bound of output. 
                    '''
                    tmp = x_train.reshape(1, -1)

                    before_softmax = model.layers[-2]
                    intermediate_layer_model = Model(inputs=model.inputs,
                                                     outputs=before_softmax.output)
                    lock.acquire()
                    with thread_config.graph.as_default():
                        # must use when using thread
                        prediction = intermediate_layer_model.predict(tmp)
                    lock.release()
                    # logger.debug(f'The prediction of the current seed (before softmax): {prediction}')

                    smt_constraints.append(f'; output constraints')
                    true_label = open(thread_config.true_label_seed_file, "r").readline()
                    old_probability = prediction[0][int(true_label)]
                    smt_constraint = f'(assert (< {left} {old_probability}) )'
                    logger.debug(f'Output constraint = {smt_constraint}')
                    smt_constraints.append(smt_constraint)

        elif keras_model.is_CNN(model):
            logger.debug(f'Does not support CNN')

        else:
            logger.debug(f'Unable to detect the type of neural network')

        return constraints, smt_constraints

    def create_bound_of_feature_constraints(self, model_object,
                                            feature_lower_bound, feature_upper_bound,
                                            delta_lower_bound, delta_upper_bound, delta_prefix):
        assert (feature_lower_bound <= feature_upper_bound)
        assert (delta_lower_bound <= delta_upper_bound)
        assert (delta_prefix != None and delta_prefix != '')
        assert (isinstance(model_object, ABSTRACT_DATASET))
        smt_constraints = []

        model = model_object.get_model()
        if keras_model.is_ANN(model):
            first_layer = model.layers[0]
            weights = first_layer.get_weights()  # in which the second are biases, the first is kernel
            kernel = weights[0]
            n_features = kernel.shape[0]

            '''
            add delta constraint type
            '''
            # use :.25f to avoid 'e' in the value of bound, for example, 1e-3
            smt_constraints.append(f'; bound of delta')
            for feature_idx in range(n_features):
                delta_constraint = f'(assert(and (<= {delta_prefix}_{feature_idx} {delta_upper_bound:.10f}) (>= {delta_prefix}_{feature_idx} {delta_lower_bound:.10f})))'
                smt_constraints.append(delta_constraint)

            '''
            add constraint type 2
            '''
            smt_constraints.append(f'; bound of features')
            for feature_idx in range(n_features):
                # Note 1: use :.25f to avoid 'e' in the value of bound, for example, 1e-3
                # for example: '(assert(and (>= feature_494 0) (<= feature_494 0.1)))'
                # corresponding to: feature_494>=0 and feature_494<=0.1
                # Note 2: Because the model is ANN, the features fed into a 1-D array.
                smt_constraint = f'(assert(and ' \
                                 f'(>= feature_{feature_idx} {feature_lower_bound:.10f}) ' \
                                 f'(<= feature_{feature_idx} {feature_upper_bound:.10f})' \
                                 f'))'
                smt_constraints.append(smt_constraint)

        elif keras_model.is_CNN(model):
            logger.debug(f'Does not support CNN')

        else:
            logger.debug(f'Unable to detect the type of neural network')

        return smt_constraints

    def adversarial_image_generation(self, seeds, thread_config, model_object, just_find_one_seed):
        assert (isinstance(model_object, ABSTRACT_DATASET))
        assert (isinstance(thread_config, ThreadConfig))
        assert (len(seeds.shape) == 1)

        for seed_index in seeds:
            seed_index = int(seed_index)
            thread_config.setSeedIndex(seed_index)
            thread_config.load_config_from_file()

            logger.debug(
                f'{thread_config.thread_name}: seed index = {seed_index}, delta_bound = [{thread_config.delta_lower_bound}, {thread_config.delta_upper_bound}]')
            if thread_config.attacked_neuron['enabled'] == "True":
                logger.debug(
                    f"attacked neuron: layer index = {thread_config.attacked_neuron['layer_index']}, neuron index = {thread_config.attacked_neuron['neuron_index']}")

            # just for testing
            statistics_computator = STATISTICS()
            statistics_computator.set_model(model_object.get_model())
            statistics_computator.set_X(model_object.get_Xtest()) # not xtrain
            neuron_value = statistics_computator.get_range_of_a_neuron_given_an_observation(
                neuron_unit_index=thread_config.attacked_neuron['neuron_index'],
                neuron_layer_index=thread_config.attacked_neuron['layer_index'],
                observation_index=seed_index
            )
            logger.debug(f"Value of neuron given the observation {seed_index} "
                         f"given neuron_layer_index = {thread_config.attacked_neuron['layer_index']} "
                         f"and neuron_unit_index = {thread_config.attacked_neuron['neuron_index']} "
                         f"= {neuron_value}")
            # end

            stop = False
            if just_find_one_seed:
                logger.debug(f"just_find_one_seed is turned on")
                # get selected seed before
                if os.path.exists(thread_config.selected_seed_index_file_path):
                    selected_seeds = pd.read_csv(thread_config.selected_seed_index_file_path, header=None).to_numpy()
                    if len(selected_seeds) >= 1:
                        logger.debug(f"Found seeds: selected seeds = {selected_seeds}. Terminate!")

                        logger.debug(f"Deleting {thread_config.analyzed_seed_index_file_path}")
                        if os.path.exists(thread_config.analyzed_seed_index_file_path):
                            os.remove(thread_config.analyzed_seed_index_file_path)

                        logger.debug(f"Deleting {thread_config.selected_seed_index_file_path}")
                        if os.path.exists(thread_config.selected_seed_index_file_path):
                            os.remove(thread_config.selected_seed_index_file_path)
                        stop = True
                        break
            if not stop:
                '''
                if the seed is never analyzed before
                '''
                # append the current seed index to the analyzed seed index file
                lock.acquire()
                with open(thread_config.analyzed_seed_index_file_path, mode='a') as f:
                    logger.debug(f"Adding seed {seed_index} to {thread_config.analyzed_seed_index_file_path}")
                    f.write(str(seed_index) + ',')
                lock.release()

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

                with open(thread_config.seed_index_file, mode='w') as f:
                    f.write(str(seed_index))

                # generate constraints
                logger.debug(f'{thread_config.thread_name}: generate constraints')
                self.create_constraints_file(model_object, seed_index, thread_config)

                # call SMT-Solver
                logger.debug(f'{thread_config.thread_name}: call SMT-Solver to solve the constraints')
                command = f"{thread_config.z3_path} -smt2 {thread_config.constraints_file} > {thread_config.z3_solution_file}"
                logger.debug(f'\t{thread_config.thread_name}: command = {command}')
                os.system(command)

                # parse solver solution
                logger.debug(f'{thread_config.thread_name}: parse solver solution')
                command = thread_config.z3_solution_parser_command + f' {thread_config.z3_solution_file} ' + f'{thread_config.z3_normalized_output_file}'

                logger.debug(f'{thread_config.thread_name}: \t{command}')
                os.system(command)

                # export the modified image to files
                max_wait = 20
                while (max_wait >= 0 and not os.path.exists(thread_config.z3_normalized_output_file)):
                    time.sleep(0.5)
                    max_wait = max_wait - 1

                # get the modified image
                modified_image = get_new_image(solution_path=thread_config.z3_normalized_output_file).reshape(
                    -1)  # flatten
                assert (len(modified_image.shape) == 1)

                # get the original image
                original_image = pd.read_csv(thread_config.seed_file, header=None).to_numpy().reshape(-1)  # flatten
                assert (len(original_image.shape) == 1)

                if len(modified_image) > 0 and len(original_image) > 0 and len(modified_image) == len(original_image):

                    # export the original image to files
                    if len(original_image) > 0:
                        csv_old_image_path = thread_config.old_csv_image_file_path
                        with open(csv_old_image_path, mode='w') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(original_image)
                        png_old_image_path = thread_config.old_image_file_path
                        matplotlib.image.imsave(png_old_image_path,
                                                original_image.reshape(model_object.get_image_shape()))

                    # export the modified image to file
                    if len(modified_image) > 0:
                        csv_new_image_path = thread_config.new_csv_image_file_path
                        with open(csv_new_image_path, mode='w') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(modified_image)
                        png_new_image_path = thread_config.new_image_file_path
                        matplotlib.image.imsave(png_new_image_path,
                                                modified_image.reshape(model_object.get_image_shape()))

                    # check the valid property of the modified image
                    is_valid, original_prediction, modified_prediction = is_valid_modified_image(
                        model_object=model_object, threadconfig=thread_config,
                        csv_original_image_path=csv_old_image_path,
                        csv_new_image_path=csv_new_image_path)
                    is_valid = True # for testing
                    if is_valid:
                        logger.debug(f"Good news! The modified image is valid")

                        lock.acquire()
                        with open(thread_config.selected_seed_index_file_path, mode='a') as f:
                            logger.debug(f"Appending seed {seed_index} to the selected index file")
                            f.write(str(seed_index) + ',')
                        lock.release()

                        # create figure comparison
                        if thread_config.should_plot:
                            png_comparison_image_path = thread_config.comparison_file_path
                            logger.debug(f"Creating figure comparison")
                            create_figure_comparison(model_object, original_image, modified_image, original_prediction,
                                                     modified_prediction,
                                                     png_comparison_image_path)

                        # compute distance
                        l0_distance = compute_L0_distance(original_image, modified_image)
                        l1_distance = compute_L1_distance(original_image, modified_image)
                        l2_distance = compute_L2_distance(original_image, modified_image)
                        linf_distance = compute_inf_distance(original_image, modified_image)

                        diff_pixels = compute_the_different_pixels(original_image, modified_image)
                        logger.debug(f'diff_pixels = {diff_pixels}')

                        # add to comparison file
                        with open(thread_config.true_label_seed_file, 'r') as f:
                            true_label = int(f.read())

                        lock.acquire()
                        with open(thread_config.full_comparison, mode='a') as f:
                            writer = csv.writer(f)

                            if thread_config.attacked_neuron['enabled'] == 'True':
                                writer.writerow([seed_index, thread_config.attacked_neuron['layer_index'],
                                                 thread_config.attacked_neuron['neuron_index'],
                                                 thread_config.delta_lower_bound, thread_config.delta_upper_bound,
                                                 true_label, original_prediction, modified_prediction, is_valid,
                                                 diff_pixels, l0_distance, l1_distance, l2_distance, linf_distance])
                            else:
                                writer.writerow(
                                    [seed_index, thread_config.delta_lower_bound, thread_config.delta_upper_bound,
                                     true_label, original_prediction, modified_prediction, is_valid,
                                     diff_pixels, l0_distance, l1_distance, l2_distance, linf_distance])
                        lock.release()
                    else:
                        # the comparison between prediction is invalid
                        if os.path.exists(csv_old_image_path):
                            os.remove(csv_old_image_path)
                        if os.path.exists(csv_new_image_path):
                            os.remove(csv_new_image_path)
                        if os.path.exists(png_new_image_path):
                            os.remove(png_new_image_path)
                        if os.path.exists(png_old_image_path):
                            os.remove(png_old_image_path)
                else:
                    logger.debug(f'The constraints have no solution')
                logger.debug('--------------------------------------------------')
            # break

    def initialize_dnn_model_from_configuration_file(self, model_object):
        jsonController = ConfigController()
        dataset = jsonController.get_config(["dataset"])
        model_object.set_num_classes(jsonController.get_config([dataset, "num_classes"]))
        model_object.read_data(trainset_path=jsonController.get_config([dataset, "train_set"]),
                               testset_path=jsonController.get_config([dataset, "test_set"]))
        model_object.load_model(weight_path=jsonController.get_config([dataset, "weight"]),
                                structure_path=jsonController.get_config([dataset, "structure"]),
                                trainset_path=jsonController.get_config([dataset, "train_set"]))
        model_object.set_name_dataset(dataset)
        model_object.set_image_shape((28, 28))
        model_object.set_selected_seed_index_file_path(
            jsonController.get_config(["files", "selected_seed_index_file_path"]))
        return model_object


if __name__ == '__main__':
    logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
    logging.root.setLevel(logging.DEBUG)

    deepconcolic = SMT_DNN()
    model_object = deepconcolic.initialize_dnn_model_from_configuration_file(FASHION_MNIST())
    # seeds = deepconcolic.read_seeds_from_file(csv_file='/home/pass-la-1/PycharmProjects/mydeepconcolic/result/fashion_mnist/seeds.txt')
    seeds = deepconcolic.read_seeds_from_config(model_object)
    deepconcolic.generate_samples(model_object=model_object, seeds=seeds,
                                  n_threads=ConfigController().get_config(["n_threads"]), just_find_one_seed=False)
