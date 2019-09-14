'''
Command:
/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/lib/z3-4.8.5-x64-osx-10.14.2/bin/z3 -smt2 /Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/constraint.txt > /Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/dataset/solution.txt
'''
import os
import random as ran
from threading import Thread
from types import SimpleNamespace

import tensorflow as tf
from keras.models import Model

from src.config_parser import *
from src.saved_models.mnist_ann_keras import *
from src.test_summarizer import *
from src.utils import keras_activation, keras_layer, keras_model

MINUS_INF = -10000000
INF = 10000000

DELTA_PREFIX_NAME = get_config(["constraint_config", "delta_prefix_name"])

global logger
logger = logging.getLogger()

global graph


def create_constraint_between_layers(model_object):
    assert (isinstance(model_object, abstract_dataset))
    constraints = []
    smt_constraints = []

    model = model_object.get_model()
    for current_layer_idx, current_layer in enumerate(model.layers):

        if current_layer_idx == 0:
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
                    type constraint 1
                    '''
                    constraint = f'{var} = '
                    for feature_idx in range(n_features):
                        previous_var = f'feature_{feature_idx}'
                        weight = kernel[feature_idx][current_pos]
                        weight = weight / 255
                        constraint += f'{previous_var} * {weight:.25f} + '

                    constraint += f'{biases[current_pos]:.25f}'
                    constraints.append(constraint)

                    '''
                    type constraint 2
                    '''
                    smt_constraint = ''
                    for feature_idx in range(n_features):
                        previous_var = f'feature_{feature_idx}'
                        weight = kernel[feature_idx][current_pos]

                        weight = weight / 255  # rather than normalizing feature input
                        if feature_idx == 0:
                            smt_constraint = f'(* {previous_var} {weight:.25f}) '
                        else:
                            smt_constraint = f'(+ {smt_constraint} (* {previous_var} {weight:.25f})) '

                    smt_constraint = f'(+ {smt_constraint} {biases[current_pos]:.25f}) '
                    smt_constraint = f'(assert(= {var} {smt_constraint}))'
                    smt_constraints.append(smt_constraint)

            else:
                logger.debug(f'Do not support CNN')
                continue

        elif keras_layer.is_2dconv(current_layer):
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


def create_activation_constraints(model_object):
    assert (isinstance(model_object, abstract_dataset))
    constraints = []
    smt_constraints = []

    model = model_object.get_model()
    if keras_model.is_ANN(model):
        for current_layer_idx, current_layer in enumerate(model.layers):
            units = keras_layer.get_number_of_units(model, current_layer_idx)

            if keras_activation.is_activation(current_layer):
                constraints.append(f'; {current_layer.name}')
                smt_constraints.append(f'; {current_layer.name}')

                # Create the formula based on the type of activation
                if keras_activation.is_relu(current_layer):
                    for unit_idx in range(units):
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
                            if unit_idx == units - 1:
                                constraint += f'exp(-u_{current_layer_idx}_{unit_idx})'
                            else:
                                constraint += f'exp(-u_{current_layer_idx}_{unit_idx}) + '
                        constraints.append(constraint)

                        for unit_idx in range(units):
                            constraint = f'v_{current_layer_idx}_{unit_idx} = exp(-u_{current_layer_idx}_{unit_idx}) / {sum}'
                            constraints.append(constraint)
                        '''
                        constraint type 2
                        '''
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

            elif (keras_layer.is_dense(current_layer) and not keras_layer.is_activation(current_layer)) or \
                    keras_layer.is_dropout(current_layer):
                # the current layer is dense and not an activation
                constraints.append(f'; {current_layer.name}')
                smt_constraints.append(f'; {current_layer.name}')

                for unit_idx in range(units):
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


def create_variable_declarations(model_object, type_feature=get_config(["constraint_config", "feature_input_type"])):
    assert (isinstance(model_object, abstract_dataset))
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


def create_feature_constraints_from_an_observation(model_object, x_train, delta=DELTA_PREFIX_NAME):
    assert (isinstance(model_object, abstract_dataset))
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


def create_output_constraints_from_an_observation(model_object, x_train, y_train, thread_config):
    assert (isinstance(model_object, abstract_dataset))
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

            smt_type_constraint = get_config(["constraint_config", "output_layer_type_constraint"])

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

                with thread_config.graph.as_default():
                    # must use when using thread
                    prediction = intermediate_layer_model.predict(tmp)

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


def create_bound_of_feature_constraints(model_object,
                                        feature_lower_bound, feature_upper_bound,
                                        delta_lower_bound, delta_upper_bound, delta_prefix=DELTA_PREFIX_NAME):
    assert (feature_lower_bound <= feature_upper_bound)
    assert (delta_lower_bound <= delta_upper_bound)
    assert (delta_prefix != None and delta_prefix != '')
    assert (isinstance(model_object, abstract_dataset))
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

    layers_constraints, smt_layers_constraints = create_constraint_between_layers(model_object)

    activation_constraints, smt_activation_constraints = create_activation_constraints(model_object)

    input_constraints, smt_input_constraints = create_feature_constraints_from_an_observation(model_object, x_train)

    output_constraints, smt_output_constraints = create_output_constraints_from_an_observation(model_object, x_train,
                                                                                               y_train, thread_config)
    smt_bound_input_constraints = create_bound_of_feature_constraints(model_object=model_object,
                                                                      delta_prefix=DELTA_PREFIX_NAME,
                                                                      feature_lower_bound=get_config(
                                                                          ["constraint_config", "feature_lower_bound"]),
                                                                      feature_upper_bound=get_config(
                                                                          ["constraint_config", "feature_upper_bound"]),
                                                                      delta_lower_bound=get_config(
                                                                          ["constraint_config", "delta_lower_bound"]),
                                                                      delta_upper_bound=get_config(
                                                                          ["constraint_config", "delta_upper_bound"]))

    # create constraint file
    with open(thread_config.constraints_file, 'w') as f:
        f.write(f'(set-option :timeout {get_config(["z3", "time_out"])})\n')
        f.write(f'(using-params smt :random-seed {ran.randint(1, 101)})\n')

        for constraint in smt_exp:
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
            seed_index = int(seed_index)
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
            logger.debug(f'{thread_config.thread_name}: call SMT-Solver to solve the constraints')
            command = f"{thread_config.z3_path} -smt2 {thread_config.constraints_file} > {thread_config.z3_solution_file}"
            # logger.debug(f'\t{thread_config.thread_name}: command = {command}')
            os.system(command)

            # parse solver solution
            logger.debug(f'{thread_config.thread_name}: parse solver solution')
            tmp1 = thread_config.z3_solution_file
            tmp2 = thread_config.z3_normalized_output_file
            command = get_config(["z3", "z3_solution_parser_command"]) + f' {tmp1} ' + f'{tmp2}'

            logger.debug(f'{thread_config.thread_name}: \t{command}')
            os.system(command)

            # export the modified image to files
            modified_image = get_new_image(solution_path=thread_config.z3_normalized_output_file)  # 1-D image
            modified_image /= 255  # if the input model is in range of [0..1] and the value of pixel in image is in [0..255], we need to scale the image
            if len(modified_image) > 0:
                csv_new_image_path = get_config(['files', 'new_csv_image_file_path'])\
                    .replace('{seed_index}', str(seed_index))
                with open(csv_new_image_path, mode='w') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(modified_image)
            png_new_image_path = get_config(['files', 'new_image_file_path']).replace('{seed_index}', str(seed_index))
            matplotlib.image.imsave(png_new_image_path, modified_image.reshape(model_object.get_image_shape()))

            # export the original image to files
            original_image = pd.read_csv(thread_config.seed_file, header=None).to_numpy()
            if len(original_image) > 0:
                csv_old_image_path = get_config(['files', 'old_csv_image_file_path']).replace('{seed_index}',
                                                                                              str(seed_index))
                with open(csv_old_image_path, mode='w') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(original_image)
            png_old_image_path = get_config(['files', 'old_image_file_path']).replace('{seed_index}', str(seed_index))
            matplotlib.image.imsave(png_old_image_path, original_image.reshape(model_object.get_image_shape()))

            # check the valid property of the modified image
            if len(modified_image) > 0 and len(original_image) > 0:
                png_comparison_image_path = get_config(['files', 'comparison_file_path']).replace('{seed_index}',
                                                                                                  str(seed_index))
                is_valid, original_prediction, modified_prediction = is_valid_modified_image(
                    model_object=model_object, config=thread_config,
                    csv_new_image_path=csv_new_image_path)
                if is_valid:
                    with open(thread_config.selected_seed_index_file_path, mode='a') as f:
                        f.write(str(seed_index) + ',')

                # create figure comparison
                if thread_config.should_plot:
                    create_figure_comparison(model_object, seed, original_prediction, modified_prediction,
                                             png_comparison_image_path,
                                             png_new_image_path)

            else:
                logger.debug(f'The constraints have no solution')
            logger.debug('--------------------------------------------------')
            # break


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
    config.graph = tf.get_default_graph()
    config.analyzed_seed_index_file_path = get_config(["files", "analyzed_seed_index_file_path"])
    config.selected_seed_index_file_path = get_config(["files", "selected_seed_index_file_path"])
    config.thread_name = f'thread_{thread_idx}'
    config.should_plot = False  # should be False when running in multithread
    config.z3_solution_parser_command = get_config(["z3", "z3_solution_parser_command"])
    config.new_image_file_path = get_config(["files", "new_image_file_path"])
    config.comparison_file_path = get_config(["files", "comparison_file_path"])
    return config


def read_seeds_from_config():
    start_seed = get_config([model_object.get_name_dataset(), "start_seed"])
    end_seed = get_config([model_object.get_name_dataset(), "end_seed"])
    seeds = np.arange(start_seed, end_seed)
    return seeds


def read_seeds_from_file(csv_file):
    selected_seed_indexes = pd.read_csv(csv_file, header=None).to_numpy()
    selected_seed_indexes = selected_seed_indexes.reshape(-1)
    return selected_seed_indexes


def generate_samples(model_object, seeds, n_threads):
    """
    Generate adversarial samples
    :return:
    """
    '''
    generate adversarial samples
    '''
    if n_threads >= 2:
        # prone to error
        n_single_thread_seeds = int(np.floor((len(seeds)) / n_threads))
        logger.debug(f'n_single_thread_seeds = {n_single_thread_seeds}')
        threads = []

        for thread_idx in range(n_threads):
            # get range of seed in the current thread
            if thread_idx == n_threads - 1:
                thread_seeds = np.arange(n_single_thread_seeds * (n_threads - 1), len(seeds))
            else:
                thread_seeds = np.arange(n_single_thread_seeds * thread_idx, n_single_thread_seeds * (thread_idx + 1))

            # read the configuration of a thread
            thread_config = set_up_config(thread_idx)
            thread_config.image_shape = model_object.get_image_shape()

            # create new thread
            t = Thread(target=image_generation, args=(thread_seeds, thread_config, model_object))
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
        # run on the main thread
        image_generation(seeds, main_thread_config, model_object)


def export_to_image(model_object):
    # load selected indexes
    selected_seed_indexes = pd.read_csv(get_config(["files", "selected_seed_index_file_path"]), header=None).to_numpy()
    selected_seed_indexes = selected_seed_indexes.reshape(-1)

    # config
    config = set_up_config(1235678910)  # can use any number
    config.should_plot = True

    for seed_index in selected_seed_indexes:
        seed_index = int(seed_index)
        logger.debug(f'Seed index = {seed_index}')
        with open(config.seed_index_file, mode='w') as f:
            f.write(str(seed_index))

        # generate constraints
        logger.debug(f'{config.thread_name}: generate constraints')
        create_constraints_file(model_object, seed_index, config)

        # ecall SMT-Solver
        logger.debug(f'{config.thread_name}: call SMT-Solver to solve the constraints')
        command = f"{config.z3_path} -smt2 {config.constraints_file} > {config.z3_solution_file}"
        logger.debug(f'\t{config.thread_name}: command = {command}')
        os.system(command)

        # parse the solution of constraints
        logger.debug(f'{config.thread_name}: parse solver solution')
        command = f'{config.z3_solution_parser_command} {config.z3_solution_file} {config.z3_normalized_output_file}'
        logger.debug(f'{config.thread_name}: {command}')
        os.system(command)

        # comparison
        img = get_new_image(solution_path=config.z3_normalized_output_file)

        if len(img) > 0:
            csv_new_image_path = f'../result/{model_object.get_name_dataset()}/{seed_index}.csv'
            with open(csv_new_image_path, mode='w') as f:
                seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                seed.writerow(img)

            # plot the seed and the new image
            png_comparison_image_path = str(config.comparison_file_path).replace('{seed_index}', str(seed_index))
            png_new_image_path = str(config.new_image_file_path).replace('{seed_index}', str(seed_index))
            is_valid_modified_image(model_object=model_object, config=config,
                                    csv_new_image_path=csv_new_image_path,
                                    png_comparison_image_path=png_comparison_image_path,
                                    png_new_image_path=png_new_image_path)


def initialize_dnn_model():
    # custom code
    model_object = MNIST()
    dataset = get_config(["dataset"])
    model_object.set_num_classes(get_config([dataset, "num_classes"]))
    model_object.read_data(trainset_path=get_config([dataset, "train_set"]),
                           testset_path=get_config([dataset, "test_set"]))
    model_object.load_model(weight_path=get_config([dataset, "weight"]),
                            structure_path=get_config([dataset, "structure"]),
                            trainset_path=get_config([dataset, "train_set"]))
    model_object.set_name_dataset(dataset)
    model_object.set_image_shape((28, 28))
    model_object.set_selected_seed_index_file_path(get_config(["files", "selected_seed_index_file_path"]))
    return model_object


if __name__ == '__main__':
    logging.basicConfig()
    logging.root.setLevel(logging.DEBUG)

    model_object = initialize_dnn_model()
    seeds = read_seeds_from_file(csv_file=get_config(["files", "selected_seed_index_file_path"]))
    generate_samples(model_object=model_object, seeds=seeds, n_threads=get_config(["n_threads"]))
    # export_to_image(model_object)
