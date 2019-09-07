'''
Command:
/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/lib/z3-4.8.5-x64-osx-10.14.2/bin/z3 -smt2 /Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/result/constraint.txt > /Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/result/solution.txt
'''
# TODO: them bias vao cong thuc weight
import os
import random
import random as ran

from keras.models import Model

from src.utils import keras_activation, keras_layer, keras_model
from src.test_summarizer import *

MINUS_INF = -10000000
INF = 10000000

DELTA_PREFIX_NAME = 'delta'

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# TODO: Evaluate the value of neuron with samples belonging to a class

def create_constraint_between_layers(model):
    constraints = []
    smt_constraints = []

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

                        weight = weight / 255 # rather than normalizing feature input
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


def create_activation_constraints(model):
    constraints = []
    smt_constraints = []

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


def create_variable_declarations(model, type_feature='int', delta=DELTA_PREFIX_NAME):
    constraints = []
    constraints.append(f'; variable declaration')
    # types.append(f'(declare-fun {delta} () Int)')

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

                    #constraint = f'(declare-fun feature_{feature_idx}_tmp () Real)'
                    #constraints.append(constraint)

                    #constraints.append(f'(assert(= feature_{feature_idx}_tmp (/ feature_{feature_idx} 255)))')
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


def create_feature_constraints_from_an_observation(model, x_train, delta=DELTA_PREFIX_NAME):
    constraints = []
    smt_constraints = []

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


def create_output_constraints_from_an_observation(model,
                                                  x_train, y_train):
    constraints = []
    smt_constraints = []

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

            '''
            constraint type 2.
            '''
            '''
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
            '''

            '''
            constraint type 3. Add a constraint related to the bound of output. 
            '''
            tmp = x_train.reshape(1, -1)
            before_softmax = model.layers[-2]
            intermediate_layer_model = Model(inputs=model.inputs,
                                             outputs=before_softmax.output)
            prediction = intermediate_layer_model.predict(tmp)
            logger.debug(f'The prediction of the current seed (before softmax): {prediction}')

            smt_constraints.append(f'; output constraints')
            true_label = open(TRUE_LABEL_SEED_FILE, "r").readline()
            old_probability = prediction[0][int(true_label)]
            smt_constraint = f'(assert (< {left} {old_probability}) )'
            logger.debug(f'Output constraint = {smt_constraint}')
            smt_constraints.append(smt_constraint)

    elif keras_model.is_CNN(model):
        logger.debug(f'Does not support CNN')

    else:
        logger.debug(f'Unable to detect the type of neural network')

    return constraints, smt_constraints


def create_bound_of_feature_constraints(model,
                                        feature_lower_bound, feature_upper_bound,
                                        delta_lower_bound, delta_upper_bound, delta_prefix=DELTA_PREFIX_NAME):
    assert (feature_lower_bound <= feature_upper_bound)
    assert (delta_lower_bound <= delta_upper_bound)
    assert (delta_prefix != None and delta_prefix != '')
    smt_constraints = []

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


def create_constraints_file(obj, seed_index):
    assert (isinstance(obj, abstract_dataset))
    assert (seed_index >= 0)

    # get an observation
    x_train, y_train = obj.get_an_observation(seed_index)
    with open(SEED_FILE, mode='w') as f:
        seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        seed.writerow(x_train[0])
    with open(TRUE_LABEL_SEED_FILE, 'w') as f:
        f.write(str(y_train))

    # generate constraints
    smt_exp = define_mathematical_function()

    variable_types = create_variable_declarations(obj.get_model())

    layers_constraints, smt_layers_constraints = create_constraint_between_layers(obj.get_model())

    activation_constraints, smt_activation_constraints = create_activation_constraints(obj.get_model())

    input_constraints, smt_input_constraints = create_feature_constraints_from_an_observation(obj.get_model(), x_train)

    output_constraints, smt_output_constraints = create_output_constraints_from_an_observation(obj.get_model(), x_train,
                                                                                               y_train)
    smt_bound_input_constraints = create_bound_of_feature_constraints(obj.get_model(), delta_prefix=DELTA_PREFIX_NAME,
                                                                      feature_lower_bound=0,
                                                                      feature_upper_bound=255,
                                                                      delta_lower_bound=0, delta_upper_bound=10)

    # create constraint file
    with open(CONSTRAINTS_FILE, 'w') as f:
        f.write('(set-option :timeout 10000)\n')
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


if __name__ == '__main__':

    SEED_FILE = f'../data/seed.csv'
    TRUE_LABEL_SEED_FILE = f'../data/true_label.txt'
    CONSTRAINTS_FILE = f'../data/constraint.txt'
    SEED_INDEX_FILE = f'../data/seed_index.txt'
    seed_start = 0  # mnist: 728->60000, shvn: stop at 109
    seed_end = 73257

    Z3_PATH = '/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/lib/z3-4.8.5-x64-osx-10.14.2/bin/z3'
    Z3_INPUT_PATH = '/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/data/constraint.txt'
    Z3_OUTPUT_PATH = '/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/data/solution.txt'
    Z3_NORMALIED_OUTPUT_PATH = "../data/norm_solution.txt"

    # initialize the environment
    '''
    if os.path.exists('/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/data'):
        shutil.rmtree('/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/data')
    os.makedirs('/Users/ducanhnguyen/Documents/python/pycharm/mydeepconcolic/data')
    '''

    # construct model from file
    from saved_models.shvn_ann_keras import *

    obj = SHVN()
    obj.set_num_classes(10)
    obj.read_data(training_path='../SVHN/train_32x32.mat', testing_path='../SVHN/test_32x32.mat')
    obj.load_model(kernel_path='../saved_models/shvn_ann_keras.h5', model_path='../saved_models/shvn_ann_keras.json',
                   training_path='../SVHN/train_32x32.mat')
    image_shape = (32, 32, 3)
    logger.debug(obj.get_model().summary())

    # generate adversarial samples
    seeds = np.arange(seed_start, seed_end)
    random.shuffle(seeds)  # 9175 shvn

    for seed_index in seeds:
        # seed_index=9175
        if os.path.exists(TRUE_LABEL_SEED_FILE):
            os.remove(TRUE_LABEL_SEED_FILE)
        if os.path.exists(SEED_FILE):
            os.remove(SEED_FILE)
        if os.path.exists(CONSTRAINTS_FILE):
            os.remove(CONSTRAINTS_FILE)
        if os.path.exists(Z3_OUTPUT_PATH):
            os.remove(Z3_OUTPUT_PATH)
        if os.path.exists(Z3_NORMALIED_OUTPUT_PATH):
            os.remove(Z3_NORMALIED_OUTPUT_PATH)

        logger.debug(f'seed index = {seed_index}')
        with open(SEED_INDEX_FILE, mode='w') as f:
            f.write(str(seed_index))

        # generate constraints
        logger.debug(f'generate constraints')
        create_constraints_file(obj, seed_index)

        # call SMT-Solver
        logger.debug(f'call SMT-Solver to solve the constraints')
        command = f"{Z3_PATH} -smt2 {Z3_INPUT_PATH} > {Z3_OUTPUT_PATH}"
        logger.debug(f'\tcommand = {command}')
        os.system(command)

        # parse solver solution
        logger.debug(f'parse solver solution')
        command = '/Library/Java/JavaVirtualMachines/jdk-11.0.2.jdk/Contents/Home/bin/java -Dfile.encoding=UTF-8 -p /Users/ducanhnguyen/eclipse-workspace/z3_parser2/bin:/Users/ducanhnguyen/eclipse-workspace/z3_parser2/lib/hamcrest-core-1.3.jar:/Users/ducanhnguyen/eclipse-workspace/z3_parser2/lib/jeval.jar:/Users/ducanhnguyen/eclipse-workspace/z3_parser2/lib/junit-4.13-beta-3.jar:/Users/ducanhnguyen/eclipse-workspace/z3_parser2/lib/org.apache.commons.io.jar -m z3_parser2/z3_parser.Z3SolutionParser'
        logger.debug(f'\t{command}')
        os.system(command)

        # comparison
        img = get_new_image(solution_path=Z3_NORMALIED_OUTPUT_PATH)

        if len(img) > 0:  # 16627, 1121
            csv_new_image_path = f'../data/{seed_index}.csv'
            with open(csv_new_image_path, mode='w') as f:
                seed = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                seed.writerow(img)

            # plot the seed and the new image
            png_new_image_path = f'../data/{seed_index}.png'
            seed, new_image, similar = plot_seed_and_new_image(model=obj.get_model(), seed_path=SEED_FILE,
                                                               new_image_path=csv_new_image_path,
                                                               png_new_image_path=png_new_image_path,
                                                               image_shape=image_shape)
            os.remove(csv_new_image_path)
            # if similar:
            #    os.remove(png_new_image_path)
        else:
            logger.debug(f'The constraints have no solution')
        logger.debug('--------------------------------------------------')
        #break
