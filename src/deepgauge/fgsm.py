'''
Paper: Explaining and harnessing adversarial examples, https://arxiv.org/abs/1412.6572, Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy

'''
from src.saved_models.mnist_ann_keras import MNIST
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential
import keras.backend as K
import tensorflow as tf
import numpy as np
from src.saved_models.mnist_ann_keras import MNIST
import logging
import matplotlib.pyplot as plt
from keras.models import Model
import matplotlib

global logger

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


class FGSM:
    def __init__(self):
        pass

    def make_comparison_image(self, model_object, adversarial_image, original_image, path, original_prediction, modified_prediction, e):
        assert (isinstance(adversarial_image, np.ndarray))
        assert (len(adversarial_image.shape) == 2)  # is a 2-D array

        fig = plt.figure()
        nrow = 1
        ncol = 2

        # add the original image to the plot
        original_image = original_image.reshape(28, 28)
        fig1 = fig.add_subplot(nrow, ncol, 1)
        fig1.title.set_text(f'The original image\n(prediction = {original_prediction})')
        plt.imshow(original_image, cmap="gray")

        # add the modified image to the plot
        modified_image = adversarial_image.reshape(28, 28)
        fig2 = fig.add_subplot(nrow, ncol, 2)
        fig2.title.set_text(f'The adversarial image\n(prediction = {modified_prediction}), e = {e}')
        plt.imshow(modified_image, cmap="gray")

        # save to disk
        plt.savefig(path, pad_inches=0, bbox_inches='tight')
        logger.debug('Saved image')

    def get_uninitialized_variables(self, sess):
        global_vars = tf.global_variables()
        intialization = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, intialization) if not f]
        initialized_vars = [v for (v, f) in zip(global_vars, intialization) if f]
        return not_initialized_vars, initialized_vars

    def generate_loss_gradients(self, tf_target, model):
        assert (isinstance(model, Sequential))
        # the number of elements in target and in the output of model must be equal
        assert (tf_target.shape[0] == model.output.shape[1])

        # Gradient of loss wrt the input of the model (Tensor)
        tf_loss = K.sum(K.categorical_crossentropy(target=tf_target, output=model.output))
        tf_loss_gradients = K.gradients(loss=tf_loss, variables=[model.input, tf_target])
        return tf_loss_gradients

    def get_weights(self):
        weight_dict = dict()
        sess = tf.Session()
        is_not_initialized, initialized_vars = self.get_uninitialized_variables(sess)

        for not_initialized in is_not_initialized:
            isinstance(not_initialized, tf.Variable)
            # get name
            name = not_initialized.name  # example: 'dense_1/kernel:0'
            layer_name = str(name).split("/")[0]

            # get kernel
            kernel = model.get_layer(name=layer_name).get_weights()[0]
            weight_dict[layer_name + "/kernel:0"] = kernel

            # get bias
            bias = model.get_layer(name=layer_name).get_weights()[1]
            weight_dict[layer_name + "/bias:0"] = bias
        return weight_dict

    def generate_sign_matrix(self, tf_target, model, x_train, tf_loss_gradients, target, weight_dict):
        '''

        :param target:
        :param model:
        :param x_train: [0..1]
        :return:
        '''
        '''
        get the sign of loss gradients wrt the input of the model
        '''
        with tf.Session() as sess:
            # load weights
            is_not_initialized, initialized_vars = self.get_uninitialized_variables(sess)
            for not_initialized in is_not_initialized:
                isinstance(not_initialized, tf.Variable)
                if not_initialized.name in weight_dict:
                    trained = weight_dict[not_initialized.name]
                    assign_op = tf.assign(ref=not_initialized, value=trained)
                    sess.run(assign_op)

            # compute the gradient
            grad = sess.run(tf_loss_gradients, feed_dict={model.input: x_train, tf_target: target})[0]

            # logger.info(grad)
            sign = np.sign(grad)

            sess.close()
        return sign

    def compute(self, e, sign_matrix, x_train):
        '''

        :param e: [0..1]
        :param sign_matrix: contain -1 or 1
        :param x_train: int
        :return: [0..1], int
        '''
        adversarial_image = x_train + sign_matrix * e
        assert (isinstance(adversarial_image, np.ndarray))
        adversarial_image = adversarial_image.reshape(-1)

        # normalize
        normalized_adversarial_image = np.zeros(shape=(len(adversarial_image)))
        for index in range(len(adversarial_image)):
            if adversarial_image[index] > 1:
                normalized_adversarial_image[index] = 1
            elif adversarial_image[index] < 0:
                normalized_adversarial_image[index] = 0
            else:
                normalized_adversarial_image[index] = adversarial_image[index]

        normalized_adversarial_image = normalized_adversarial_image.reshape(1, -1)

        return normalized_adversarial_image

    def predict(self, model, adversarial_image):
        '''

        :param model:
        :param adversarial_image: [0..1]
        :return:
        '''
        assert (isinstance(model, Sequential))
        assert (isinstance(adversarial_image, np.ndarray))
        adversarial_image = adversarial_image.reshape(1, -1)

        with tf.get_default_graph().as_default():
            # must use when using thread
            prediction = Model(inputs=model.inputs, outputs=model.layers[-1].output).predict(x=adversarial_image)
            y_hat = np.argmax(prediction.reshape(-1))

        return y_hat


if __name__ == '__main__':
    logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
    logging.root.setLevel(logging.INFO)

    # create model
    model_object = MNIST()
    model_object.set_num_classes(10)
    model = model_object.load_model(
        weight_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/mnist_ann_keras_f1_original.h5',
        structure_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/src/saved_models/mnist_ann_keras_f1_original.json',
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/digit-recognizer/train.csv')
    model_object.read_data(
        trainset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/digit-recognizer/train.csv',
        testset_path='/home/pass-la-1/PycharmProjects/mydeepconcolic/dataset/digit-recognizer/test.csv')
    model = model_object.get_model()
    assert (isinstance(model, Sequential))
    logger.debug(model.summary())

    # create FGSM
    fgsm = FGSM()
    tf_target = tf.placeholder(shape=(model_object.get_num_classes(),), dtype=tf.float32)
    tf_loss_gradients = fgsm.generate_loss_gradients(tf_target, model)
    logger.debug(f"tf_loss_gradients = {tf_loss_gradients}")

    selected_seeds = []
    n_observations = model_object.get_Xtest().shape[0]
    weight_dict = fgsm.get_weights()
    for seed_index in range(4810, n_observations):
        logger.info("")
        logger.debug(f"number of default graph nodes = {len(tf.get_default_graph().as_graph_def().node)}")
        logger.info(f"seed_index = {seed_index}")

        # get a seed
        x_test, y_true_ground = model_object.get_an_observation_from_test_set(index=seed_index)
        x_test = x_test.reshape(1, -1)  # 0..1
        target = np.zeros(shape=(model_object.get_num_classes()))
        target[y_true_ground] = 1
        y_hat = fgsm.predict(model, x_test)

        # if the model predicts correctly the current seed
        if y_hat == y_true_ground:

            # generate adversarial samples
            sign = fgsm.generate_sign_matrix(model=model, x_train=x_test, target=target,
                                             tf_loss_gradients=tf_loss_gradients, tf_target=tf_target,
                                             weight_dict=weight_dict)
            logger.info(f"sum = {np.sum(sign)}")  # int
            for e in np.arange(start=0, stop=1 / 255 * 10, step=1 / 255):

                adversarial_image = fgsm.compute(e=e, sign_matrix=sign, x_train=x_test)

                # predict again with the adversarial sample
                y_adversarial_hat = fgsm.predict(model, adversarial_image)

                if y_true_ground != y_adversarial_hat:
                    path = f"/home/pass-la-1/PycharmProjects/mydeepconcolic/result/mnist/{seed_index}_{e}_comparison.png"
                    fgsm.make_comparison_image(
                        model_object = model_object,
                        adversarial_image=adversarial_image,
                        original_image=x_test,
                        path=path,
                        original_prediction=y_hat,
                        modified_prediction=y_adversarial_hat,
                        e=e)

                    # save the adversarial image to file
                    import csv

                    adversarial_image_csv_path = f"/home/pass-la-1/PycharmProjects/mydeepconcolic/result/mnist/{seed_index}_{e}_new.csv"
                    with open(adversarial_image_csv_path, mode='w') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(adversarial_image.reshape(-1))

                    #
                    logger.info(f"e = {e}")
                    logger.info(f"Ground true= {y_true_ground}")
                    logger.info(f"Prediction label of original sample = {y_hat}")
                    logger.info(f"Prediction label of adversarial sample = {y_adversarial_hat}")
                    selected_seeds.append(f"seex_index = {seed_index}, e = {e}")

                    # only choose the least e
                    break

        logger.debug(f"selected_seeds = {selected_seeds}")
