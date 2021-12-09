import csv
import numpy as np
import os
from src.model_loader import initialize_dnn_model
from src.utils import utilities
import numpy as np
import tensorflow as tf
from keras.models import Model
from src.utils.feature_ranker1d import feature_ranker1d

if __name__ == '__main__':
    model_object = initialize_dnn_model()
    print(model_object.get_model().summary())

    # classifier = model_object.get_model()
    input = model_object.get_Xtrain()[0].reshape(-1, 784)
    trueLabel = 5
    targetLabel = 3

    classifier = Model(inputs=model_object.get_model().inputs,
                                     outputs=model_object.get_model().layers[-2].output)

    gradientTrueLabel = feature_ranker1d.compute_gradient_wrt_features(tf.convert_to_tensor(input),
                                                                       trueLabel,
                                                                       classifier)
    print(gradientTrueLabel/255)

    # gradientTargetLabel = feature_ranker1d.compute_gradient_wrt_features(tf.convert_to_tensor(input),
    #                                                                      targetLabel,
    #                                                                      classifier)