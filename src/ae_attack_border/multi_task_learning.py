import keras

from src.ae_attack_border.data import wrongseeds_AlexNet
import tensorflow as tf

if __name__ == '__main__':
    ATTACKED_MODEL_H5 = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/model/Alexnet.h5"
    BASE_PATH = "/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/epsilon=0,5/Alexnet/autoencoder_Alexnetborder"
    WRONG_SEEDS = wrongseeds_AlexNet

    N_ATTACKING_SAMPLES = 2000
    N_CLASSES = 10

    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    X_train = X_train / 255

    dnn = keras.models.load_model(filepath=ATTACKED_MODEL_H5, compile=False)
