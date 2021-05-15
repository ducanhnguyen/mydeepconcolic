import os

import keras
import tensorflow as tf

from src.ae_attack_border.ae_reader import get_X_attack, generate_adv_for_single_attack_BORDER_PATTERN
from src.ae_attack_border.data import wrongseeds_AlexNet

if __name__ == '__main__':
    ATTACKED_MODEL_H5 = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/model/Alexnet.h5"

    WRONG_SEEDS = wrongseeds_AlexNet
    ORI = 2
    TARGET = 1
    N_ATTACKING_SAMPLES = 2000
    N_CLASSES = 10

    a = ["0,0", "0,05", "0,1", "0,15", "0,2", "0,25", "0,3", "0,35", "0,4", "0,45", "0,5",
         "0,55", "0,6", "0,65", "0,7", "0,75", "0,8", "0,85", "0,9", "0,95", "1,0"]
    # a = ["0,0", "0,05", "0,1", "0,15", "0,2", "0,25", "0,3", "0,35"]
    dnn = keras.models.load_model(filepath=ATTACKED_MODEL_H5, compile=False)

    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    X_train = X_train / 255
    X_attack, selected_seed = get_X_attack(X_train, y_train, WRONG_SEEDS, ORI, N_ATTACKING_SAMPLES)
    arr = []
    for item in a:
        AE_MODEL_H5 = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/multi-epsilon/Alexnet/{ORI}_{TARGET}_relu/autoencoder_Alexnetborder_{ORI}_{TARGET}_epsilon{item}.h5"
        if os.path.exists(AE_MODEL_H5):
            print(f"Analyzing {item}")
            ae = keras.models.load_model(filepath=AE_MODEL_H5, compile=False)
            n_adv, advs, oris = generate_adv_for_single_attack_BORDER_PATTERN(X_attack, TARGET, ae, dnn)
            print(f'epsilon {item}: n_adv = {n_adv}')
            arr.append(n_adv)
        else:
            print(f"{AE_MODEL_H5} does not exist!")
    print(arr)
