import keras

from src.ae_attack_border import AE_LOSSES

if __name__ == '__main__':
    ATTACKED_MODEL_H5 = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/model/Lenet.h5"
    dnn = keras.models.load_model(filepath=ATTACKED_MODEL_H5, compile=False)

