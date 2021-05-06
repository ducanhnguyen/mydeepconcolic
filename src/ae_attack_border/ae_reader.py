import cv2
import keras
import numpy as np
import tensorflow as tf

from src.utils import utilities


def get_border(images: np.ndarray) -> np.ndarray:
    border_results = []
    for image in images:
        border_img = (image * 255).astype(np.uint8)
        border_img = np.array(cv2.Canny(border_img, 100, 200)).reshape((28, 28, 1))
        border_results.append(border_img)
    return np.array(border_results, dtype=np.float32) / 255.


def get_internal_images(images: np.ndarray, border_images=None) -> np.ndarray:
    internal_results = []
    if border_images is None:
        border_images = get_border(images)

    for border_image, image in zip(border_images, images):
        border_image_flat = border_image.flatten()
        image_flat = image.flatten()
        border_position = np.where(border_image_flat == 1.)
        internal_result = np.array(image_flat)
        internal_result[border_position] = 0
        internal_result = internal_result.reshape((28, 28, 1))

        internal_results.append(internal_result)

    return np.array(internal_results)


if __name__ == '__main__':
    ORI_LABEL = 6
    TARGET_LABEL = 4
    ATTACKED_MODEL_H5 = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/model/Lenet.h5"
    AE_MODEL_H5 = f"/Users/ducanhnguyen/Documents/mydeepconcolic/result/ae-attack-border/epsilon=0,5/Lenet/autoencoder_Lenetborder_{ORI_LABEL}_{TARGET_LABEL}.h5"
    wrongseeds_LeNet = [80, 132, 494, 500, 788, 1201, 1244, 1604, 2426, 2554, 2622, 2676, 3210, 4014, 4164, 4402, 4438,
                        4476, 4715, 5065, 5174, 5482, 5723, 5798, 5821, 5855, 6197, 6202, 6246, 6315, 6818, 6885, 7006,
                        7080, 7264, 7606, 7972, 7994, 8200, 8202, 8270, 8480, 8729, 8772, 8849, 9256, 9266]
    N_ATTACKING_SAMPLES = 2000

    print(f"autoencoder {AE_MODEL_H5}")
    print(f"attack {ORI_LABEL} -> {TARGET_LABEL}")

    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    X_train = X_train / 255

    dnn = keras.models.load_model(filepath=ATTACKED_MODEL_H5, compile=False)
    ae = keras.models.load_model(filepath=AE_MODEL_H5, compile=False)

    # get X_attack
    X_attack = []
    for idx in range(len(X_train)):
        if idx not in wrongseeds_LeNet and y_train[idx] == ORI_LABEL:
            X_attack.append(X_train[idx])
    X_attack = np.asarray(X_attack)
    X_attack = X_attack[:N_ATTACKING_SAMPLES]
    print(f'The shape of X_attack = {X_attack.shape}')

    # attack
    candidates = []
    border_origin_images = get_border(X_attack)

    showEdge = False
    if showEdge:
        utilities.show_two_images(left_title=f"ori (idx = {idx})",
                                  x_28_28_left=X_attack[0].reshape(28, 28),
                                  right_title="border_origin_images",
                                  x_28_28_right=border_origin_images[0].reshape(28, 28),
                                  display=True
                                  )

    internal_origin_images = get_internal_images(X_attack, border_images=border_origin_images)
    showInternal = False
    if showInternal:
        utilities.show_two_images(left_title=f"ori (idx = {idx})",
                                  x_28_28_left=X_attack[0].reshape(28, 28),
                                  right_title="internal_origin_images",
                                  x_28_28_right=internal_origin_images[0].reshape(28, 28),
                                  display=True
                                  )

    candidate_generated_borders = ae.predict(X_attack) * border_origin_images
    candidates = np.clip(candidate_generated_borders + internal_origin_images, 0, 1)

    # compute the number of adv
    candidates = np.asarray(candidates)
    print(f'The shape of candidate adv = {candidates.shape}')
    Y_pred = dnn.predict(candidates.reshape(-1, 28, 28, 1))
    y_pred = np.argmax(Y_pred, axis=1)
    print(f"candidate label = {y_pred}")
    print(f"#adv = {np.sum(y_pred == TARGET_LABEL)}")
