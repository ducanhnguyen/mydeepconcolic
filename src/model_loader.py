import os

from src.config_parser import get_config
from src.mnist_dataset import mnist_dataset
from src.saved_models.mnist_ann_keras import MNIST_ANN_KERAS
from src.saved_models.mnist_deepcheck import MNIST_DEEPCHECK
from src.saved_models.mnist_simard import MNIST_SIMARD
from src.saved_models.mnist_simple import MNIST_SIMPLE


def initialize_dnn_model_from_name(name_model):
    global NORMALIZATION_FACTOR
    # custom code

    print('Model ' + name_model)
    model_object = None

    if name_model == "mnist_ann_keras":
        NORMALIZATION_FACTOR = mnist_dataset.NORMALIZATION_FACTOR
        model_object = MNIST_ANN_KERAS()

    elif name_model == "mnist_simard":
        model_object = MNIST_SIMARD()
        NORMALIZATION_FACTOR = mnist_dataset.NORMALIZATION_FACTOR

    elif name_model == "mnist_simple":
        model_object = MNIST_SIMPLE()
        NORMALIZATION_FACTOR = mnist_dataset.NORMALIZATION_FACTOR

    elif name_model == "mnist_deepcheck":
        model_object = MNIST_DEEPCHECK()
        NORMALIZATION_FACTOR = mnist_dataset.NORMALIZATION_FACTOR

    if model_object is None:
        return

    model_object.set_num_classes(get_config([name_model, "num_classes"]))
    model_object.read_data(trainset_path=get_config([name_model, "train_set"]),
                           testset_path=get_config([name_model, "test_set"]))
    model_object.load_model(weight_path=get_config([name_model, "weight"]),
                            structure_path=get_config([name_model, "structure"]),
                            trainset_path=get_config([name_model, "train_set"]))
    model_object.set_name_dataset(name_model)
    model_object.set_image_shape((28, 28))
    model_object.set_selected_seed_index_file_path(get_config(["files", "selected_seed_index_file_path"]))
    if not os.path.exists(get_config(["output_folder"])):
        os.makedirs(get_config(["output_folder"]))
    return model_object


def initialize_dnn_model():
    global NORMALIZATION_FACTOR
    # custom code
    name_model = get_config(attributes=["dataset"], recursive=True)
    return initialize_dnn_model_from_name(name_model)
