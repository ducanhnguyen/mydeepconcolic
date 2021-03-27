'''
Given a set of changed pixels, find the joint set of this set and n-most important pixels
'''
from src.edge_detection import is_edge
from src.model_loader import initialize_dnn_model
from src.utils.feature_ranker1d import feature_ranker1d
from src.utils.feature_ranker_2d import RANKING_ALGORITHM

if __name__ == '__main__':
    #
    model_object = initialize_dnn_model()

    #
    seed_idx = 7387
    x_28_28 = model_object.get_Xtrain()[seed_idx].reshape(28, 28)
    x_784 = model_object.get_Xtrain()[seed_idx].reshape(784)
    changed_features = []
    idx = 0
    for i in range(28):
        for j in range(28):
            if is_edge(i, j, x_28_28):
                changed_features.append(idx)
            idx += 1
    print(f'size of changed_features = {len(changed_features)}')
    print(f'changed_features = {changed_features}')

    #
    true_label = model_object.get_ytrain()[seed_idx]
    print(f'true label = {true_label}')
    target_labels = [idx for idx in range(10) if idx != true_label]
    x_28_28 = x_28_28.reshape(-1, 784)
    grad_arr = []
    for target_label in target_labels:
        important_features = feature_ranker1d.find_important_features_of_a_sample(
            input_image=x_28_28.reshape(-1, 784),
            n_important_features=len(changed_features),
            algorithm=RANKING_ALGORITHM.COI,
            gradient_label=target_label,
            classifier=model_object.get_model())
        # joint set
        joint = [idx for idx in changed_features if idx in important_features]
        grad_arr.append(len(joint))

    # sort and display
    grad_arr, target_labels = zip(*sorted(zip(grad_arr, target_labels), reverse=True))
    for k, v in zip(grad_arr, target_labels):
        if v != true_label:
            print(f'label {v}: len(joint pixels) = {k}')
