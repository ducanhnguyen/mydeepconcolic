import csv
import numpy as np
import os
from src.model_loader import initialize_dnn_model, initialize_dnn_model_from_name
from src.utils import utilities
import numpy as np

if __name__ == '__main__':
    TARGET_LABEL = -1  # -1 means untargeted attack
    BASEPATH = f"/Users/ducanhnguyen/Documents/NpixelAttackDeepFault/datatest/ImprovedDeepFault/fashionmnist_deepcheck/out_z3based"


    model_object = initialize_dnn_model_from_name("fashionmnist_deepcheck")
    print(model_object.get_model().summary())


    img = f"{BASEPATH}/img/"
    if (not os.path.exists(img)):
        os.makedirs(img)

    l0s = []
    l2s = []
    count = 0
    for index in range(0, 10000):
        ori = model_object.get_Xtrain()[index]

        adv_path = f"{BASEPATH}/{index}/{index}_adv.csv"
        if os.path.exists(adv_path):
            pred_ori = model_object.get_model().predict(ori.reshape(-1, 784));
            pred_ori_label = np.argmax(pred_ori)
            true_label = model_object.get_ytrain()[index]
            print("\n")
            print(adv_path)

            if pred_ori_label == true_label :
                # load adv
                adv = []
                with open(adv_path) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                    for row in csv_reader:
                        adv.append(np.asarray(row).astype(int))
                adv = np.asarray(adv)
                pred_adv = model_object.get_model().predict((adv / 255).reshape(-1, 784))
                for i in range(len(pred_adv)):
                    print(f"({i}) {pred_adv[i]}")
                adv_label = np.argmax(pred_adv)
                print('adv_label = ' + str(adv_label))
                satisfy = False
                if (TARGET_LABEL == -1 and adv_label != true_label):
                    satisfy = True
                elif (TARGET_LABEL != -1 and adv_label != true_label and adv_label == TARGET_LABEL):
                    satisfy = True
                if satisfy:
                    print("PASS")
                    ori = np.round(ori * 255)
                    l0 = utilities.compute_l0(ori, adv, normalized=True)
                    l2 = utilities.compute_l2(ori, adv)
                    l0s.append(l0)
                    l2s.append(l2)
                    utilities.show_two_images(ori.reshape(28, 28), adv.reshape(28, 28),
                                              left_title=f"label {true_label}",
                                              right_title=f"label {adv_label}, l0 = {l0}",
                                              display=None,
                                              path=f"{BASEPATH}/img/{index}_adv.png")
                else:
                    print("FAIL")
            else:
                print(f"Ignore: trueLabel = {true_label} ({pred_ori[0][true_label]}), pred_label = {pred_ori_label} ({pred_ori[0][pred_ori_label]})")

    print(f"avg L0 =  {np.average(l0s)}")
    print(f"avg L2 = {np.average(l2s)}")
    print(f"L0 = {l0s}")
    print(f"L2 = {l2s}")