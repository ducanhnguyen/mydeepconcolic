from src.log_analyzer import analyze_randomly, analyze_by_threshold
import matplotlib.pyplot as plt

def plot_n_pixel_attack_randomly_vs_directly(plt, type_of_attack):  # simard = M1, ann-keras = m2, simple= m3, deepcheck = m4
    # when not using sample ranking algorithm
    num_adv_arr, total_samples = analyze_randomly(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_simard/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '-k', linewidth=1, label='M1 (random)')

    num_adv_arr, total_samples = analyze_randomly(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_ann_keras/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '-sk', linewidth=1, markevery=250, markersize=3, label='M2 (random)')

    num_adv_arr, total_samples = analyze_randomly(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_simple/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '->k', linewidth=1, markevery=250, markersize=3, label='M3 (random)')

    num_adv_arr, total_samples = analyze_randomly(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_deepcheck/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '-.k', linewidth=1, label='M4 (random)')

    # when using sample ranking algorithm
    num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_simard/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '-g', linewidth=1, label='M1 (ranking)')

    num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_ann_keras/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '-sg', linewidth=1, markevery=0.05, markersize=3, label='M2 (ranking)')

    num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_simple/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '->g', linewidth=1, markevery=0.05, markersize=3, label='M3 (ranking)')

    num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
        f'/Users/ducanhnguyen/Documents/mydeepconcolic/result/{type_of_attack}/result/mnist_deepcheck/summary.csv')
    if total_samples is not None and num_adv_arr is not None:
        plt.plot(total_samples, num_adv_arr, '-.g', linewidth=1, label='M4 (ranking)')

    plt.set(xlabel="% attacking samples", ylabel="# adversarial examples")


if __name__ == '__main__':
    # fig, axs = plt.subplots(1,3,sharex=True, sharey=True)
    fig, axs = plt.subplots(3,1)
    # fig.suptitle('Vertically stacked subplots')
    plot_n_pixel_attack_randomly_vs_directly(axs[0], type_of_attack='edgeAttack_delta100_secondLabelTarget')
    plot_n_pixel_attack_randomly_vs_directly(axs[1], type_of_attack='nonzeroAttack_delta100_secondLabelTarget')
    plot_n_pixel_attack_randomly_vs_directly(axs[2], type_of_attack='bestFeatureAttack_delta255_secondLabelTarget')
    plt.tight_layout(pad=0.1)
    plt.show()


    # num_adv_arr, total_samples, threshold_arr = analyze_by_threshold(
    #     '/Users/ducanhnguyen/Documents/mydeepconcolic/result/changeOnEdge_delta100_upperBound/result/mnist_deepcheck/summary.csv')
    # for u, v in zip(num_adv_arr, total_samples):
    #     print(f'%adv = {u}, %total = {v}')
