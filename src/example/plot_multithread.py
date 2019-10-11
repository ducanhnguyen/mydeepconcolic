from threading import Thread

import matplotlib.pyplot as plt


def plot():
    fig, ax = plt.subplots()
    ax.plot([1,2,3], [1,2,3])
    plt.show()

def main():
    thread = Thread(target=plot)
    thread.setDaemon(True)
    thread.start()
    print('Done')