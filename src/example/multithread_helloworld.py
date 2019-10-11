from threading import Thread


def display(name, start, end):
    for num in range(start, end):
        print(f'{name}: {num}')

if __name__ == '__main__':
    t1 = Thread(target=display, args=('thread 1', 2, 2000000))
    t2 = Thread(target=display, args=('thread 2', 100, 20000000))
    t1.start()
    t2.start()