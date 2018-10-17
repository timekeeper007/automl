import os
import time

import numpy as np

MSIZE = 500
BASE_PATH = './tmp'
TIMES_TASK1 = 1000
TIMES_TASK2 = 50


def task1(matrix):
    np.dot(matrix, matrix)


def run_task1(i):
    return task1(np.random.random((MSIZE, MSIZE)))


def task2(path):
    with open(path, 'w') as f:
        for i in range(1000):
            f.write('\n'.join('0' * 10 for _ in range(1000)))


def run_task2(i):
    task2(os.path.join(BASE_PATH, '{}.txt'.format(i)))


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('{}: {:.2f}'.format(self.name, time.time() - self.start))


def run_sequential():
    with Timer('usual: sequential_task1'):
        for i in range(TIMES_TASK1):
            run_task1(i)

    with Timer('usual: sequential_task2'):
        for i in range(TIMES_TASK2):
            run_task2(i)

import threading

def run_threading():

    threads = [threading.Thread(target=run_task1, args=(_,)) for _ in range(TIMES_TASK1)]+ \
              [threading.Thread(target=run_task2, args=(_,)) for _ in range(TIMES_TASK2)]
    with Timer('thread: sequential_task1'):
        for t in threads[:TIMES_TASK1]:
            t.start()

    with Timer('thread: sequential_task2'):
        # threads2 = [threading.Thread(target=run_task2, args=(_,)) for _ in range(TIMES_TASK2)]
        for t in threads[TIMES_TASK1:]:
            t.start()

import multiprocessing
def run_multiprocessing():


    with multiprocessing.Pool() as pool:
        with Timer('multi: sequential_task1'):
            pool.map(run_task1, [i for i in range(TIMES_TASK1)], chunksize=2)

        with Timer('multi: sequential_task2'):
        # with multiprocessing.Pool() as pool:
            pool.map(run_task2, [i for i in range(TIMES_TASK2)], chunksize=2)


def main():
    if not os.path.isdir(BASE_PATH):
        os.mkdir(BASE_PATH)



    # run_sequential()

    run_threading()

    run_multiprocessing()


if __name__ == '__main__':
    main()