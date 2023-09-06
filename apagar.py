"""This code tests the serialization of data into `pathos`' subprocesses.
Uncomment the lines in main() to change what is serialized and copied.
"""

import sys
import time

import numpy as np
from pathos.pools import ProcessPool

from rtrend_tools.utils import map_parallel_or_sequential


# Testing whether pathos serializes big unused structures.
class BigArray:

    def __init__(self, size=150E6):
        self.size = size
        self.data = np.arange(size)


def main():
    ncpus = 3
    a = BigArray()

    def task(inputs):
        # print(a.data.shape)  # This causes each process to hold a copy of the big array

        print("sleeping...")
        time.sleep(10)

    content = list(range(ncpus))
    # content = ncpus * [a]  # This passes the big array as an argument, causing it to be copied

    map_parallel_or_sequential(task, content, ncpus)


if __name__ == '__main__':
    main()
