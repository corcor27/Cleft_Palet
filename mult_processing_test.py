# SuperFastPython.com
# example of a parallel for loop
import time
from time import sleep
from random import random
from multiprocessing import Pool
"""
# task to execute in another process
def task(arg):
    # generate a value between 0 and 1
    value = random()
    # block for a fraction of a second to simulate work
    sleep(value)
    # return the generated value
    return value
start = time.time()
# entry point for the program
if __name__ == '__main__':
    # create the process pool
    with Pool() as pool:
        # call the same function with different data in parallel
        for result in pool.map(task, range(10)):
            # report the value to show progress
            print(result)
"""

def task(arg):
    # generate a value between 0 and 1
    value = random()
    # block for a fraction of a second to simulate work
    sleep(value)
    # return the generated value
    return value
start = time.time()
# entry point for the program
if __name__ == '__main__':
    # call the same function with different data sequentially
    for result in map(task, range(10)):
        # report the value to show progress
        print(result)
        

end = time.time()
print("time taken {}".format(end- start))