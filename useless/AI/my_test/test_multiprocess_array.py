"""
Test multiprocessing Array behavior
"""
from multiprocessing import Process, Array
import scipy
import numpy


def f(a):
    a[0] = -a[0]

if __name__ == '__main__':
    # Create the array
    unshared_arr = list(range(5, 10))
    # unshared_arr[1] = numpy.ones(3)   # fail
    print(unshared_arr)
    a = Array('d', unshared_arr)
    print("Originally, arr = %s"%(a))

    # Create, start, and finish the child process
    p = Process(target=f, args=(a,))
    p.start()
    p.join()

    # Print out the changed values
    print("Now, arr = %s"% a)