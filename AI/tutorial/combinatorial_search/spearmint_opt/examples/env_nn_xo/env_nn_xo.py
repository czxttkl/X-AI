import math
import numpy as np

d = 3

def evaluate(job_id, params):
    x = params['X']

    # obj = float(np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0]- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10)

    if np.sum(x) > d:
        return np.nan

    # obj = np.sum(x[2:5])
    # obj = x[0] + x[1] + x[3] - np.log(1 / np.abs(np.sum(x) - d + 0.0001))
    obj =  x[0] + x[1] + x[2]

    # con1 = float(5.0 - y)  # y <= 5
    # con1 = float(y - 5.0)  # y >= 5
    # con1 = float(x[1] - x[0] + 0.5)  # x <= y
    # con2 = float(x[0] - x[1] + 0.5)  # y <= x
    # con1 = float(np.sum(x) - d + 0.5)  # |x| >= d
    # con2 = float(d -
    #              np.sum(x)
    #              )  # |x| <= d

    return {
        "obj"       : obj
    }

    # True minimum is at 2.945, 2.945, with a value of 0.8447


def main(job_id, params):
    try:
        return evaluate(job_id, params)
    except Exception as ex:
        print(ex)
        print('An error occurred in branin_con.py')
        return np.nan
