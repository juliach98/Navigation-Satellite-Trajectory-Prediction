import numpy as np
from funcs.sp3 import getSatPos
from scipy.optimize import minimize
import time
from funcs.filterUKF import filterUKF
from funcs.init import InitializeProblem
from spiceypy import furnsh
from os.path import abspath
from math import log, pi
from funcs.constants import Constants
import argparse


def mlParamIndent(params, sp3InerTab, init, consts):
    """
    Maximum likelihood parametric identification.

    Parameters
    ----------
    params : array_like
        Parameters for identification.
    sp3InerTab : array_like
        Array of shape (96, 96) containing satellites coordinates in Earth-centered (Terrestrial)
        inertial coordinate system.
    init: class
        Contain variables describing the problem.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    f : float
        Functional value for minimization.

    """
    init.radParams = params
    print('radParams =', init.radParams)
    f = 0
    tempF = filterUKF(sp3InerTab, init, consts)[0]

    f = (f + tempF) / 2 + log(2*pi) * (init.N * init.m) / 2
    print('f =', f)
    return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', required=True, type=int)
    parser.add_argument('--month', required=True, type=int)
    parser.add_argument('--day', required=True, type=int)
    args = parser.parse_args()

    # Project contains .sp3 files for dates from 10.07.2016 to 30.07.2016
    # To find data for more dates see: https://cddis.nasa.gov/
    init = InitializeProblem(args.year, args.month, args.day)
    # All constants that are using for the computations
    consts = Constants()

    # Path to general-purpose ephemeris files that helps observing the planets or their moons
    filePath = "./resources/de421.bsp"
    furnsh(abspath(filePath))

    sat = 5  # Satellite number

    # Get certain satellite coordinates for the chosen date
    sp3InerTab = getSatPos(sat, init, consts)

    # Parametric identification for solar radiation pressure parameters
    initial_theta = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    mlFunc = lambda x: mlParamIndent(x, sp3InerTab, init, consts)
    bnds = ((0, 2), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1))

    start = time.time()
    res = minimize(mlFunc, initial_theta, bounds=bnds)
    print('time =', time.time() - start)

    # Predict satellite trajectory for the next day
    init.day += 1
    init.radParams = res.x

    sp3InerTab = getSatPos(sat, init, consts)

    tmp, trajectory, RMSE = filterUKF(sp3InerTab, init, consts)

    with open('predicted_trajectory.txt', 'w') as f:
        for j in range(trajectory.shape[1]):
            for i in range(trajectory.shape[0]):
                f.write(str(trajectory[i][j]) + '\t')
            f.write('\n')

    print('RSME =', RMSE)
