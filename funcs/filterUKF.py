import numpy as np
from numpy.linalg import norm
from math import sqrt, log
from scipy.integrate import solve_ivp
from funcs.sp3 import getInterpolatedSatPos
from spiceypy import spkpos
from funcs.coordConvert import c2tMatrix
from funcs.forces import getEarthHarmForce, solarRadiation


class InitMatricesUKF(object):
    """ Initialize matrices for UKF filter"""
    def __init__(self, init, P, Q, R):
        """Constructor"""
        self.P = P * np.eye(init.n)     # Error-covariance matrix
        self.Q = Q * np.eye(init.n)     # Noise covariance matrix for state equation
        self.R = R * np.eye(init.m)     # Noise covariance matrix for observation equation
        self.Wm = np.array([])
        self.leftMatrix = np.array([])
        self.diagMatrix = np.array([])


def calcH(x):
    """
    Calculate nonlinear function h(x) for dynamic system equations.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    h : array_like
        Result of using h(x).

    """
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])
    h = H.dot(x)
    return h


def calcF(t, x, sp3InerTab, init, consts):
    """
    Calculate nonlinear function f(x(t)) for dynamic system equations.

    Parameters
    ----------
    t : float
        Moment of time.
    x : array_like
        Input array of shape (6, 1) containing satellite coordinates and velocity.
    sp3InerTab : array_like
        Array of shape (96, 96) containing satellites coordinates in Earth-centered (Terrestrial)
        inertial coordinate system.
    init: class
        Contain variables describing the problem.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    gf : array_like
        Result of using f(x(t)).

    """
    # time1 = time.time()
    satPos = x[0:3]  # Satellite coordinates
    satSpeed = x[3:]  # Satellite velocity

    et = (init.mjd + (t + consts.dTT - consts.dUTC) / 86400 - consts.mjd00) * 86400
    # Get Sun and Moon coordinates
    sunPos = spkpos('SUN', et, 'J2000', 'NONE', 'EARTH')
    moonPos = spkpos('MOON', et, 'J2000', 'NONE', 'EARTH')

    sunR = norm(sunPos[0])
    sunR3 = sunR**3

    satR = norm(satPos)
    satR3 = satR**3

    satSun = sunPos[0] - satPos
    satSunR = norm(satSun)
    satSunR3 = satSunR**3

    satMoon = moonPos[0] - satPos
    satMoonR = norm(satMoon)
    satMoonR3 = satMoonR**3

    # Form matrix to convert from Celestial coordinate system to Earth-centered (Terrestrial) coordinate system
    # time2 = time.time()
    rot = c2tMatrix(t + consts.dGPS + consts.dUTC, init, consts)
    # print('c2tMatrix = ', time.time() - time2)

    # Calculate the gravitational effect of the Sun and the Moon
    fSun = consts.gmSun * (satSun / satSunR3 - sunPos[0] / sunR3)
    fMoon = consts.gmMoon * (satMoon / satMoonR3 - moonPos[0] / satMoonR3)

    fEarthCenter = -consts.gmEarth * satPos / satR3

    # Calculate perturbations from the non-sphericity of the Earth geopotential
    utc = init.mjd + (t + consts.dTT - consts.dUTC) / 86400
    # time3 = time.time()
    fEarthHarm = getEarthHarmForce(utc, satPos, sunPos[0], moonPos[0], rot, init.mjd, consts)
    # print('getEarthHarmForce = ', time.time() - time3)
    fEarthHarm = consts.gmEarthEGM08 * fEarthHarm

    # Calculate disturbance from the solar radiation pressure on the satellite
    # time4 = time.time()
    fSunRad = solarRadiation(satPos, satSpeed, sunPos[0], moonPos[0], init.radParams, consts)
    # print('solarRadiation = ', time.time() - time4)

    f = fEarthCenter + fEarthHarm + fSun + fMoon + fSunRad
    f = f + (-consts.gmEarth * satPos / satR3)

    pos, speed = getInterpolatedSatPos(t, 1, sp3InerTab)
    gf = np.hstack([speed, f])
    # print('all =', time.time() - time1)
    return gf


def calcDiffFunc(t, inputVec, sp3InerTab, L, lambda_, init, matr, consts):
    """
    Calculate function for differential equation in Unscented Kalman filter.

    Parameters
    ----------
    t : float
        Moment of time.
    inputVec : array_like
        Array of shape (n*n + n, 1) containing vector X and matrix P.
    sp3InerTab : array_like
        Array of shape (96, 96) containing satellites coordinates in Earth-centered (Terrestrial)
        inertial coordinate system.
    L: int
        Size of M(x(t0)).
    lambda_: float
        Variable for UKF filter.
    init: class
        Contain variables describing the problem.
    matr: class
        Contain matrices for UKF filter.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    outVec : array_like
        Array of shape (n*n + n, 1) containing vector X and matrix P after using the function.

    """
    n = init.n

    prevX = inputVec[0:n]
    prevPVec = inputVec[n:n*n + n]
    prevP = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            prevP[i][j] = prevPVec[i + n*j]

    sqrtMatrix = (L + lambda_) * prevP
    sigmaX = np.zeros((2*L+1, L))
    sigmaX[0, :] = prevX

    for i in range(L):
        sigmaX[i+1, :] = prevX + sqrtMatrix[i, :]
        sigmaX[i+1 + L, :] = prevX - sqrtMatrix[i, :]

    sigmaFX = np.zeros((2*L+1, L))
    for i in range(2*L+1):
        sigmaFX[i, :] = calcF(t, sigmaX[i, :], sp3InerTab, init, consts)

    nextX = np.dot(sigmaFX.transpose(), np.vstack(matr.Wm))

    P1 = sigmaX.transpose().dot(matr.leftMatrix).dot(matr.diagMatrix).dot(matr.leftMatrix.transpose()).dot(sigmaFX)
    P2 = sigmaFX.transpose().dot(matr.leftMatrix).dot(matr.diagMatrix).dot(matr.leftMatrix.transpose()).dot(sigmaX)
    P3 = matr.Q
    nextP = P1 + P2 + P3

    nextPVec = np.zeros((n*n, 1))
    for j in range(n):
        for i in range(n):
            nextPVec[i + n * j] = nextP[i][j]

    outVec = np.hstack([nextX.transpose(), nextPVec.transpose()])
    outVec = np.reshape(outVec, n*n + n)
    return outVec


def filterUKF(sp3InerTab, init, consts):
    """
    Unscented Kalman filter.

    Parameters
    ----------
    sp3InerTab : array_like
        Array of shape (96, 96) containing satellites coordinates in Earth-centered (Terrestrial)
        inertial coordinate system.
    init: class
        Contain variables describing the problem.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    tempVar : float
        Functional value for minimization.

    """
    n = init.n
    m = init.m
    N = init.N
    timeGrid = init.timeGrid

    matr = InitMatricesUKF(init, 0.1, 0.0001, 1)

    tabA = int(timeGrid[0] / 900)
    tabB = int(timeGrid[N - 1] / 900 + 1)
    currentSp3Tab = sp3InerTab[tabA:tabB, :]

    Y = np.zeros((m, N))
    tmp = np.zeros(n)
    for i in range(N):
        tmp[:init.m] = currentSp3Tab[i, :]
        Y[:, i] = calcH(tmp)

    Mx0 = np.zeros(6)
    Mx0[0:3], Mx0[3:] = getInterpolatedSatPos(timeGrid[0], 1, sp3InerTab)

    L = Mx0.size
    alpha = 0.01
    beta = 2.0
    coefK = 0.0
    lambda_ = alpha * alpha * (L + coefK) - L

    Wc = np.zeros(2*L+1)
    matr.Wm = np.zeros(2*L+1)
    matr.Wm[0] = lambda_ / (L + lambda_)
    Wc[0] = lambda_ / (L + lambda_) + (1 - alpha**2 + beta)

    for i in range(1, 2*L+1):
        matr.Wm[i] = 1 / (2 * (L + lambda_))
        Wc[i] = matr.Wm[i]

    matr.diagMatrix = np.diag(Wc)

    tmpMatrix = np.zeros((2*L+1, 2*L+1))
    for i in range(2*L+1):
        tmpMatrix[i, :] = matr.Wm
    tmpMatrix = tmpMatrix.transpose()

    matr.leftMatrix = np.eye(2*L+1) - tmpMatrix

    prX = np.zeros((n, N))
    corX = np.zeros((n, N))
    prY = np.zeros((m, N))

    corX[:, 0] = Mx0
    corP = matr.P

    seq = np.zeros((m, N))
    tempVar = 0

    diffFunc = lambda x, y: calcDiffFunc(x, y, sp3InerTab, L, lambda_, init, matr, consts)

    for K in range(1, N):
        # Solve differential equation
        odeSolution = solve_ivp(diffFunc, [timeGrid[K - 1], timeGrid[K]],
                                np.hstack((corX[:, K - 1], np.reshape(corP, n * n))))
        lenT = odeSolution.t.size
        prX[:, K] = odeSolution.y[0:n, lenT-1]
        prPVec = odeSolution.y[init.n:n*n + n, lenT-1]

        prP = np.zeros((n, n))
        for j in range(n):
            for i in range(n):
                prP[i, j] = prPVec[i + n * j]

        sqrtMatrix = np.linalg.cholesky((L + lambda_) * prP)

        sigmaX = np.zeros((2*L+1, L))
        sigmaX[0, :] = prX[:, K]

        for i in range(L):
            sigmaX[i+1, :] = prX[:, K] + sqrtMatrix[i, :]
            sigmaX[i+1+L, :] = prX[:, K] - sqrtMatrix[i, :]

        sigmaY = np.zeros((2*L+1, m))

        for i in range(2*L+1):
            sigmaY[i, :] = calcH(sigmaX[i, :])

        prY[:, K] = sigmaY.transpose().dot(matr.Wm.transpose())

        Pyy = sigmaY.transpose().dot(matr.leftMatrix).dot(matr.diagMatrix)\
                  .dot(matr.leftMatrix.transpose()).dot(sigmaY) + matr.R
        Pxy = sigmaX.transpose().dot(matr.leftMatrix).dot(matr.diagMatrix).dot(matr.leftMatrix.transpose()).dot(sigmaY)

        KK = np.dot(Pxy, np.linalg.pinv(Pyy))
        seq[:, K] = Y[:, K] - prY[:, K]

        corX[:, K] = prX[:, K] + np.dot(KK, seq[:, K])
        corP = prP - np.dot(KK, Pyy).dot(KK.transpose())

        # Calculate functional value
        tempVar += seq[:, K].transpose().dot(np.linalg.inv(Pyy)).dot(seq[:, K]) + log(np.linalg.det(Pyy))

    resX = corX[0:3, :]
    corXError = currentSp3Tab.transpose() - resX
    errorNorm = np.sqrt(np.sum(np.power(corXError, 2), 0))
    RMSE = norm(errorNorm) / sqrt(errorNorm.size)

    return tempVar, resX, RMSE




