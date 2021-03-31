import numpy as np
import gnsspy as gnss
from os.path import abspath
from funcs.dateConvert import mjd2gpsw
from funcs.coordConvert import c2tMatrix


def readEphemerides(mjd):
    """
    Read .sp3 file containing data on satellites flights.

    Parameters
    ----------
    mjd : int
        Input Modified Julian date.

    Returns
    -------
    nsats : int
        Number of satellites.
    ephTab : array_like
        Array of shape (96, 96) containing coordinates in geographic (Earth) coordinate system
        of each satellite from 00h00min00sec to 23h45min00sec with a frequency of 15 min.
    """
    week, day = mjd2gpsw(mjd)
    sp3File = abspath("./resources/igs" + str(int(week)) + str(int(day)) + ".sp3")

    res = gnss.read_sp3File(sp3File)
    nsats = int(res.T.columns[-1][1][1:])
    ephTab = np.zeros((nsats*3, nsats*3))

    for i in range(nsats*3):
        for j in range(nsats):
            ephTab[i][3*j] = res.X[nsats*i + j]
            ephTab[i][3*j+1] = res.Y[nsats*i + j]
            ephTab[i][3*j+2] = res.Z[nsats*i + j]

    return nsats, ephTab


def toInertial(ephTab, timeTab, init, consts):
    """
    Convert satellites coordinates from the geographic (Earth) coordinate system
    to the Earth-centered inertial coordinate frame.

    Parameters
    ----------
    ephTab : array_like
        Input array of shape (96, 96) containing coordinates in geographic (Earth) coordinate system
        of each satellite from 00h00min00sec to 23h45min00sec with a frequency of 15 min.
    mjd : int
        Modified Julian date.
    timeTab : array_like
        Array containing time table for satellites coordinates in .sp3 file.

    Returns
    -------
    inerTab : array_like
        Array of shape (96, 96) containing satellites coordinates in Earth-centered (Terrestrial)
        inertial coordinate system.
    """
    epochCnt = ephTab.shape[0]
    inerTab = np.zeros(ephTab.shape)

    for epoch in range(epochCnt):
        # Form matrix to convert from Celestial coordinate system to Earth-centered (Terrestrial) coordinate system
        c2tMatr = c2tMatrix(timeTab[epoch], init, consts)

        for sat in range(int(ephTab.shape[1]/3)):
            idx0 = sat*3
            idx1 = (sat + 1)*3

            res = np.transpose(c2tMatr).dot(np.vstack(ephTab[epoch][idx0:idx1]))
            inerTab[epoch][idx0:idx1] = np.hstack(res)

    return inerTab


def getInterpolatedSatPos(time, sat, posTab):
    """
    Interpolate satellite coordinates and velocity at an arbitrary moment in time using Everetta method.

    Parameters
    ----------
    time : float
        Moment of time for interpolation.
    sat : int
        Satellite number.
    posTab : array_like
        Array of shape (96, 96) containing satellites coordinates in Earth-centered (Terrestrial)
        inertial coordinate system.

    Returns
    -------
    pos : array_like
        Array of shape (3, 1) containing satellite coordinates after interpolation.
    vel : array_like
        Array of shape (3, 1) containing satellite velocity after interpolation.
    """
    step = 900
    timeStart = 0
    ik = np.floor((time-timeStart) / step + 1)
    T0 = (ik - 1) * step + timeStart
    imin = int(ik - 5)
    imax = int(ik + 5)
    jmin = int((sat-1)*3)
    jmax = int((sat-1)*3 + 3)
    pos = posTab[imin:imax, jmin:jmax]

    if pos.any() == 0:
        pos = np.zeros(3)
        vel = np.zeros(3)
        return pos, vel

    xNext = np.zeros((5, 3))
    xNext[0, :] = pos[5, :]
    xNext[1:, :] = pos[6:10, :] + np.flip(pos[1:5, :], axis=0)

    xPrev = np.zeros((5, 3))
    xPrev[0, :] = pos[4, :]
    xPrev[1:, :] = pos[5:9, :] + np.flip(pos[0:4, :], axis=0)

    # Everett coefficients
    everettCoef = np.array([
        [1.7873015873015873, -0.9359567901234568, 0.1582175925925926, -0.9755291005291005e-02, 0.1929012345679012e-03],
        [-0.4960317460317460, 0.6057098765432098, -0.1171296296296296, 0.7605820105820106e-02, -0.1543209876543210e-03],
        [0.1206349206349206, -0.1632716049382716, 0.4606481481481481e-01, -0.3505291005291005e-02,
         0.7716049382716048e-04],
        [-0.1984126984126984e-01, 0.2779982363315696e-01, -0.8796296296296296e-02, 0.8597883597883598e-03,
         -0.2204585537918871e-04],
        [0.1587301587301587e-02, -0.2259700176366843e-02, 0.7523148148148148e-03, -0.8267195767195767e-04,
         0.2755731922398589e-05]])

    # Multiplicative factors
    multFactors = np.array([[1.0, 3.0, 5.0, 7.0, 9.0], [1.0, 3.0, 5.0, 7.0, 9.0], [1.0, 3.0, 5.0, 7.0, 9.0]])

    pNext = xNext.transpose().dot(everettCoef)
    pPrev = xPrev.transpose().dot(everettCoef)

    vNext = pNext * multFactors / step
    vPrev = pPrev * multFactors / step

    cNext = np.vstack((pNext, vNext))
    cPrev = np.vstack((pPrev, vPrev))

    p = (time - T0) / step
    q = 1.0 - p
    p2 = p * p
    q2 = q * q

    sum1 = np.zeros(6)
    sum2 = np.zeros(6)

    for j in range(4, 0, -1):
        sum1 = p2 * (sum1 + cNext[:, j])
        sum2 = q2 * (sum2 + cPrev[:, j])

    pos = p * (pNext[:, 0] + sum1[0:3]) + q * (pPrev[:, 0] + sum2[0:3])
    vel = (vNext[:, 0] + sum1[3:]) - (vPrev[:, 0] + sum2[3:])
    return pos, vel


def getSatPos(sat, init, consts):
    """
    Get certain satellite coordinates for the chosen date.

    Parameters
    ----------
    sat : float
        Satellite number.
    mjd : int
        Modified Julian date.

    Returns
    -------
    sp3InerTab : array_like
        Array of shape (96, 3) containing satellite coordinates in Earth-centered (Terrestrial)
        inertial coordinate system.

    """
    # Differential UTC
    dUTC = -36.0
    # Differential GPS
    dGPS = 19.0

    # Get data on satellites flights from .sp3 file
    nsats, eph = readEphemerides(init.mjd)

    sp3TimeTab = np.linspace(0, 85500, nsats * 3)  # Time grid

    # Convert satellites coordinates from the geographic (Earth) coordinate system
    # to the Earth-centered inertial coordinate frame
    ephIner = toInertial(eph, sp3TimeTab + dGPS + dUTC, init, consts)

    idx0 = (sat - 1) * 3
    idx1 = sat * 3
    # Get coordinates for certain satellite
    sp3InerTab = ephIner[:, idx0:idx1]

    return sp3InerTab
