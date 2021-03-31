import numpy as np
from math import sin, cos, atan, atan2, sqrt, pi


def deltaAT(year, month, mjd, time):
    """
    For a given UTC date, calculate delta(AT) = TAI-UTC.

    This function is based on International Astronomical Union's
    SOFA (Standards of Fundamental Astronomy) software collection.
    https://www.iausofa.org/

    Parameters
    ----------
    year : int
        Input Gregorian year.
    month : int
        Input Gregorian month.
    mjd : int
        Input Modified Julian date.
    time : float
        Input time.

    Returns
    -------
    delta : float
        (TAI - UTC) in seconds.

    """
    # Number of Delta(AT) changes(increase by 1 for each new leap second)
    deltaATN = 42

    # Number of Delta(AT) expressions before leap seconds were introduced
    eraN = 14

    # Dates(year, month) on which new Delta(AT) came into force
    idAT = np.array([[1960, 1], [1961, 1], [1961, 8], [1962, 1], [1963, 11], [1964, 1], [1964, 4],
                     [1964, 9], [1965, 1], [1965, 3], [1965, 7], [1965, 9], [1966, 1], [1968, 2],
                     [1972, 1], [1972, 7], [1973, 1], [1974, 1], [1975, 1], [1976, 1], [1977, 1],
                     [1978, 1], [1979, 1], [1980, 1], [1981, 7], [1982, 7], [1983, 7], [1985, 7],
                     [1988, 1], [1990, 1], [1991, 1], [1992, 7], [1993, 7], [1994, 7], [1996, 1],
                     [1997, 7], [1999, 1], [2006, 1], [2009, 1], [2012, 7], [2015, 7], [2017, 1]])

    # New Delta(AT) which came into force on the given dates
    dAT = np.array([1.4178180, 1.4228180, 1.3728180, 1.8458580, 1.9458580, 3.2401300, 3.3401300, 3.4401300,
                     3.5401300, 3.6401300, 3.7401300, 3.8401300, 4.3131700, 4.2131700,
                     10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                     26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37])

    # Reference dates(MJD) and drift rates(s / day), pre leap seconds
    drift = np.array([[37300, 0.001296], [37300, 0.001296], [37300, 0.001296], [37665, 0.0011232], [37665, 0.0011232],
                      [38761, 0.001296], [38761, 0.001296], [38761, 0.001296], [38761, 0.001296], [38761, 0.001296],
                      [38761, 0.001296], [38761, 0.001296], [39126, 0.002592], [39126, 0.002592]])

    # Combine year and month
    m = 12 * year + month

    # Find the most recent table entry
    flag = 0
    more = True
    for i in range(deltaATN, 1, -1):
        if more:
            flag = i
            more = m < (12 * idAT[i-1, 0] + idAT[i-1, 1])

    # Get the Delta(AT)
    delta = dAT[flag-1]

    # If pre - 1972, adjust for drift
    if flag <= eraN:
        delta = delta + (mjd + time - drift[flag-1, 0]) * drift[flag-1, 1]

    return delta


def rotateRX(angle, r):
    """
    Rotate an r-matrix about the x-axis.

    This function is based on International Astronomical Union's
    SOFA (Standards of Fundamental Astronomy) software collection.
    https://www.iausofa.org/

    Parameters
    ----------
    angle : float
        Input angle in radians.
    r : array_like
        Input r-matrix.

    Returns
    -------
    rx : array_like
        Rotated r-matrix.

    """

    s = sin(angle)
    c = cos(angle)

    rx = np.copy(r)
    rx[1, 0] = c * r[1, 0] + s * r[2, 0]
    rx[1, 1] = c * r[1, 1] + s * r[2, 1]
    rx[1, 2] = c * r[1, 2] + s * r[2, 2]
    rx[2, 0] = - s * r[1, 0] + c * r[2, 0]
    rx[2, 1] = - s * r[1, 1] + c * r[2, 1]
    rx[2, 2] = - s * r[1, 2] + c * r[2, 2]

    return rx


def rotateRY(angle, r):
    """
    Rotate an r-matrix about the y-axis.

    This function is based on International Astronomical Union's
    SOFA (Standards of Fundamental Astronomy) software collection.
    https://www.iausofa.org/

    Parameters
    ----------
    angle : float
        Input angle in radians.
    r : array_like
        Input r-matrix.

    Returns
    -------
    ry : array_like
        Rotated r-matrix.

    """
    s = sin(angle)
    c = cos(angle)

    ry = np.copy(r)
    ry[0, 0] = c * r[0, 0] - s * r[2, 0]
    ry[0, 1] = c * r[0, 1] - s * r[2, 1]
    ry[0, 2] = c * r[0, 2] - s * r[2, 2]
    ry[2, 0] = s * r[0, 0] + c * r[2, 0]
    ry[2, 1] = s * r[0, 1] + c * r[2, 1]
    ry[2, 2] = s * r[0, 2] + c * r[2, 2]

    return ry


def rotateRZ(angle, r):
    """
    Rotate an r-matrix about the z-axis.

    This function is based on International Astronomical Union's
    SOFA (Standards of Fundamental Astronomy) software collection.
    https://www.iausofa.org/

    Parameters
    ----------
    angle : float
        Input angle in radians.
    r : array_like
        Input r-matrix.

    Returns
    -------
    rz : array_like
        Rotated r-matrix.

    """
    s = sin(angle)
    c = cos(angle)

    rz = np.copy(r)
    rz[0, 0] = c * r[0, 0] + s * r[1, 0]
    rz[0, 1] = c * r[0, 1] + s * r[1, 1]
    rz[0, 2] = c * r[0, 2] + s * r[1, 2]
    rz[1, 0] = - s * r[0, 0] + c * r[1, 0]
    rz[1, 1] = - s * r[0, 1] + c * r[1, 1]
    rz[1, 2] = - s * r[0, 2] + c * r[1, 2]

    return rz


def calcS(date1, date2, x, y, consts):
    """
    The CIO locator s, positioning the Celestial Intermediate Origin on
    the equator of the Celestial Intermediate Pole, given the CIP's X,Y
    coordinates.  Compatible with IAU 2000A precession-nutation.

    This function is based on International Astronomical Union's
    SOFA (Standards of Fundamental Astronomy) software collection.
    https://www.iausofa.org/

    Parameters
    ----------
    date1, date2 : float
        TT as a 2-part Julian Date.
    x, y : float
        CIP coordinates.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    s : float
        CIO locator s in radians. The CIO locator s is the difference between the right ascensions
        of the same point in two systems:  the two systems are the GCRS and the CIP,CIO,
        and the point is the ascending node of the CIP equator.

    """
    # Number of terms in the series
    ns0 = 33
    ns1 = 3
    ns2 = 25
    ns3 = 4

    # Interval between fundamental epoch J2000.0 and current date(JC)
    t = ((date1 - consts.dj00) + date2) / consts.djc

    tvec = np.array([1, t, t ** 2, t ** 3, t ** 4])

    fundA = np.zeros(8)

    # Fundamental Arguments
    fundA[0:5] = consts.fundArg.dot(tvec) % consts.turnas * consts.asec2rad

    # Mean longitude of Venus (IERS Conventions 2003)
    fundA[5] = (3.176146697 + 1021.3285546211 * t) % (2*pi)

    # Mean longitude of Earth (IERS Conventions 2003)
    fundA[6] = (1.753470314 + 628.3075849991 * t) % (2*pi)

    # General accumulated precession in longitude
    fundA[7] = (0.024381750 + 0.00000538691 * t) * t

    # Evaluate s
    # Polynomial coefficients
    sp = [94e-6, 3808.35e-6, -119.94e-6, -72574.09e-6, 27.70e-6, 15.61e-6]

    for i in range(ns0):
        tmp = consts.cs0[:, i].dot(fundA)
        s0 = sp[0] + consts.ss0[0, i] * sin(tmp) + consts.ss0[1, i] * cos(tmp)

    for i in range(ns1):
        tmp = consts.cs1[:, i].dot(fundA)
        s1 = sp[1] + (consts.ss1[0, i] * sin(tmp) + consts.ss1[1, i] * cos(tmp))

    for i in range(ns2):
        tmp = consts.cs2[:, i].dot(fundA)
        s2 = sp[2] + (consts.ss2[0, i] * sin(tmp) + consts.ss2[1, i] * cos(tmp))

    for i in range(ns3):
        tmp = consts.cs3[:, i].dot(fundA)
        s3 = sp[3] + (consts.ss3[0, i] * sin(tmp) + consts.ss3[1, i] * cos(tmp))

    tmp = consts.cs4.dot(fundA)
    s4 = sp[4] + (consts.ss4[0] * sin(tmp) + consts.ss4[1] * cos(tmp))

    s = (sp[0] + (sp[1] + (sp[2] + (sp[3] + (sp[4] + sp[5] * t) * t) * t) * t) * t) * consts.asec2rad - x * y / 2

    return s


def calcNutation(date1, date2, consts):
    """
    Nutation, IAU 2000A model (MHB2000 luni-solar and planetary nutation
    with free core nutation omitted).

    This function is based on International Astronomical Union's
    SOFA (Standards of Fundamental Astronomy) software collection.
    https://www.iausofa.org/

    Parameters
    ----------
    date1, date2 : float
        TT as a 2-part Julian Date.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    dp, de : float
        Nutation, luni-solar + planetary.

    """
    # Units of 0.1 microarcsecond to radians
    unit2rad = consts.asec2rad / 1e7

    # Number of terms in the luni-solar nutation model
    lunSolarN = 678

    # Number of terms in the planetary nutation model
    planetN = 687

    lunSolarMult = np.reshape(consts.lunSolarArgMult, (lunSolarN, 5)).transpose()
    lunSolarNutCoef = np.reshape(consts.lunSolarNutationCoef, (lunSolarN, 6)).transpose()
    planetArgMult = np.reshape(consts.planetArgMult, (planetN, 14)).transpose()
    planetNutCoef = np.reshape(consts.planetNutationCoef, (planetN, 4)).transpose()

    # Interval between fundamental date J2000.0 and given date(JC)
    t = ((date1 - consts.dj00) + date2) / consts.djc
    tvec = np.array([1, t, t**2, t**3, t**4])
    tvec = np.vstack(tvec)

    # Fundamental arguments
    fundA = consts.fundArg.dot(tvec) % consts.turnas * consts.asec2rad

    # Initialize the nutation values
    dp = 0
    de = 0

    # Summation of luni - solar nutation series(in reverse order)

    # Argument and functions
    arg = (lunSolarMult.transpose().dot(np.vstack(fundA))) % (2*pi)

    for i in range(lunSolarN):
        sarg = sin(arg[i])
        carg = cos(arg[i])
        # Term
        dp = dp + (lunSolarNutCoef[0, i] + lunSolarNutCoef[1, i] * t) * sarg + lunSolarNutCoef[2, i] * carg
        de = de + (lunSolarNutCoef[3, i] + lunSolarNutCoef[4, i] * t) * carg + lunSolarNutCoef[5, i] * sarg

    # Convert from 0.1 microarcsec units to radians
    dpLunSolar = dp * unit2rad
    deLunSolar = de * unit2rad

    # n.b.The MHB2000 code computes the luni - solar and planetary nutation
    # in different routines, using slightly different Delaunay
    # arguments in the two cases.This behaviour is faithfully
    # reproduced here.Use of the IERS 2003 expressions for both
    # cases leads to negligible changes, well below 0.1 microarcsecond.
    mhbA = consts.mhbArg.dot(tvec[0:2]) % (2*pi)

    # Planetary longitudes, Mercury through Uranus (IERS 2003)
    planetA = consts.planetArg.dot(tvec[0:2]) % (2*pi)

    # Neptune longitude (MHB2000)
    neptLongitude = (5.321159000 + 3.8127774000 * t) % (2*pi)

    # General accumulated precession in longitude (IERS 2003)
    pr = (0.024381750 + 0.00000538691 * t) * t

    # Initialize the nutation values
    dp = 0
    de = 0

    # Summation of planetary nutation series ( in reverse order)
    planetArgMult = planetArgMult.transpose()
    arg = (planetArgMult[:, 0:5].dot(mhbA) + planetArgMult[:, 5:12].dot(planetA) +
           np.vstack(planetArgMult[:, 12].dot(neptLongitude)) + np.vstack(planetArgMult[:, 13].dot(pr))) % (2*pi)
    for i in range(planetN):
        sarg = sin(arg[i])
        carg = cos(arg[i])
        # Term
        dp = dp + planetNutCoef[0, i] * sarg + planetNutCoef[1, i] * carg
        de = de + planetNutCoef[2, i] * sarg + planetNutCoef[3, i] * carg

    # Convert from 0.1 microarcsec units to radians.
    dpPlanet = dp * unit2rad
    dePlanet = de * unit2rad

    # Add luni - solar and planetary components.
    dp = dpLunSolar + dpPlanet
    de = deLunSolar + dePlanet
    return dp, de


def calcXYS(date1, date2, consts):
    """
    For a given TT date, compute the X,Y coordinates of the Celestial
    Intermediate Pole and the CIO locator s, using the IAU 2000A
    precession-nutation model.

    This function is based on International Astronomical Union's
    SOFA (Standards of Fundamental Astronomy) software collection.
    https://www.iausofa.org/

    Parameters
    ----------
    date1, date2 : float
        TT as a 2-part Julian Date.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    x, y : float
        The Celestial Intermediate Pole coordinates. X, Y components
        of the unit vector in the Geocentric Celestial Reference System.
    s: float
        The CIO locator s (in radians) positions the Celestial
        Intermediate Origin on the equator of the CIP.

    """
    # Form the bias-precession-nutation matrix, IAU 2000A
    # Nutation
    dpNut, deNut = calcNutation(date1, date2, consts)

    # IAU 2000 precession-rate adjustments

    # Precession and obliquity corrections(radians per century)
    preCor = -0.29965 * consts.asec2rad
    oblCor = -0.02524 * consts.asec2rad

    # Interval between fundamental epoch J2000.0 and given date(JC)
    t = ((date1 - consts.dj00) + date2) / consts.djc

    # Precession rate contributions with respect to IAU 1976 / 80
    dpNutPr = preCor * t
    deNutPr = oblCor * t

    # Mean obliquity of date
    meanObl = (84381.448 + (-46.8150 + (-0.00059 + 0.001813 * t) * t) * t) * consts.asec2rad

    # Mean obliquity, consistent with IAU 2000 precession-nutation
    meanOblC = meanObl + deNutPr

    # Frame bias
    # The frame bias corrections in longitude and obliquity
    dpNutBias = -0.041775 * consts.asec2rad
    deNutBias = -0.0068192 * consts.asec2rad

    # The ICRS RA of the J2000.0 equinox(Chapront et al., 2002)
    ra0 = -0.0146 * consts.asec2rad

    # Precession angles(Lieske et al. 1977)
    psia77 = (5038.7784 + (-1.07259 - 0.001147 * t) * t) * t * consts.asec2rad
    oma77 = consts.eps0 + ((0.05127 - 0.007726 * t) * t) * t * consts.asec2rad
    chia = (10.5526 + (-2.38064 - 0.001125 * t) * t) * t * consts.asec2rad

    # Apply IAU 2000 precession corrections
    psia = psia77 + dpNutPr
    oma = oma77 + deNutPr

    # Frame bias matrix: GCRS to J2000.0
    biasMatrix = np.eye(3)
    biasMatrix = rotateRZ(ra0, biasMatrix)
    biasMatrix = rotateRY(dpNutBias * sin(consts.eps0), biasMatrix)
    biasMatrix = rotateRX(-deNutBias, biasMatrix)

    # Precession matrix: J2000.0 to mean of date
    prMatrix = np.eye(3)
    prMatrix = rotateRX(consts.eps0, prMatrix)
    prMatrix = rotateRZ(-psia, prMatrix)
    prMatrix = rotateRX(-oma, prMatrix)
    prMatrix = rotateRZ(chia, prMatrix)

    # Bias - precession matrix: GCRS to mean of date
    biasPrMatrix = prMatrix.dot(biasMatrix)

    # Nutation matrix
    # Build the rotation matrix
    nutMatrix = np.eye(3)
    nutMatrix = rotateRX(meanOblC, nutMatrix)
    nutMatrix = rotateRZ(-dpNut, nutMatrix)
    nutMatrix = rotateRX(-(meanOblC + deNut), nutMatrix)

    # Bias - precession - nutation matrix(classical)
    biasPrNutMatrix = nutMatrix.dot(biasPrMatrix)

    # Extract the X, Y coordinates
    x = biasPrNutMatrix[2, 0]
    y = biasPrNutMatrix[2, 1]

    # Obtain s
    s = calcS(date1, date2, x, y, consts)

    return x, y, s


def c2tMatrix(sec, init, consts):
    """
    Form matrix to convert from Celestial coordinate system to
    Earth-centered (Terrestrial) coordinate system.

    Parameters
    ----------
    sec : int
        Time in seconds.
    init: class
        Contain variables describing the problem.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    polarCelTerMatrix : array_like
        Matrix to convert from Celestial coordinate system to
        Earth-centered (Terrestrial) coordinate system.

    """
    # Polar motion(arcsec->radians)
    xp = 0.0349282 * consts.asec2rad
    yp = 0.4833163 * consts.asec2rad

    # CIP offsets wrt IAU 2000A(mas->radians)
    dx = 0.1725 * consts.asec2rad / 1000
    dy = -0.2650 * consts.asec2rad / 1000

    mjdjd0 = 2400000.5

    tm = sec / 86400
    utc = init.mjd + tm
    delta = deltaAT(init.year, init.month, init.mjd, tm)
    tai = utc + delta / 86400
    tt = tai + 32.184 / 86400
    tut = tm + consts.dUT1 / 86400

    # IAU 2000A, CIO based, using classical angels
    # CIP and CIO, IAU 2000A
    x, y, s = calcXYS(mjdjd0, tt, consts)

    # Add CIP corrections
    x = x + dx
    y = y + dy

    # Obtain the spherical angles E and d
    r2 = x**2 + y**2
    if r2 > 0:
        e = atan2(y, x)
    else:
        e = 0
    d = atan(sqrt(r2 / (1 - r2)))

    # GCRS to CIRS matrix
    matr = np.eye(3)
    matr = rotateRZ(e, matr)
    matr = rotateRY(d, matr)
    matr = rotateRZ(-(e + s), matr)

    # Earth rotation angle
    dj1 = mjdjd0 + init.mjd
    dj2 = tut
    # Days since fundamental epoch
    if dj1 < dj2:
        date1 = dj1
        date2 = dj2
    else:
        date1 = dj2
        date2 = dj1

    t = date1 + (date2 - consts.dj00)

    # Fractional part of T(days)
    f = date1 % 1 + date2 % 1

    # Earth rotation angle at this UT1
    era = (2*pi*(f + 0.7790572732640 + 0.00273781191135448 * t)) % (2*pi)
    if era < 0:
        era += 2*pi

    # Form celestial - terrestial matrix(no polar motion yet)
    celTerMatrix = np.copy(matr)
    celTerMatrix = rotateRZ(era, celTerMatrix)

    # Polar motion matrix(TRIRS->ITRS, IERS 2003)
    # Interval between fundamental epoch J2000.0 and current date(JC).
    t = ((mjdjd0 - consts.dj00) + tt) / consts.djc

    # Approximate s'
    sp = -47e-6 * t * consts.asec2rad

    polarMatrix = np.eye(3)
    polarMatrix = rotateRZ(sp, polarMatrix)
    polarMatrix = rotateRY(-xp, polarMatrix)
    polarMatrix = rotateRX(-yp, polarMatrix)

    # Form celestial - terrestial matrix(including polar motion)
    polarCelTerMatrix = polarMatrix.dot(celTerMatrix)
    return polarCelTerMatrix






