import numpy as np
from numpy.linalg import norm
from math import pi, atan2, sqrt, sin, cos, acos


def getSphericals(pos, k, rot):
    """
    Convert body coordinates from Greenwich coordinate system to spherical.

    Parameters
    ----------
    pos : array_like
        Body coordinates in Greenwich coordinate system.
    k : int
        Longitudes order.
    rot : array_like
        Array of shape(3, 3) containing rotational matrix.

    Returns
    -------
    latSin : float
        Latitude sine.
    latCos : float
        Latitude cosine.
    lonSin : array_like
        Array of shape(k+1, 1) containing longitude sine.
    lonCos : array_like
        Array of shape(k+1, 1) containing longitude cosine.

    """
    dist = norm(pos)

    latSin = np.dot(rot[2, :], np.vstack(pos)) / dist
    latCos = sqrt(1 - latSin**2)

    lonSin = np.zeros(k+1)
    lonCos = np.zeros(k+1)

    lonSin[0] = 0
    lonCos[0] = 1

    lonSin[1] = np.dot(rot[1, :], np.vstack(pos)) / dist / latCos
    lonCos[1] = np.dot(rot[0, :], np.vstack(pos)) / dist / latCos

    for i in range(2, k+1):
        lonSin[i] = lonSin[i-1]*lonCos[1] + lonCos[i-1]*lonSin[1]
        lonCos[i] = lonCos[i-1]*lonCos[1] - lonSin[i-1]*lonSin[1]

    return latSin, latCos, lonSin, lonCos


def legndr(order, z):
    """
    Calculate the associated Legendre functions and their derivatives.

    Parameters
    ----------
    order : int
        Maximum degree of associated Legendre functions.
    z : float
        Input argument.

    Returns
    -------
    leg : array_like
       Array of shape (order, order + 1) containing the values of the attached Legendre functions
       up to the degree 'order' and order 'order-1'.
    dleg : array_like
       Array of shape (order, order + 1) containing the values of derivatives of the attached Legendre functions
       up to the degree 'order' and order 'order-1'.

    """
    zz = sqrt(1 - z*z)

    leg = np.zeros((order, order+1))
    dleg = np.zeros((order, order+1))

    leg[0][0] = z
    leg[0][1] = zz
    leg[1][0] = 1.5 * z**2 - 0.5
    leg[1][1] = zz * 3.0*z

    for i in range(order):
        if i <= order - 3:
            for j in range(2):
                leg[i+2, j] = ((2*i + 5) * z * leg[i+1][j] - (i + j + 2) * leg[i][j]) / (i + 3 - j)

        for j in range(i):
            leg[i][j+2] = (2*j+2) * z / zz * leg[i][j+1] - (i - j + 1) * (i + j + 2) * leg[i][j]

    for i in range(order):
        dleg[i][0] = leg[i][1] / zz
        for j in range(i+1):
            dleg[i][j+1] = ((j+1) * z * leg[i][j+1] - (i + j + 2) * (i - j + 1) * zz * leg[i][j]) / zz**2

    return leg, dleg


def tideAngles(epoch, consts):
    """
    Calculate five-vector of fundamental arguments of nutation theory and
    Greenwich Mean Sidereal Time expressed in angle units.

    Parameters
    ----------
    epoch : float
        Coordinated Universal Time.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    fund : array_like
        Array of shape (5, 1) containing fundamental arguments of nutation theory.
    gmst : float
        Greenwich Mean Sidereal Time expressed in angle units.

    """
    t = (epoch - consts.mjd00) / consts.djc
    d = (epoch - consts.mjd00)
    tod = d % 1

    tvec = np.array([1, t, t**2, t**3, t**4, t**5])

    # Earth Rotation Angle
    era = (2*pi * (tod + 0.7790572732640 + 0.00273781191135448 * d)) % (2*pi)

    # Greenwich Mean Sidereal Time
    gmst = (era + np.dot(consts.gmstCoef, np.vstack(tvec)) * consts.asec2rad) % (2*pi)

    # Fundamental arguments of nutation theory
    fund = consts.fundArg.dot(tvec[0:5]) * consts.asec2rad

    return fund, gmst


def getTidalHarmVar(utc, sunPos, moonPos, rot, consts):
    """
    Calculate tidal perturbations of the Earth geopotential.

    Parameters
    ----------
    utc : float
        Coordinated Universal Time.
    sunPos : array_like
        Array of shape (3, 1) containing Sun coordinates.
    moonPos : array_like
        Array of shape (3, 1) containing Moon coordinates.
    rot : array_like
        Array of shape(3, 3) containing rotational matrix.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    dC : array_like
        Array of shape (4, 4) containing cosine tidal perturbations of the Earth geopotential.
    dS : array_like
        Array of shape (4, 4) containing sinusoidal tidal perturbations of the Earth geopotential.

    """
    #  Nominal values of solid Earth tide external potential Love numbers
    k2 = np.array([0.30190, complex(0.29830, -0.00144), complex(0.30102, -0.00130), 0])  # Anelastic Earth
    k3 = np.array([0.093, 0.093, 0.093, 0.094])  # Elastic Earth
    k2p = -0.00089
    k = np.vstack([k2, k3])

    sunDist = norm(sunPos)
    moonDist = norm(moonPos)

    # Get sine and cosine of geocentric latitude and east longitude of the Sun
    slats, clats, slons, clons = getSphericals(sunPos, 4, rot)
    # Get sine and cosine of geocentric latitude and east longitude of the Moon
    slatm, clatm, slonm, clonm = getSphericals(moonPos, 4, rot)

    # Get associated Legendre functions for the Sun
    legs = legndr(4, slats)[0]
    # Get associated Legendre functions for the Moon
    legm = legndr(4, slatm)[0]

    dC = np.zeros((4, 4))
    dS = np.zeros((4, 4))
    dCvar = np.zeros((4, 4))
    dSvar = np.zeros((4, 4))

    for i in range(1, 3):
        for j in range(i+2):
            sunEff = consts.gmSun / consts.gmEarthEGM08 * (consts.EarthRadius / sunDist)**(i+2) * legs[i][j] * \
                      (clons[j] - slons[j] * complex(0, 1))
            moonEff = consts.gmMoon / consts.gmEarthEGM08 * (consts.EarthRadius / moonDist)**(i+2) * legm[i][j] * \
                      (clonm[j] - slonm[j] * complex(0, 1))

            T = k[i-1][j] / (2*i+3) * (sunEff + moonEff)

            dC[i][j] = T.real
            dS[i][j] = -T.imag

    i = 3
    for j in range(3):
        moonEff = consts.gmMoon / consts.gmEarthEGM08 * (consts.EarthRadius / moonDist)**3 * legm[1][j] * \
                  (clonm[j] - slonm[j] * complex(0, 1))
        sunEff = consts.gmSun / consts.gmEarthEGM08 * (consts.EarthRadius / sunDist)**3 * legs[1][j] * \
                (clons[j] - slons[j] * complex(0, 1))
        T = k2p / 5 * (sunEff + moonEff)
        dC[i][j] = T.real
        dS[i][j] = -T.imag

    # Get vector of fundamental arguments of nutation theory
    # and Greenwich Mean Sidereal Time expressed in angle units
    fund_arg, gmst = tideAngles(utc, consts)

    theta = gmst + pi - np.dot(consts.n20, np.vstack(fund_arg))
    T = np.dot(consts.a20, np.cos(theta) + complex(0, 1)*np.sin(theta))
    dCvar[1][0] = T.real
    dSvar[1][0] = 0

    theta = gmst + pi - np.dot(consts.n21, np.vstack(fund_arg))
    T = np.dot(consts.a21, np.cos(theta) + complex(0, 1)*np.sin(theta))
    dCvar[1][1] = T.real
    dSvar[1][1] = -T.imag

    theta = gmst + pi - np.dot(consts.n22, np.vstack(fund_arg))
    T = np.dot(consts.a22, np.cos(theta) + complex(0, 1)*np.sin(theta))
    dCvar[1][2] = T.real
    dSvar[1][2] = -T.imag

    normC = np.array([[2.449489742783178, 1.732050807568877, 0, 0],
                      [3.162277660168380, 1.290994448735806, 0.645497224367903, 0],
                      [3.741657386773941, 1.080123449734644, 0.341565025531987, 0.139443337755679],
                      [4.242640687119285, 0.948683298050514, 0.223606797749979, 0.059761430466720]])

    dC = normC * (dC + dCvar)
    dS = normC * (dS + dSvar)

    return dC, dS


def getEarthHarmForce(utc, satPos, sunPos, moonPos, rot, mjd, consts):
    """
    Calculate perturbations from the non-sphericity of the Earth geopotential.

    Parameters
    ----------
    utc : float
        Coordinated Universal Time.
    satPos : array_like
        Array of shape (3, 1) containing satellite coordinates.
    sunPos : array_like
        Array of shape (3, 1) containing Sun coordinates.
    moonPos : array_like
        Array of shape (3, 1) containing Moon coordinates.
    rot : array_like
        Array of shape(3, 3) containing transition matrix.
    mjd : int
        Input Modified Julian date.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    f : array_like
        Array of shape (3, 1) containing perturbations from the non-sphericity of the Earth geopotential.

    """
    CHarm, SHarm = initHarms(mjd, consts)

    satR = np.linalg.norm(satPos)
    aer = consts.EarthRadius / satR

    latSin, latCos, lonSin, lonCos = getSphericals(satPos, 12, rot)
    leg, dleg = legndr(12, latSin)

    fharm = np.zeros((3, 12))

    dC, dS = getTidalHarmVar(utc, sunPos, moonPos, rot, consts)
    C = CHarm
    S = SHarm

    C[0:4, 0:4] = C[0:4, 0:4] + dC
    S[0:4, 0:4] = S[0:4, 0:4] + dS

    rlond = 1 / latCos * (rot[1, :] * lonCos[1] - rot[0, :] * lonSin[1])
    rsind = rot[2, :] - satPos / satR * latSin

    for i in range(1, 12):
        p1 = np.zeros(3)
        p2 = np.zeros(3)
        for j in range(i+2):
            p11 = S[i][j] * lonCos[j] - C[i][j] * lonSin[j]
            p12 = leg[i][j] * p11
            p13 = j * p12
            p1 = p1 + rlond * p13
            p21 = C[i][j] * lonCos[j] + S[i][j] * lonSin[j]
            p22 = rsind * dleg[i][j]
            p23 = -satPos / satR * (i+2) * leg[i][j]
            p2 = p2 + (p22 + p23) * p21
        fharm[:, i] = aer**(i+1) / satR**2 * (p1 + p2)

    f = np.sum(fharm, 1)

    return f


def angvec(a, b, c):
    """
    Calculate angle between vectors A and B with normal C.

    Parameters
    ----------
    a : array_like
        Input vector A.
    b : array_like
        Input vector B.
    c : array_like
        Normal.

    Returns
    -------
    d : float
        Angle between vectors A and B with normal C.

    """
    au = a / norm(a)
    bu = b / norm(b)

    cosAB = au.transpose().dot(bu)

    xvec = np.cross(au, bu)
    tmp = xvec.transpose().dot(c)
    if tmp >= 0:
        sinAB = norm(xvec)
    else:
        sinAB = -norm(xvec)

    d = atan2(sinAB, cosAB)
    return d


def getLambda(rs, rb, d):
    """
    Calculate eclipse factor for certain planet.

    Parameters
    ----------
    rs : float
        Apparent radius of the Sun.
    rb : float
        Apparent radius of eclipsing body.
    d : float
        Distance between sun and body centers.

    Returns
    -------
    lmbd : float
        Eclipse factor that is a number between 0.0 and 1.0 (0.0 - full eclipse, 1.0 - no eclipse).

    """
    if rs + rb <= d:
        # No eclipse
        lmbd = 1.0
    elif rb - rs >= d:
        # Full eclipse
        lmbd = 0.0
    else:
        # Partial eclipse
        if rs - rb >= d:
            # Eclipsing body lies within solar disk - fraction of solar disk is blocked
            lmbd = (rs * rs - rb * rb) / (rs * rs)
        else:
            ang1 = 2 * acos((rs * rs - rb * rb + d * d) / (2 * rs * d))
            ang2 = 2 * acos((rb * rb - rs * rs + d * d) / (2 * rb * d))

            s1 = (rs * rs * (ang1 - sin(ang1))) / 2
            s2 = (rb * rb * (ang2 - sin(ang2))) / 2

            # Area of intersection of the Sun and eclipsing body
            s = s1 + s2

            lmbd = (pi * rs * rs - s) / (pi * rs * rs)

    return lmbd


def shadow(satPos, sunPos, moonPos, consts):
    """
    Calculate eclipse factor.

    Parameters
    ----------
    satPos : array_like
        Array of shape (3, 1) containing satellite coordinates.
    sunPos : array_like
        Array of shape (3, 1) containing Sun coordinates.
    moonPos : array_like
        Array of shape (3, 1) containing Moon coordinates.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    lmbd : float
        Eclipse factor that is a number between 0.0 and 1.0 (0.0 - full eclipse, 1.0 - no eclipse).

    """
    # Position of satellite relative to the Sun
    satSunPos = satPos - sunPos

    if norm(satSunPos) < norm(sunPos):
        lmbd = 1.0
        return lmbd

    # Get the unit vector of the satellite relative to the Sun
    uSatSunPos = satSunPos / norm(satSunPos)

    sepp = np.cross(satPos, uSatSunPos)
    rsbx = satPos.transpose().dot(uSatSunPos)

    rs = consts.SunRadius / norm(satSunPos)  # Apparent (from satellite) radius of the Sun
    re = consts.EarthRadius / rsbx  # Apparent (from satellite) radius of the Earth
    d = norm(sepp) / rsbx  # Apparent distance between their centers

    # Check Earth shadowing
    lmbd = getLambda(rs, re, d)
    if lmbd < 1.0:
        # If satellite is eclipsed by the Earth,
        # then it is considered that it is not eclipsed by the Moon.
        return lmbd

    # Position of the Moon relative to the Sun
    moonSunPos = moonPos - sunPos
    # Position of satellite relative to the Moon
    satMoonPos = satPos - moonPos

    if norm(satSunPos) < norm(moonSunPos):
        lmbd = 1.0
        return lmbd

    sepp = np.cross(satMoonPos, uSatSunPos)
    rsbx = satMoonPos.transpose().dot(uSatSunPos)

    rm = consts.MoonRadius / rsbx  # Apparent (from satellite) radius of the Moon
    d = norm(sepp) / rsbx  # Apparent distance between their centers

    # Check Moon shadowing
    lmbd = getLambda(rs, rm, d)
    return lmbd


def solarRadiation(satPos, satVel, sunPos, moonPos, radParams, consts):
    """
    Calculate disturbance from the solar radiation pressure on the satellite.

    Parameters
    ----------
    satPos : array_like
        Array of shape (3, 1) containing satellite coordinates.
    satVel : array_like
        Array of shape (3, 1) containing satellite velocity.
    sunPos : array_like
        Array of shape (3, 1) containing Sun coordinates.
    moonPos : array_like
        Array of shape (3, 1) containing Moon coordinates.
    radParams: array_like
        Array of shape (9, 1) containing radiation pressure parameters.
    consts: class
        Contain constants using for the computations.

    Returns
    -------
    rad : array_like
        Array of shape (3, 1) containing disturbance from the solar radiation pressure on the satellite.

    """
    aunit = 1.495978707e8  # Astronomical unit
    sbmass = 1100.0  # Mass for Block IIR satellite
    zaxis = np.array([0, 0, 1])  # in the Earth-fixed inertial system

    # Satellite coordinates relative to the Sun
    satSunPos = satPos - sunPos

    # Points from the Earth to the satellite; it is the negative of the
    # SV-body Z-axis (zvec) which points to the Earth
    rvec = satPos / norm(satPos)
    zvec = -rvec

    # Points in the direction of the satellite's motion
    vvec = satVel / norm(satVel)

    # Points points from the sun to the satellite and defines the Sun or direct (D-) axis
    svec = satSunPos / norm(satSunPos)

    # Orbit normal in the direction the angular momentum (R x V)
    hvec = np.cross(vvec, zvec)
    hvec /= norm(hvec)

    # SV-body Y-axis, which completes a right-hand-system with xvec and zvec (viz Z = X x Y);
    # it points along the axis of the solar panels
    yvec = np.cross(rvec, svec)
    yvec /= norm(yvec)

    # Complete an orthogonal right-hand-system with D and Y in the Berne models.
    # Springer et al. [1998] derive it from D x Y, which is our -SVEC x -YVEC,
    # or equivalently, SVEC x YVEC in our derivation.
    bvec = np.cross(yvec, svec)
    bvec /= norm(bvec)

    # Ascending node of the satellite in the Earth-centered inertial system
    nvec = np.cross(zaxis, hvec)
    nvec /= norm(nvec)

    # Angle between the satellite and node in the orbital plane, thus it's argument of latitude.
    u = angvec(nvec, rvec, hvec)
    cosu = cos(u)
    sinu = sin(u)

    # Count if the Earth or Moon is partially or wholly obstructing the Sun
    lmbd = shadow(satPos, sunPos, moonPos, consts)

    # Distance factor for the radiation force
    distfct = (aunit / norm(satSunPos))**2
    d0 = 11.15e-8 / sbmass

    part = np.array([lmbd * svec,
                     yvec,
                     bvec,
                     cosu * svec,
                     sinu * svec,
                     cosu * yvec,
                     sinu * yvec,
                     cosu * bvec,
                     sinu * bvec]) * d0 * distfct

    rad = part.transpose().dot(radParams)
    return rad


def legNorm(coef):
    dim = coef.shape
    nc = np.zeros(dim)
    c = np.zeros(dim)

    for i in range(dim[0]):
        for j in range(i+2):
            c[i][j] = sqrt(2*(2*i+3) / np.prod([x for x in range(i-j+2, i+j+2)], dtype=float))
            nc[i][j] = coef[i][j] * c[i][j]
    return nc


def initHarms(mjd, consts):
    mjdMid = mjd + 0.5

    # Compute time in years since J2000
    # use start time rather than IC epoch since g-file not yet read
    dyr2000 = (mjdMid - consts.mjd00)/365.25

    # Zonals from EGM2008
    czhar = np.zeros(12)
    czhar[0] = -484.165143790815e-06 + 11.6e-12 * dyr2000
    czhar[1] = 0.957161207093473e-06 + 4.9e-12 * dyr2000
    czhar[2] = 0.539965866638991e-06 + 4.7e-12 * dyr2000
    czhar[3] = 0.686702913736681e-07
    czhar[4] = -0.149953927978527e-06
    czhar[5] = 0.905120844521618e-07
    czhar[6] = 0.494756003005199e-07
    czhar[7] = 0.280180753216300e-07
    czhar[8] = 0.533304381729473e-07
    czhar[9] = -0.507683787085927e-07
    czhar[10] = 0.364361922614572e-07

    cchar = np.zeros(77)
    cshar = np.zeros(77)

    cchar[1] = 0.243938357328313e-05
    cshar[1] = -0.140027370385934e-05

    casr = 4.84813681109536e-06

    # Get the mean pole at epoch 2000.0 from IERS Conventions 2010 after 2010.0
    xpm = 0.023513 + 0.0076141 * dyr2000
    ypm = 0.358891 - 0.0006287 * dyr2000

    # Convert pole position to radians
    xpm = xpm * casr
    ypm = ypm * casr
    cchar[0] = sqrt(3.0) * xpm * czhar[0] - xpm * cchar[1] + ypm * cshar[1]
    cshar[0] = -sqrt(3.0) * ypm * czhar[0] - ypm * cchar[1] - xpm * cshar[1]

    cchar[2] = 0.203046201047864e-05
    cshar[2] = 0.248200415856872e-06
    cchar[3] = 0.904787894809528e-06
    cshar[3] = -0.619005475177618e-06
    cchar[4] = 0.721321757121568e-06
    cshar[4] = 0.141434926192941e-05
    cchar[5] = -0.536157389388867e-06
    cshar[5] = -0.473567346518086e-06
    cchar[6] = 0.350501623962649e-06
    cshar[6] = 0.662480026275829e-06
    cchar[7] = 0.990856766672321e-06
    cshar[7] = -0.200928369177e-06
    cchar[8] = -0.188560802735e-06
    cshar[8] = 0.308853169333e-06
    cchar[9] = -0.62921192304252e-07
    cshar[9] = -0.943698073395769e-07
    cchar[10] = 0.652078043176164e-06
    cshar[10] = -0.323353192540522e-06
    cchar[11] = -0.451847152328843e-06
    cshar[11] = -0.214955408306046e-06
    cchar[12] = -0.295328761175629e-06
    cshar[12] = 0.498070550102351e-07
    cchar[13] = 0.174811795496002e-06
    cshar[13] = -0.669379935180165e-06
    cchar[14] = -0.759210081892527e-07
    cshar[14] = 0.265122593213647e-07
    cchar[15] = 0.486488924604690e-07
    cshar[15] = -0.373789324523752e-06
    cchar[16] = 0.572451611175653e-07
    cshar[16] = 0.895201130010730e-08
    cchar[17] = -0.860237937191611e-07
    cshar[17] = -0.471425573429095e-06
    cchar[18] = -0.267166423703038e-06
    cshar[18] = -0.536493151500206e-06
    cchar[19] = 0.947068749756882e-08
    cshar[19] = -0.237382353351005e-06
    cchar[20] = 0.280887555776673e-06
    cshar[20] = 0.951259362869275e-07
    cchar[21] = 0.330407993702235e-06
    cshar[21] = 0.929969290624092e-07
    cchar[22] = 0.250458409225729e-06
    cshar[22] = -0.217118287729610e-06
    cchar[23] = -0.274993935591631e-06
    cshar[23] = -0.124058403514343e-06
    cchar[24] = 0.164773255934658e-08
    cshar[24] = 0.179281782751438e-07
    cchar[25] = -0.358798423464889e-06
    cshar[25] = 0.151798257443669e-06
    cchar[26] = 0.150746472872675e-08
    cshar[26] = 0.241068767286303e-07
    cchar[27] = 0.231607991248329e-07
    cshar[27] = 0.588974540927606e-07
    cchar[28] = 0.800143604736599e-07
    cshar[28] = 0.652805043667369e-07
    cchar[29] = -0.193745381715290e-07
    cshar[29] = -0.859639339125694e-07
    cchar[30] = -0.244360480007096e-06
    cshar[30] = 0.698072508472777e-07
    cchar[31] = -0.257011477267991e-07
    cshar[31] = 0.892034891745881e-07
    cchar[32] = -0.659648680031408e-07
    cshar[32] = 0.308946730783065e-06
    cchar[33] = 0.672569751771483e-07
    cshar[33] = 0.748686063738231e-07
    cchar[34] = -0.124022771917136e-06
    cshar[34] = 0.120551889384997e-06
    cchar[35] = 0.142151377236084e-06
    cshar[35] = 0.214004665077510e-07
    cchar[36] = 0.214144381199757e-07
    cshar[36] = -0.316984195352417e-07
    cchar[37] = -0.160612356882835e-06
    cshar[37] = -0.742658786809216e-07
    cchar[38] = -0.936529556592536e-08
    cshar[38] = 0.199026740710063e-07
    cchar[39] = -0.163134050605937e-07
    cshar[39] = -0.540394840426217e-07
    cchar[40] = 0.627879491161446e-07
    cshar[40] = 0.222962377434615e-06
    cchar[41] = -0.117983924385618e-06
    cshar[41] = -0.969222126840068e-07
    cchar[42] = 0.188136188986452e-06
    cshar[42] = -0.300538974811744e-08
    cchar[43] = -0.475568433357652e-07
    cshar[43] = 0.968804214389955e-07
    cchar[44] = 0.837623112620412e-07
    cshar[44] = -0.131092332261065e-06
    cchar[45] = -0.939894766092874e-07
    cshar[45] = -0.512746772537482e-07
    cchar[46] = -0.700709997317429e-08
    cshar[46] = -0.154139929404373e-06
    cchar[47] = -0.844715388074630e-07
    cshar[47] = -0.790255527979406e-07
    cchar[48] = -0.492894049964295e-07
    cshar[48] = -0.506137282060864e-07
    cchar[49] = -0.375849022022301e-07
    cshar[49] = -0.797688616388143e-07
    cchar[50] = 0.82620928652347e-08
    cshar[50] = -0.304903703914366e-08
    cchar[51] = 0.405981624580941e-07
    cshar[51] = -0.917138622482163e-07
    cchar[52] = 0.125376631604340e-06
    cshar[52] = -0.379436584841270e-07
    cchar[53] = 0.100435991936118e-06
    cshar[53] = -0.238596204211893e-07
    cchar[54] = 0.156127678638183e-07
    cshar[54] = -0.271235374123689e-07
    cchar[55] = 0.201135250154855e-07
    cshar[55] = -0.990003954905590e-07
    cchar[56] = -0.305773531606647e-07
    cshar[56] = -0.148835345047152e-06
    cchar[57] = -0.379499015091407e-07
    cshar[57] = -0.637669897493018e-07
    cchar[58] = 0.374192407050580e-07
    cshar[58] = 0.495908160271967e-07
    cchar[59] = -0.156429128694775e-08
    cshar[59] = 0.342735099884706e-07
    cchar[60] = 0.465461661449953e-08
    cshar[60] = -0.898252194924903e-07
    cchar[61] = -0.630174049861897e-08
    cshar[61] = 0.245446551115189e-07
    cchar[62] = -0.310727993686101e-07
    cshar[62] = 0.420682585407293e-07
    cchar[63] = -0.522444922089646e-07
    cshar[63] = -0.184216383163730e-07
    cchar[64] = 0.462340571475799e-07
    cshar[64] = -0.696711251523700e-07
    cchar[65] = -0.535856270449833e-07
    cshar[65] = -0.431656037232084e-07
    cchar[66] = 0.142665936828290e-07
    cshar[66] = 0.310937162901519e-07
    cchar[67] = 0.396211271409354e-07
    cshar[67] = 0.250622628960907e-07
    cchar[68] = -0.677284618097416e-07
    cshar[68] = 0.383823469584472e-08
    cchar[69] = 0.308775410911475e-07
    cshar[69] = 0.759066416791107e-08
    cchar[70] = 0.313421100991039e-08
    cshar[70] = 0.389801868153392e-07
    cchar[71] = -0.190517957483100e-07
    cshar[71] = 0.357268620672699e-07
    cchar[72] = -0.258866871220994e-07
    cshar[72] = 0.169362538600173e-07
    cchar[73] = 0.419147664170774e-07
    cshar[73] = 0.249625636010847e-07
    cchar[74] = -0.619955079880774e-08
    cshar[74] = 0.309398171578482e-07
    cchar[75] = 0.113644952089825e-07
    cshar[75] = -0.638551119140755e-08
    cchar[76] = -0.242377235648074e-08
    cshar[76] = -0.110993698692881e-07

    CHarm = np.zeros((12, 13))
    SHarm = np.zeros((12, 13))

    k = 0
    for i in range(1, 12):
        CHarm[i][0] = czhar[i-1]
        for j in range(1, i+2):
            CHarm[i][j] = cchar[k]
            SHarm[i][j] = cshar[k]
            k += 1

    CHarm = legNorm(CHarm)
    SHarm = legNorm(SHarm)
    CHarm[:, 0] /= sqrt(2)

    return CHarm, SHarm
