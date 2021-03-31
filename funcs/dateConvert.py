import numpy as np


def cd2mjd(year, month, day):
    """
    Convert a calendar date to a modified Julian date.

    Parameters
    ----------
    year : int
        Input Gregorian year.
    month : int
        Input Gregorian month.
    day : int
        Input Gregorian day.

    Returns
    -------
    mjd : int
        Modified Julian date.
    """
    a = np.fix((14 - month) / 12)
    y = year + 4800 - a
    m = month + 12*a - 3
    jd = day + np.fix((153*m + 2) / 5) + 365 * y + np.fix(y / 4) - np.fix(y / 100) + np.fix(y / 400) - 32045
    mjd = jd - 2400001

    return int(mjd)


def mjd2gpsw(mjd):
    """
    Convert a modified Julian date to GPS week and day of the week.

    Parameters
    ----------
    mjd : int
        Input Modified Julian date.

    Returns
    -------
    week : int
        GPS week.
    day : int
        GPS day of the week.
    """
    gpsEpochMjd = 44244
    week = np.fix((mjd - gpsEpochMjd) / 7)
    day = mjd - gpsEpochMjd - week * 7
    return week, day
