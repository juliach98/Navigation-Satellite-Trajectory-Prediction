import numpy as np
from funcs.dateConvert import cd2mjd


class InitializeProblem(object):
    """ Initialization of variables describing the problem. """

    def __init__(self, year, month, day):
        self._year = year                               # Gregorian year
        self._month = month                             # Gregorian month
        self._day = day                                 # Gregorian day
        self._mjd = cd2mjd(year, month, day)            # Modified Julian Date
        self.n = 6                                      # State equation (x(tk+1)) size
        self.m = 3                                      # Observation equation (y(tk+1)) size
        self._timeGrid = np.linspace(3600, 81000, 87)   # Time grid in sec for 1 day (15 min step)
        self._N = self._timeGrid.size                   # Time grid size
        self.radParams = np.zeros(9)                    # Radiation parameters

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, year):
        self._year = year
        self._mjd = cd2mjd(self._year, self._month, self._day)

    @property
    def month(self):
        return self._month

    @month.setter
    def month(self, month):
        self._month = month
        self._mjd = cd2mjd(self._year, self._month, self._day)

    @property
    def day(self):
        return self._day

    @day.setter
    def day(self, day):
        self._day = day
        self._mjd = cd2mjd(self._year, self._month, self._day)

    @property
    def mjd(self):
        return self._mjd

    @property
    def N(self):
        return self._N

    @property
    def timeGrid(self):
        return self._timeGrid

    @timeGrid.setter
    def timeGrid(self, timeGrid):
        self._timeGrid = timeGrid
        self._N = timeGrid.size

