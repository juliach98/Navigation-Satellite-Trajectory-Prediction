# Navigation-Satellite-Trajectory-Prediction
Solving the problem of predicting the satellite trajectory by evaluating the unknown parameters of the solar radiation pressure model based on the application of modern mathematical methods.

# Project description
The quality of the ephemeris-temporal support for Global Navigation Satellite System technologies depends on adequacy of the applied mathematical models describing the orbital motion of navigation satellites. Nowadays, it is possible, with an insignificant error, to take into account the non-sphericity of the geopotential and the gravitational influence of the Moon, the Sun and other planets. However, the situation is different taking into account the perturbations from the solar radiation pressure on the satellite. That is why evaluation of the unknown parameters of the solar radiation pressure model is needed.

Developed software predict trajectory of navigation satellite by the following scheme:
1. Evaluate the unknown parameters of the solar radiation pressure model for the chosen day.
2. Use calculated parameters to make a navigation satellite trajectory prediction for the next day.

To learn more information see: http://www.ict.nsc.ru/jct/getfile.php?id=1969
# How to use?
Choose date(year, month, day) and type

```
python main.py --year YEAR --month MONTH --day DAY
```

Example:

``` 
python main.py --year 2016 --month 7 --day 14
```

## Before using
Install libraries NumPy, SciPy, Pandas, SpiceyPy
```
pip install LIB_NAME
```
and GNSSpy
```
pip install git+https://github.com/GNSSpy-Project/gnsspy
```

## Input data
Input data files are need to be in .sp3 format. .sp3 ephemeris files (also known as "Precise Ephemeris") contain data records for the orbit and clock states for the entire GPS constellation. Project contains .sp3 files for dates from 10.07.2016 to 30.07.2016 in resources directory. To find data for more dates see: https://cddis.nasa.gov/

## Output data
Output data file is in .txt format. It contains predicted trajectory (satellite coordinates each 15 min). Also in terminal it can be seen RMSE(root-mean-square error) which is showing how accurately the prediction was made regarding the known satellite trajectory for that day.
