import orekit

from math import radians
import matplotlib.pyplot as plt

orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()


from org.orekit.errors import OrekitException
from org.orekit.frames import Frame
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit
from org.orekit.orbits import Orbit
from org.orekit.orbits import PositionAngle
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.data import DataProvidersManager
from org.orekit.data import ZipJarCrawler
from org.orekit.time import AbsoluteDate
from org.orekit.time import TimeScalesFactory

a = 24396159.0    # semi major axis in meters
e = 0.72831215  # eccentricity
i = radians(7.0) # inclination
omega = radians(180.0) # perigee argument
raan = radians(261) #right ascension of ascending node
lM = 0.0 # mean anomaly

# Set inertial frame
inertialFrame = FramesFactory.getEME2000()

# Initial date in UTC time scale
utc = TimeScalesFactory.getUTC()
initialDate = AbsoluteDate(2017, 1, 1, 23, 30, 00.000, utc)

# Setup orbit propagator
#gravitation coefficient
mu = 3.986004415e+14

# Orbit construction as Keplerian
initialOrbit = KeplerianOrbit(a, e, i, omega, raan, lM, PositionAngle.MEAN, inertialFrame, initialDate, mu)

# Simple extrapolation with Keplerian motion
kepler = KeplerianPropagator(initialOrbit)

# Set the propagator to slave mode (could be omitted as it is the default mode)
kepler.setSlaveMode()

# Setup propagation time
# Overall duration in seconds for extrapolation
duration = 24*60.0**2

# Stop date
finalDate = AbsoluteDate(initialDate, duration, utc)

# Step duration in seconds
stepT = 30.0

# Perform propagation
#Extrapolation loop
cpt = 1.0
extrapDate = initialDate

px, py = [], []

while extrapDate.compareTo(finalDate) <= 0:
    currentState = kepler.propagate(extrapDate)
    print('step {}: time {} {}\n'.format(cpt, currentState.getDate(), currentState.getOrbit()))
    coord = currentState.getPVCoordinates().getPosition()
    px.append(coord.getX())
    py.append(coord.getY())
    # P[:,cpt]=[coord.getX coord.getY coord.getZ]
    extrapDate = AbsoluteDate(extrapDate, stepT, utc)
    cpt += 1

plt.plot(px, py)
plt.show()