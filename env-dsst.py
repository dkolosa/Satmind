import orekit
from math import radians, degrees
import os, sys

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
from org.orekit.time import AbsoluteDate
from org.orekit.time import TimeScalesFactory
from org.hipparchus.ode import AbstractIntegrator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.geometry.euclidean.threed import Vector3D

from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation.sampling import OrekitFixedStepHandler
from org.orekit.propagation.events import EventDetector
from org.orekit.propagation.events.handlers import EventHandler
from org.orekit.propagation.events import PositionAngleDetector
from org.orekit.propagation.events import AbstractDetector
from org.orekit.orbits import OrbitType
from org.orekit.orbits import PositionAngle
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity import ThirdBodyAttraction
from org.orekit.forces.radiation import SolarRadiationPressure
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient
from org.orekit.forces.radiation import RadiationSensitive
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.utils import IERSConventions

from org.orekit.utils import Constants

from org.orekit.forces.maneuvers import ConstantThrustManeuver

from org.orekit.python import PythonEventHandler, PythonOrekitFixedStepHandler


def main():

    # Initial date in UTC time scale
    utc = TimeScalesFactory.getUTC()
    initial_date = AbsoluteDate(2018, 7, 9, 23, 30, 00.000, utc)

    # The duration in hours
    duration = 24.0 * 60 **2

    # Earth rotation rate
    rotation_rate = Constants.WGS84_EARTH_ANGULAR_VELOCITY

    # Create the inital orbit of the spacecraft
    orbit = createOrbit(initial_date)

    # spacecraft mass
    mass = 1000.0

    #create the numerical propagator
    prop = createPropagator(orbit, mass)

    setForceModel(prop, initial_date)


    initial_state = SpacecraftState(orbit)


    # prop.setEphemerisMode()


    output_step = 10.
    # numEphemeris = numProp.getGeneratedEphemeris()
    handler = OutputHandler()
    prop.setMasterMode(output_step, handler)

    final_date = prop.propagate(initial_date.shiftedBy(duration))


def createOrbit(initial_date):
    """ Crate the initial orbit using Keplarian elements"""

    a = 24396159.0  # semi major axis (m)
    e = 0.720  # eccentricity
    i = radians(10.0)  # inclination
    omega = radians(50.0)  # perigee argument
    raan = radians(150)  # right ascension of ascending node
    lM = 0.0  # mean anomaly

    mu = 3.986004415e+14

    # Set inertial frame
    inertialFrame = FramesFactory.getEME2000()

    orbit = KeplerianOrbit(a, e, i, omega, raan, lM, PositionAngle.MEAN, inertialFrame, initial_date, mu)

    return orbit


def createPropagator(orbit, mass):
    """ Set up the propagator to be used"""


    tol = NumericalPropagator.tolerances(1.0, orbit, orbit.getType())
    minStep = 1.e-3
    maxStep = 1.e+3

    integrator = DormandPrince853Integrator(minStep, maxStep, 1e-5, 1e-10)
    integrator.setInitialStepSize(100.0)

    numProp = NumericalPropagator(integrator)
    numProp.setInitialState(SpacecraftState(orbit, mass))
    return numProp


def setForceModel(numProp, initial_date):
    """ Set up the force model that will be used"""

# force model gravity field
    provider = GravityFieldFactory.getNormalizedProvider(10, 10)
    holmesFeatherstone = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), provider)

    numProp.addForceModel(holmesFeatherstone)

    duration = 1000.0
    isp = 120.0
    thrust = 100.0
    direction = Vector3D.PLUS_I

    thrust = ConstantThrustManeuver(initial_date, duration, thrust, isp, direction)

    numProp.addForceModel(thrust)


class OutputHandler(PythonOrekitFixedStepHandler):

    def init(selfself, s0, t):
        print('Orbital Elements')

    def handleStep(self, currentState, isLast):
        o = OrbitType.KEPLERIAN.convertType(currentState.getOrbit())
        print(o.getDate())
        print('a:{:5.3f}, e:{:5.3f}, i:{:5.3f}, theta:{:5.3f}'.format(o.getA(), o.getE(),
                                                                      degrees(o.getI()), degrees(o.getLv())))
        if isLast:
            print('this was the last step ')


class DateHandler(PythonEventHandler):

    def eventOccured(self, sc_state, T, detector):
        # Call the RL algorithm to produce the thrust
        thrust = q-network(sc_state)
        
        prog.addForceModel(thrust)
        return EventHandler.Action.CONTINUE

    def resetState(self, detector, oldstate):
        return oldstate


if __name__ == '__main__':
    main()