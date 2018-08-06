
import orekit
from math import radians, degrees
import os, sys, datetime
import matplotlib.pyplot as plt

import numpy as np

orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir

setup_orekit_curdir()

from org.orekit.errors import OrekitException
from org.orekit.frames import FramesFactory, Frame
from org.orekit.orbits import KeplerianOrbit, Orbit
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.geometry.euclidean.threed import Vector3D

from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation.sampling import OrekitFixedStepHandler
from org.orekit.orbits import OrbitType, PositionAngle
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel

from org.orekit.utils import IERSConventions, Constants

from org.orekit.forces.maneuvers import ConstantThrustManeuver

from org.orekit.frames import LOFType
from org.orekit.attitudes import LofOffset

from org.orekit.python import PythonEventHandler, PythonOrekitFixedStepHandler

FUEL_MASS = "Fuel Mass"

UTC = TimeScalesFactory.getUTC()


class OrekitEnv:
    """ This class uses Orekit to create an environment to propagate a satellite

        Params:
        _prop: The propagation object
        _initial_date: The initial start date of the propagation
        _orbit: The orbit type (Keplerian, Circular, etc...)
        _currentDate: The current date during propagation
        _currentorbit: The current orbit paramenters
        _px: spacecraft position in the x-direction
        _py: spacecraft position in the y-direction
        _sc_fuel: The spacecraft with fuel accounted for
        _extrap_Date: changing date during propagation state
        _sc_state: The spacecraft without fuel

    """



    def __init__(self):
        """ initializes the orekit VM and included libraries"""

        self._prop = None
        self._initial_date = None
        self._orbit = None
        self._currentDate = None
        self._currentOrbit = None
        self._px = []
        self._py = []
        self._sc_fuel = None
        self._extrap_Date = None

    def set_date(self, date=None, absolute_date=None, step=0):
        if date != None:
            year, month, day, hour, minute, sec = date
            self._initial_date = AbsoluteDate(year, month, day, hour, minute, sec, UTC)
        elif absolute_date != None and step != 0:
            self._extrap_Date = AbsoluteDate(absolute_date, step, UTC)
        else:
            # no argument given, use current date and time
            now = datetime.datetime.now()
            year, month, day, hour, minute, sec = now.year, now.month, now.day, now.hour, now.minute, float(now.second)
            self._initial_date = AbsoluteDate(year, month, day, hour, minute, sec, UTC)

    def shift_date(self, step):
        self._extrap_Date = AbsoluteDate(self._extrap_Date, step, UTC)
        pass

    def create_orbit(self, state):
        """ Crate the initial orbit using Keplarian elements"""
        a, e, i, omega, raan, lM = state

        mu = 3.986004415e+14

        # Set inertial frame
        inertialFrame = FramesFactory.getEME2000()

        self._orbit = KeplerianOrbit(a, e, i, omega, raan, lM, PositionAngle.MEAN, inertialFrame, self._initial_date, mu)

        self._currentOrbit = self._orbit

        # return orbit

    def set_spacecraft(self, mass, fuel_mass):
        sc_state = SpacecraftState(self._orbit, mass)
        self._sc_fuel = sc_state.addAdditionalState(FUEL_MASS, fuel_mass)



    def create_Propagator(self, prop_master_mode=False):
        """ Set up the propagator to be used"""

        tol = NumericalPropagator.tolerances(1.0, self._orbit, self._orbit.getType())
        minStep = 1.e-3
        maxStep = 1.e+3

        integrator = DormandPrince853Integrator(minStep, maxStep, 1e-5, 1e-10)
        integrator.setInitialStepSize(100.0)

        numProp = NumericalPropagator(integrator)
        numProp.setInitialState(self._sc_fuel)

        if prop_master_mode:
            output_step = 5.0
            handler = OutputHandler()
            numProp.setMasterMode(output_step, handler)
        else:
            numProp.setSlaveMode()

        self._prop = numProp


    def render_plots(self):
        plt.plot(np.array(self._px) / 1000, np.array(self._py) / 1000)
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        # earth = Circle(xy=(0,0), radius=6371.0)
        # plt.figimage(earth)
        plt.show()



    def setForceModel(self):
        """ Set up environment force models"""

        # force model gravity field
        provider = GravityFieldFactory.getNormalizedProvider(10, 10)
        holmesFeatherstone = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True),
                                                               provider)
        self._prop.addForceModel(holmesFeatherstone)

    def reset(self):
        # TODO reset the state of the sc to the initial state
        self._currentDate = self._initial_date
        self._currentOrbit = self._orbit


    def getTotalMass(self):
        return self._sc_fuel.getAdditionalState(FUEL_MASS)[0] + self._sc_fuel.getMass()

    def step(self, thrust_mag,stepT):
        # TODO makes one propagation step
        # Keep track of fuel, thrust, position, date

        # 5 sec steps
        isp = 1200.0
        direction = Vector3D.PLUS_I
        inertialFrame = FramesFactory.getEME2000()
        attitude = LofOffset(inertialFrame, LOFType.LVLH)

        # start date, duration, thrust, isp, direction
        thrust = ConstantThrustManeuver(self._extrap_Date, stepT, thrust_mag, isp, attitude, direction)
        self._prop.addForceModel(thrust)
        currentState = self._prop.propagate(self._extrap_Date)
        # print('step {}: time {} {}\n'.format(cpt, currentState.getDate(), currentState.getOrbit()))
        self._currentDate = currentState.getDate()
        self._currentOrbit = currentState.getOrbit()
        coord = currentState.getPVCoordinates().getPosition()
        # Calculate the fuel used and update spacecraft fuel mass
        self._sc_fuel = self._sc_fuel.addAdditionalState(FUEL_MASS,
                                                         self._sc_fuel.getAdditionalState(FUEL_MASS)[0] + thrust.getFlowRate() * stepT)
        self._px.append(coord.getX())
        self._py.append(coord.getY())
        if self._sc_fuel.getAdditionalState(FUEL_MASS)[0] <= 0:
            print("Ran out of fuel")
            exit()
        return [coord.getX(), coord.getY()]


class OutputHandler(PythonOrekitFixedStepHandler):
    """ Implements a custom handler for every value """
    def init(self, s0, t):
        print('Orbital Elements:')

    def handleStep(self, currentState, isLast):
        o = OrbitType.KEPLERIAN.convertType(currentState.getOrbit())
        print(o.getDate())
        print('a:{:5.3f}, e:{:5.3f}, i:{:5.3f}, theta:{:5.3f}'.format(o.getA(), o.getE(),
                                                                      degrees(o.getI()), degrees(o.getLv())))
        if isLast:
            print('this was the last step ')


def main():

    env = OrekitEnv()
    year, month, day, hr, minute, sec = 2018, 8, 1, 9, 30, 00.00
    date = [year, month, day, hr, minute, sec]
    env.set_date(date)

    mass = 1000.0
    fuel_mass = 500.0
    duration = 24.0*60.0**2

    # set the sc initial state
    a = 24_396_159.0  # semi major axis (m)
    e = 0.1  # eccentricity
    i = radians(2.0)  # inclination
    omega = radians(2.0)  # perigee argument
    raan = radians(1.0)  # right ascension of ascending node
    lM = 0.0  # mean anomaly
    state = [a, e, i, omega, raan, lM]

    env.create_orbit(state)
    env.set_spacecraft(mass, fuel_mass)
    env.create_Propagator()
    env.setForceModel()

    final_date = env._initial_date.shiftedBy(duration)
    env._extrap_Date = env._initial_date
    stepT = 100.0

    thrust_mag = 30.0
    while env._extrap_Date.compareTo(final_date) <= 0:
        position = env.step(thrust_mag, stepT)
        env.shift_date(stepT)
        # env._extrap_Date = AbsoluteDate(env._extrap_Date, stepT, UTC)

    print("done")

    print(env.getTotalMass())
    env.render_plots()


if __name__ == '__main__':
    main()