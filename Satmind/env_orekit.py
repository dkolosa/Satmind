
import orekit
from math import radians, degrees
import os, sys, datetime
import matplotlib.pyplot as plt
import numpy as np

orekit.initVM()

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
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.forces.gravity import NewtonianAttraction
from org.orekit.utils import Constants

from java.util import Arrays
from orekit import JArray_double

setup_orekit_curdir()

FUEL_MASS = "Fuel Mass"

UTC = TimeScalesFactory.getUTC()
DIRECTION = Vector3D.PLUS_J
inertial_frame = FramesFactory.getEME2000()
attitude = LofOffset(inertial_frame, LOFType.LVLH)
MU = Constants.EGM96_EARTH_MU

class OrekitEnv:
    """
    This class uses Orekit to create an environment to propagate a satellite
    """

    def __init__(self, state, state_targ, date, duration, mass, fuel_mass, stepT):
        """
        initializes the orekit VM and included libraries
        Params:
        _prop: The propagation object
        _initial_date: The initial start date of the propagation
        _orbit: The orbit type (Keplerian, Circular, etc...)
        _currentDate: The current date during propagation
        _currentOrbit: The current orbit paramenters
        _px: spacecraft position in the x-direction
        _py: spacecraft position in the y-direction
        _sc_fuel: The spacecraft with fuel accounted for
        _extrap_Date: changing date during propagation state
        _sc_state: The spacecraft without fuel
        """

        self._prop = None
        self._initial_date = None
        self._orbit = None
        self._currentDate = None
        self._currentOrbit = None
        self._px = []
        self._py = []
        self._sc_fuel = None
        self._extrap_Date = None
        self._targetOrbit = None

        self.set_date(date)
        self._extrap_Date = self._initial_date
        self.create_orbit(state, self._initial_date, target=False)
        self.set_spacecraft(mass, fuel_mass)
        self.create_Propagator()
        self.setForceModel()
        self.final_date = self._initial_date.shiftedBy(duration)
        self.create_orbit(state_targ, self.final_date, target=True)

        self.stepT = stepT

    def set_date(self, date=None, absolute_date=None, step=0):
        """
        Set up the date for an orekit secnario
        :param date: list [year, month, day, hour, minute, second] (optional)
        :param absolute_date: a orekit Absolute Date object (optional)
        :param step: seconds to shift the date by (int)
        :return:
        """
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

    def create_orbit(self, state, date, target=False):
        """
         Crate the initial orbit using Keplarian elements
        :param state: a state list [a, e, i, omega, raan, lM]
        :param date: A date given as an orekit absolute date object
        :param target: a target orbit list [a, e, i, omega, raan, lM]
        :return:
        """
        a, e, i, omega, raan, lM = state

        # Set inertial frame
        inertialFrame = FramesFactory.getEME2000()
        set_orbit = KeplerianOrbit(a, e, i, omega, raan, lM, PositionAngle.MEAN, inertialFrame, date, MU)

        if target:
            self._targetOrbit = set_orbit
        else:
            self._currentOrbit = set_orbit
            self._orbit = set_orbit

    def set_spacecraft(self, mass, fuel_mass):
        """
        Add the fuel mass to the spacecraft
        :param mass: dry mass of spacecraft (kg, flaot)
        :param fuel_mass:
        :return:
        """
        sc_state = SpacecraftState(self._orbit, mass)
        self._sc_fuel = sc_state.addAdditionalState (FUEL_MASS, fuel_mass)

    def create_Propagator(self, prop_master_mode=False):
        """
        Creates and initializes the propagator
        :param prop_master_mode: Set propagator to slave of master mode (slave default)
        :return:
        """
        # tol = NumericalPropagator.tolerances(1.0, self._orbit, self._orbit.getType())
        minStep = 1.e-3
        maxStep = 1.e+3

        position_tolerance = 10.0
        # tolerances = NumericalPropagator.tolerances(position_tolerance, self._orbit, self._orbit.getType())
        # abs_tolerance = JArray_double.cast_(tolerances[0])
        # rel_telerance = JArray_double.cast_(tolerances[1])

        integrator = DormandPrince853Integrator(minStep, maxStep, 1e-5, 1e-10)
        # integrator = DormandPrince853Integrator(minStep, maxStep, abs_tolerance, rel_telerance)

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
        # self._prop.setAttitudeProvider(attitude)

    def render_plots(self):
        """
        Renders the x-y plots of the spacecraft trajectory
        :return:
        """
        plt.plot(np.array(self._px) / 1000, np.array(self._py) / 1000)
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        # earth = Circle(xy=(0,0), radius=6371.0)
        # plt.figimage(earth)
        plt.show()

    def setForceModel(self):
        """
        Set up environment force models
        """
        # force model gravity field
        # provider = GravityFieldFactory.getNormalizedProvider(10, 10)
        # holmesFeatherstone = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True),
        #                                                        provider)
        # self._prop.addForceModel(holmesFeatherstone)
        earth = NewtonianAttraction(MU)
        self._prop.addForceModel(earth)


    def reset(self):
        """
        Resets the orekit enviornment
        :return:
        """
        self._currentDate = self._initial_date
        self._currentOrbit = self._orbit
        self._extrap_Date = self._initial_date
        self._prop = None
        self.create_Propagator()
        self.setForceModel()
        self.set_spacecraft(1000.0, 500.0)
        self._px = []
        self._py = []

        a = self._currentOrbit.getA()
        lm = self._currentOrbit.getLM()
        adot = self._currentOrbit.getADot()
        lmdot = self._currentOrbit.getLMDot()
        state = [a, 0]

        return np.array(state, ndmin=1)


    def getTotalMass(self):
        """
        Get the total mass of the spacecraft
        :return: dry mass + fuel mass (kg)
        """
        return self._sc_fuel.getAdditionalState(FUEL_MASS)[0] + self._sc_fuel.getMass()

    def step(self, thrust_mag):
        """
        Take a propagation step
        :param thrust_mag: Thrust magnitude (Newtons, float)
        :param stepT: duration of propagation and thrust magnitude (seconds, int)
        :return: spacecraft state (np.array), reward value (float), done (bbol)
        """
        # Keep track of fuel, thrust, position, date
        done = False
        reward = 0
        isp = 1200.0
        # start date, duration, thrust, isp, direction
        thrust = ConstantThrustManeuver(self._extrap_Date, self.stepT, thrust_mag, isp, attitude, DIRECTION)
        self._prop.addForceModel(thrust)
        currentState = self._prop.propagate(self._extrap_Date.shiftedBy(self.stepT))
        # print('step {}: time {} {}\n'.format(cpt, currentState.getDate(), currentState.getOrbit()))
        self._currentDate = currentState.getDate()
        self._extrap_Date = self._currentDate
        self._currentOrbit = currentState.getOrbit()
        coord = currentState.getPVCoordinates().getPosition()
        # Calculate the fuel used and update spacecraft fuel mass
        self._sc_fuel = self._sc_fuel.addAdditionalState(FUEL_MASS, self._sc_fuel.getAdditionalState(FUEL_MASS)[0]
                                                         + thrust.getFlowRate() * self.stepT)
        self._px.append(coord.getX())
        self._py.append(coord.getY())

        a = self._currentOrbit.getA()
        e = self._currentOrbit.getE()
        E = self._currentOrbit.getLE()

        o = OrbitType.KEPLERIAN.convertType(currentState.getOrbit())

        # lm = self._currentOrbit.getLM() / self._targetOrbit.getLM()
        adot = 2*np.sqrt(a/MU) * (thrust_mag/self.getTotalMass()) * (a*np.sqrt(1-e**2))/(1-e*np.cos(o.getLE()))
        # lmdot = self._currentOrbit.getLMDot()
        state = [a, adot]

        reward, done = self.dist_reward(np.array(state))
        if reward == 100:
            done = True

        return np.array(state), reward, done

    def dist_reward(self, state):
        """
        Computes the reward based on the state of the agent
        :param state: Spacecraft state
        :return: reward value (float)
        """

        target_a = self._targetOrbit.getA()
        initial_a = self._orbit.getA()
        current_a = self._currentOrbit.getA()

        # reward function is between -1 and 1
        dist = target_a - current_a
        dist_org = target_a - initial_a

        if self._sc_fuel.getAdditionalState(FUEL_MASS)[0] <= 0:
            print("Ran out of fuel")
            done = True
            reward = -100

        if dist < -100:
            reward = -100
            done = True
            # print('Overshoot')
        elif -100 <= dist <= 100:
            if abs(self._currentOrbit.getE() - self._targetOrbit.getE()) <= 0.1:
                reward = 100
                done = True
        else:
            reward = (current_a/target_a)*(self._currentOrbit.getE()/self._targetOrbit.getE())
            done = False
            # reward = 1 - dist**.4
        return reward, done


class OutputHandler(PythonOrekitFixedStepHandler):
    """
    Implements a custom handler for every value
    """
    def init(self, s0, t):
        """
        Initilization at every prpagation step
        :param s0: initial state (spacecraft State)
        :param t: initial time (int)
        :return:
        """
        print('Orbital Elements:')

    def handleStep(self, currentState, isLast):
        """
        Perform a step at every propagation step
        :param currentState: current spacecraft state (Spacecraft state)
        :param isLast: last step in the propagation (bool)
        :return:
        """
        o = OrbitType.KEPLERIAN.convertType(currentState.getOrbit())
        print(o.getDate())
        print('a:{:5.3f}, e:{:5.3f}, i:{:5.3f}, theta:{:5.3f}'.format(o.getA(), o.getE(),
                                                                      degrees(o.getI()), degrees(o.getLv())))
        if isLast:
            print('this was the last step ')


def main():

    year, month, day, hr, minute, sec = 2018, 8, 1, 9, 30, 00.00
    date = [year, month, day, hr, minute, sec]

    mass = 1000.0
    fuel_mass = 500.0
    duration = 24.0 * 60.0 ** 2 * 1

    # set the sc initial state
    a = 5_500.0e3  # semi major axis (m)
    e = 0.01  # eccentricity
    i = radians(0.001)  # inclination
    omega = radians(0.01)  # perigee argument
    raan = radians(0.01)  # right ascension of ascending node
    lM = 0.0  # mean anomaly
    state = [a, e, i, omega, raan, lM]

    # target state
    a_targ = 10_000.0e3
    e_targ = e
    i_targ = i
    omega_targ = omega
    raan_targ = raan
    lM_targ = lM
    state_targ = [a_targ, e_targ, i_targ, omega_targ, raan_targ, lM_targ]
    stepT = 100.0


    env = OrekitEnv(state, state_targ, date, duration, mass, fuel_mass, stepT)

    thrust_mag = 1.0
    isp = 1200.0

    a, lv = [], []

    while env._extrap_Date.compareTo(env.final_date) <= 0:
        position, r, done, _ = env.step(thrust_mag)
        a.append(position[0])
        lv.append(position[1])

    print("done")
    env.render_plots()


if __name__ == '__main__':
    main()