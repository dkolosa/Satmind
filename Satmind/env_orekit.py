
import orekit
from math import radians, degrees, sqrt, pi
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os, random

orekit.initVM()

from org.orekit.frames import FramesFactory, Frame
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.orbits import KeplerianOrbit, Orbit
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
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

from org.hipparchus.geometry.euclidean.threed import Vector3D
from java.util import Arrays
from orekit import JArray_double

setup_orekit_curdir()

FUEL_MASS = "Fuel Mass"

UTC = TimeScalesFactory.getUTC()
inertial_frame = FramesFactory.getEME2000()
attitude = LofOffset(inertial_frame, LOFType.LVLH)
EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
MU = Constants.WGS84_EARTH_MU


class OrekitEnv:
    """
    This class uses Orekit to create an environment to propagate a satellite
    """

    def __init__(self, state, state_targ, date, duration, mass, stepT):
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

        self.dry_mass = mass[0]
        self.fuel_mass = mass[1]
        self.cuf_fuel_mass = self.fuel_mass
        self.initial_mass = self.dry_mass + self.fuel_mass


        self.px = []
        self.py = []
        self.pz = []
        self.a_orbit = []
        self.ex_orbit = []
        self.ey_orbit = []
        self.hx_orbit = []
        self.hy_orbit = []
        self.lv_orbit = []

        self.adot_orbit = []
        self.exdot_orbit = []
        self.eydot_orbit = []
        self.hxdot_orbit = []
        self.hydot_orbit = []

        self._sc_fuel = None
        self._extrap_Date = None
        self._targetOrbit = None
        self.target_px = []
        self.target_py = []
        self.target_pz = []

        self._orbit_tolerance = {'a': 10000, 'ex': 0.01, 'ey': 0.01, 'hx': 0.001, 'hy': 0.001, 'lv': 0.01}

        self.randomize = True
        self._orbit_randomizer = {'a': 1000.0e3, 'e': 0.02, 'i': 0.02, 'w': 2.0, 'omega': 2.0, 'lv': 5.0}
        self.seed_state = state
        self.seed_target = state_targ
        self.target_hit = False

        self.set_date(date)
        self._extrap_Date = self._initial_date
        self.create_orbit(state, self._initial_date, target=False)
        self.set_spacecraft(self.initial_mass, self.cuf_fuel_mass)
        self.create_Propagator()
        self.setForceModel()
        self.final_date = self._initial_date.shiftedBy(duration)
        self.create_orbit(state_targ, self.final_date, target=True)

        self.stepT = stepT
        self.action_space = 3  # output thrust
        self.observation_space = 10  # states
        self.action_bound = 0.9  # TFC coefficient
        self._isp = 3100.0

        self.r_target_state = np.array(
            [self._targetOrbit.getA(), self._targetOrbit.getEquinoctialEx(), self._targetOrbit.getEquinoctialEy(),
             self._targetOrbit.getHx(), self._targetOrbit.getHy(), self._targetOrbit.getLv()])

        self.r_initial_state = np.array([self._orbit.getA(), self._orbit.getEquinoctialEx(), self._orbit.getEquinoctialEy(),
                                  self._orbit.getHx(), self._orbit.getHy(), self._orbit.getLv()])

        self.r_target_state = self.get_state(self._targetOrbit)
        self.r_initial_state = self.get_state(self._orbit)

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

        a += EARTH_RADIUS
        i = radians(i)
        omega = radians(omega)
        raan = radians(raan)
        lM = radians(lM)

        aDot, eDot, iDot, paDot, rannDot, anomalyDot = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Set inertial frame
        set_orbit = KeplerianOrbit(a, e, i, omega, raan, lM,
                                   aDot, eDot, iDot, paDot, rannDot, anomalyDot,
                                   PositionAngle.TRUE, inertial_frame, date, MU)

        if target:
            self._targetOrbit = set_orbit
        else:
            self._currentOrbit = set_orbit
            self._orbit = set_orbit

    def convert_to_keplerian(self, orbit):

        ko = KeplerianOrbit(orbit)
        # a = ko.getA()
        # e = ko.getE()
        # i = ko.getI()
        # w = ko.getPerigeeArgument()
        # omega = ko.getRightAscensionOfAscendingNode()
        # ta = ko.getTrueAnomaly()
        # E = ko.getEccentricAnomaly()
        #
        # adot = ko.getADot()
        # edot = ko.getEDot()
        # idot = ko.getIDot()
        # wdot = ko.getPerigeeArgumentDot()
        # omegadot = ko.getRightAscensionOfAscendingNodeDot()
        # # tadot = ko.getTrueAnomalyDot()
        # Edot = ko.getEccentricAnomalyDot()
        #
        # state = np.array([a, e, i, w, omega, E, adot, edot, idot, wdot, omegadot, Edot])
        # state = np.nan_to_num(state)

        state = np.array([a, e, i, w, omega, E, adot, edot, idot, wdot, omegadot, Edot])
        state = np.nan_to_num(state)

        return state

    def set_spacecraft(self, mass, fuel_mass):
        """
        Add the fuel mass to the spacecraft
        :param mass: dry mass of spacecraft (kg, flaot)
        :param fuel_mass:
        :return:
        """
        sc_state = SpacecraftState(self._orbit, mass)
        # print(f'{sc_state.getMass()}')
        self._sc_fuel = sc_state.addAdditionalState (FUEL_MASS, fuel_mass)

    def create_Propagator(self, prop_master_mode=False):
        """
        Creates and initializes the propagator
        :param prop_master_mode: Set propagator to slave of master mode (slave default)
        :return:
        """
        # tol = NumericalPropagator.tolerances(1.0, self._orbit, self._orbit.getType())
        minStep = 0.001
        maxStep = 500.0

        position_tolerance = 60.0
        tolerances = NumericalPropagator.tolerances(position_tolerance, self._orbit, self._orbit.getType())
        abs_tolerance = JArray_double.cast_(tolerances[0])
        rel_telerance = JArray_double.cast_(tolerances[1])

        # integrator = DormandPrince853Integrator(minStep, maxStep, 1e-5, 1e-10)
        integrator = DormandPrince853Integrator(minStep, maxStep, abs_tolerance, rel_telerance)

        integrator.setInitialStepSize(10.0)

        numProp = NumericalPropagator(integrator)
        numProp.setInitialState(self._sc_fuel)
        numProp.setMu(MU)
        numProp.setOrbitType(OrbitType.KEPLERIAN)

        if prop_master_mode:
            output_step = 5.0
            handler = OutputHandler()
            numProp.setMasterMode(output_step, handler)
        else:
            numProp.setSlaveMode()

        self._prop = numProp
        self._prop.setAttitudeProvider(attitude)

    def render_plots(self, episode=1, save=True, show=True):
        """
        Renders the x-y plots of the spacecraft trajectory
        :return:
        """
        if 0 <= episode < 10:
            episode = '00'+str(episode)
        elif 10 <= episode < 100:
            episode = '0'+str(episode)
        elif episode >= 100:
            episode = str(episode)

        os.makedirs('results/' + episode, exist_ok=True)
        save_path = 'results/' + episode + '/'

        plt.clf()
        oe_fig = plt.figure(1)
        oe_params = ('sma', 'e_x', 'e_y', 'h_x', 'h_y', 'lv')
        oe = [np.asarray(self.a_orbit)/1e3, self.ex_orbit, self.ey_orbit, self.hx_orbit, self.hy_orbit, self.lv_orbit]
        oe_target = [self.r_target_state[0]/1e3, self.r_target_state[1], self.r_target_state[2], self.r_target_state[3],
                   self.r_target_state[4], self.r_target_state[5]]
        plt.plot(np.asarray(self.px) / 1000, np.asarray(self.py) / 1000, '-b',
                 np.asarray(self.target_px)/1000, np.asarray(self.target_py)/1000, '-r')
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.tight_layout()
        if save:
            plt.savefig(save_path + '2d.pdf')
        if show: plt.show()
        fig = plt.figure(2)
        ax = fig.gca(projection='3d')
        ax.plot(np.asarray(self.px)/1000, np.asarray(self.py)/1000, np.asarray(self.pz)/1000,label='Satellite Trajectory')
        ax.plot(np.asarray(self.target_px)/1000, np.asarray(self.target_py)/1000, np.asarray(self.target_pz)/1000,
                color='red',label='Target trajectory')
        ax.legend()
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_zlim(-1500, 1500)
        if save:
            plt.savefig(save_path + '3d.pdf')
        plt.figure(3)
        for i in range(len(oe_params)):
            plt.subplot(3,2,i+1)
            plt.plot(oe[i])
            plt.scatter(len(oe[i]), oe_target[i], c='red')
            plt.ylabel(oe_params[i])
        plt.tight_layout()
        if save:
            plt.savefig(save_path + 'oe.pdf')
        if show: plt.show()
        ax.clear()

    def setForceModel(self):
        """
        Set up environment force models
        """
        # force model gravity field
        newattr = NewtonianAttraction(MU)
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)  # International Terrestrial Reference Frame, earth fixed

        earth = OneAxisEllipsoid(EARTH_RADIUS,
                                Constants.WGS84_EARTH_FLATTENING,itrf)
        gravityProvider = GravityFieldFactory.getNormalizedProvider(8, 8)
        self._prop.addForceModel(HolmesFeatherstoneAttractionModel(earth.getBodyFrame(), gravityProvider))

        # self._prop.addForceModel(newattr)

    def reset(self):
        """
        Resets the orekit enviornment
        :return:
        """

        self._prop = None
        self._currentDate = None
        self._currentOrbit = None

        # Randomizes the initial orbit
        if self.randomize:
            self._orbit = None
            a_rand = random.uniform(self.seed_state[0]-self._orbit_randomizer['a'], self._orbit_randomizer['a']+ self.seed_state[0])
            e_rand = random.uniform(self.seed_state[1]-self._orbit_randomizer['e'], self._orbit_randomizer['e']+ self.seed_state[1])
            i_rand = random.uniform(self.seed_state[2]-self._orbit_randomizer['i'], self._orbit_randomizer['i']+ self.seed_state[2])
            w_rand = random.uniform(self.seed_state[2]-self._orbit_randomizer['w'], self._orbit_randomizer['w']+ self.seed_state[3])
            omega_rand = random.uniform(self.seed_state[3]-self._orbit_randomizer['omega'], self._orbit_randomizer['omega']+ self.seed_state[4])
            lv_rand = random.uniform(self.seed_state[4]-self._orbit_randomizer['lv'], self._orbit_randomizer['lv']+ self.seed_state[5])
            state = [a_rand, e_rand, i_rand, w_rand, omega_rand, lv_rand]
            self.create_orbit(state, self._initial_date, target=False)
        else:
            self._currentOrbit = self._orbit
            # print(self._orbit)

        self._currentDate = self._initial_date
        self._extrap_Date = self._initial_date

        self.set_spacecraft(self.initial_mass, self.fuel_mass)
        self.cuf_fuel_mass = self.fuel_mass
        self.create_Propagator()
        self.setForceModel()

        self.px = []
        self.py = []
        self.pz = []

        self.a_orbit = []
        self.ex_orbit = []
        self.ey_orbit = []
        self.hx_orbit = []
        self.hy_orbit = []
        self.lv_orbit = []

        self.adot_orbit = []
        self.exdot_orbit = []
        self.eydot_orbit = []
        self.hxdot_orbit = []
        self.hydot_orbit = []

        state = np.array([self._orbit.getA() / self.r_target_state[0],
                          self._orbit.getEquinoctialEx(),
                          self._orbit.getEquinoctialEy(),
                          self._orbit.getHx(),
                          self._orbit.getHy(), 0, 0, 0, 0, 0])
        return state

    @property
    def getTotalMass(self):
        """
        Get the total mass of the spacecraft
        :return: dry mass + fuel mass (kg)
        """
        return self._sc_fuel.getAdditionalState(FUEL_MASS)[0] + self._sc_fuel.getMass()

    def get_state(self, orbit, with_derivatives=True):

        if with_derivatives:
            state = [orbit.getA(), orbit.getEquinoctialEx(), orbit.getEquinoctialEy(),
                     orbit.getHx(), orbit.getHy(), orbit.getLv(),
                     orbit.getADot(), orbit.getEquinoctialExDot(),
                     orbit.getEquinoctialEyDot(),
                     orbit.getHxDot(), orbit.getHyDot(), orbit.getLvDot()]
        else:
            state = [orbit.getA(), orbit.getEquinoctialEx(), orbit.getEquinoctialEy(),
                       orbit.getHx(), orbit.getHy(), orbit.getLv()]

        return state

    def step(self, thrust):
        """
        Take a propagation step
        :param thrust_mag: Thrust magnitude (Newtons, float)
        :return: spacecraft state (np.array), reward value (float), don\
        e (bbol)
        """
        thrust_mag = np.linalg.norm(thrust)
        thrust_dir = thrust / thrust_mag
        DIRECTION = Vector3D(float(thrust_dir[0]), float(thrust_dir[1]), float(thrust_dir[2]))

        if thrust_mag <= 0:
            # DIRECTION = Vector3D.MINUS_J
            thrust_mag = abs(float(thrust_mag))
        else:
            # DIRECTION = Vector3D.PLUS_J
            thrust_mag = float(thrust_mag)

        thrust_force = ConstantThrustManeuver(self._extrap_Date, self.stepT, thrust_mag, self._isp, attitude, DIRECTION)
        # thrust_force.init(SpacecraftState(self._currentOrbit, float(self.mass + self.fuel_mass)), self._extrap_Date)
        self._prop.addForceModel(thrust_force)
        currentState = self._prop.propagate(self._extrap_Date.shiftedBy(self.stepT))
        # print(f'{currentState.getMass()}')
        self.cuf_fuel_mass = currentState.getMass() - self.dry_mass
        self._currentDate = currentState.getDate()
        self._extrap_Date = self._currentDate
        self._currentOrbit = currentState.getOrbit()
        coord = currentState.getPVCoordinates().getPosition()

        self.px.append(coord.getX())
        self.py.append(coord.getY())
        self.pz.append(coord.getZ())
        self.a_orbit.append(currentState.getA())
        self.ex_orbit.append(currentState.getEquinoctialEx())
        self.ey_orbit.append(currentState.getEquinoctialEy())
        self.hx_orbit.append(currentState.getHx())
        self.hy_orbit.append(currentState.getHy())
        self.lv_orbit.append(currentState.getLv())

        reward, done = self.dist_reward(thrust)

        state_1 = [(self._currentOrbit.getA()) / self.r_target_state[0],
                   self._currentOrbit.getEquinoctialEx(), self._currentOrbit.getEquinoctialEy(),
                   self._currentOrbit.getHx(), self._currentOrbit.getHy(),
                   self._currentOrbit.getADot(), self._currentOrbit.getEquinoctialExDot(),
                   self._currentOrbit.getEquinoctialEyDot(),
                   self._currentOrbit.getHxDot(), self._currentOrbit.getHyDot()
                   ]

        # state_1 = [self._currentOrbit.getA() / self.r_target_state[0],
        #            self._currentOrbit.getEquinoctialEx(),
        #            self._currentOrbit.getEquinoctialEy(),
        #            self._currentOrbit.getHx(),
        #            self._currentOrbit.getHy()]

        self.adot_orbit.append(self._currentOrbit.getADot())
        self.exdot_orbit.append(self._currentOrbit.getEquinoctialExDot())
        self.eydot_orbit.append(self._currentOrbit.getEquinoctialEyDot())
        self.hxdot_orbit.append(self._currentOrbit.getHxDot())
        self.hydot_orbit.append(self._currentOrbit.getHyDot())

        return state_1, reward, done

    def dist_reward(self, thrust):
        """
        Computes the reward based on the state of the agent
        :param thrust: Spacecraft thrust
        :return: reward value (float)
        """
        # a, ecc, i, w, omega, E, adot, edot, idot, wdot, omegadot, Edot = state

        done = False

        thrust_mag = np.linalg.norm(thrust)

        state = np.array([self._currentOrbit.getA(), self._currentOrbit.getEquinoctialEx(), self._currentOrbit.getEquinoctialEy(),
                          self._currentOrbit.getHx(), self._currentOrbit.getHy(), self._currentOrbit.getLv()])

        # Inclination change reward
        # reward_a  = abs(self.r_target_state[0] - state[0]) / self.r_target_state[0]
        reward_a = abs(self.r_target_state[0] - state[0]) / state[0]
        reward_ex = abs(self.r_target_state[1] - state[1])
        reward_ey = abs(self.r_target_state[2] - state[2])
        reward_hx = abs(self.r_target_state[3] - state[3])
        reward_hy = abs(self.r_target_state[4] - state[4])
        # print(f'r_a: {reward_a*10}, ex: {reward_ex*10}, ey: {reward_ey*10}, hx: {reward_hx*10}, hy: {reward_hy*10}')
        reward = -(reward_a + reward_hx*10 + reward_hy*10 + reward_ex + reward_ey*10)
        # reward = (1 - reward**.4)

        # if abs(self.r_target_state[0] - state[0]) <= self._orbit_tolerance['a']:
        #     reward = reward_hx + thrust_mag
        #     reward = (1 - reward**.4)*(self.cuf_fuel_mass/self.fuel_mass)
        #
        # if abs(self.r_target_state[3] - state[3]) <= self._orbit_tolerance['hx']:
        #     reward = reward_a + thrust_mag
        #     reward = (1 - reward**.4)*(self.cuf_fuel_mass/self.fuel_mass)


        # Terminal staes

        if abs(self.r_target_state[0] - state[0]) <= self._orbit_tolerance['a'] and \
           abs(self.r_target_state[1] - state[1]) <= self._orbit_tolerance['ex'] and \
           abs(self.r_target_state[2] - state[2]) <= self._orbit_tolerance['ey'] and \
           abs(self.r_target_state[3] - state[3]) <= self._orbit_tolerance['hx'] and \
           abs(self.r_target_state[4] - state[4]) <= self._orbit_tolerance['hy']:
            # self.final_date.durationFrom(self._extrap_Date) <= 360:
            reward += 1000
            done = True
            print('hit')
            self.target_hit = True
            return reward, done

        # if self._sc_fuel.getAdditionalState(FUEL_MASS)[0] <= 0:
        if self.cuf_fuel_mass <= 0:
            print('Ran out of fuel')
            done = True
            reward = -100
            return reward, done

        if self._currentOrbit.getA() < EARTH_RADIUS:
            reward = -100
            done = True
            print('In earth')
            return reward, done

        # if self._currentOrbit.getHx() > self.r_target_state[3]+0.01:
        #     reward = -10
        #     done = True
        #
        # if self._currentOrbit.getHy() < self._orbit.getHy()-0.001:
        #     reward = -10
        #     done = True

        # if self._extrap_Date.compareTo(self.final_date) >= 0:
        #     reward += -10
        #     done = True

        return reward, done

    def render_target(self):

        target_sc = SpacecraftState(self._targetOrbit)

        # Orbit time for one orbit regardless of semi-major axis
        orbit_time = sqrt(4 * pi**2 * self._targetOrbit.getA()**3 / MU) * 1.1
        minStep = 1.e-3
        maxStep = 1.e+3

        integrator = DormandPrince853Integrator(minStep, maxStep, 1e-5, 1e-10)
        integrator.setInitialStepSize(100.0)

        numProp = NumericalPropagator(integrator)
        numProp.setInitialState(target_sc)
        numProp.setMu(MU)
        numProp.setSlaveMode()

        target_prop = numProp
        earth = NewtonianAttraction(MU)
        target_prop.addForceModel(earth)

        target_date = self.final_date.shiftedBy(orbit_time)
        extrapDate = self.final_date
        stepT = 500.0

        # know this is the state for final_date + time for orbit
        while extrapDate.compareTo(target_date) <= 0:
            currentState = target_prop.propagate(extrapDate)
            coord = currentState.getPVCoordinates().getPosition()
            self.target_px.append(coord.getX())
            self.target_py.append(coord.getY())
            self.target_pz.append(coord.getZ())
            extrapDate = extrapDate.shiftedBy(stepT)

    def oedot_plots(self):
        oe_params = ('sma', 'e_x', 'e_y', 'h_x', 'h_y')

        oedot = [self.adot_orbit, self.exdot_orbit, self.eydot_orbit, self.hxdot_orbit,
                 self.hydot_orbit]

        for i in range(len(oe_params)):
            plt.subplot(3,2,i+1)
            plt.plot(oedot[i])
            plt.ylabel(oe_params[i])
        plt.title('Rate of change of OE')
        plt.show()


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

    from Satmind.utils import OrnsteinUhlenbeck
    year, month, day, hr, minute, sec = 2018, 8, 1, 9, 30, 00.00
    date = [year, month, day, hr, minute, sec]

    dry_mass = 500.0
    fuel_mass = 150.0
    mass = [dry_mass, fuel_mass]
    duration = 24.0 * 60.0 ** 2 * 3

    # set the sc initial state
    a = 5500.0e3  # semi major axis (m)
    e = 0.2  # eccentricity
    i = 5.0  # inclination
    omega = 20.0  # perigee argument
    raan = 20.0  # right ascension of ascending node
    lM = 10.0  # mean anomaly
    state = [a, e, i, omega, raan, lM]

    # target state
    a_targ=6600.0e3
    e_targ = .24
    i_targ = 5.3
    omega_targ = 24.0
    raan_targ = 24.0
    lM_targ = 12.0
    state_targ = [a_targ, e_targ, i_targ, omega_targ, raan_targ, lM_targ]
    stepT = 1000.0

    env = OrekitEnv(state, state_targ, date, duration, mass, stepT)

    env.render_target()
    fw = [-0.6]
    fuel = []
    noise = OrnsteinUhlenbeck(np.zeros(3))
    for f in fw:

        reward = []
        i_t = []
        s = env.reset()
        i_prev = radians(i)
        F_s = 0.5
        while env._extrap_Date.compareTo(env.final_date) <= 0:
        # while env._currentOrbit.getA() <= a_targ:
            thrust_mag = np.clip(np.array([0.1, F_s, -0.6]),0.001, 1.0)

            position, r, done = env.step(thrust_mag)
            # i_cur = env._currentOrbit.getI()
            # inc = env.convert_to_keplerian(env._currentOrbit)
            # ta = inc.getTrueAnomaly()
            reward.append(r)
            # F_s -= .005

            # i_t.append(inc.getAnomaly())
            # i_t.append(env._sc_fuel.getKeplerianPeriod())
            # reward.append(i_cur)
            # if ta >= 0:
            #     thrust_mag = np.array([0.0, 0.0, -0.1])
            # else:
            #     thrust_mag = np.array([0.0, 0.0, 0.1])
            # if done:
                # break
        # print(env._extrap_Date)
        plt.plot(reward)
        plt.show()
        # env.render_plots(save=False, show=False)
        # env.oedot_plots()
        plt.show()
        # plt.figure()
        # plt.plot(i_t)
        # print(f'Done\nIncli: {degrees(env._currentOrbit.getI())}\n=====')

if __name__ == '__main__':
    main()