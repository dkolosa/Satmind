from __future__ import print_function

import orekit
from math import radians, degrees
import matplotlib.pyplot as plt

orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir

setup_orekit_curdir()

from org.orekit.errors import OrekitException;
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


from org.orekit.python import PythonEventHandler, PythonOrekitFixedStepHandler


def main():

    a = 24396159.0    # semi major axis (m)
    e = 0.720  # eccentricity
    i = radians(10.0) # inclination
    omega = radians(50.0) # perigee argument
    raan = radians(150) #right ascension of ascending node
    lM = 0.0 # mean anomaly

    # Set inertial frame
    inertialFrame = FramesFactory.getEME2000()

    # Initial date in UTC time scale
    utc = TimeScalesFactory.getUTC()
    initial_date = AbsoluteDate(2004, 1, 1, 23, 30, 00.000, utc)

    # Setup orbit propagator
    #gravitation coefficient
    mu = 3.986004415e+14

    # Orbit construction as Keplerian
    initialOrbit = KeplerianOrbit(a, e, i, omega, raan, lM,
                                  PositionAngle.MEAN, inertialFrame,
                                  initial_date, mu)

    initial_state = SpacecraftState(initialOrbit)

    # simple_keplarian(initialOrbit, initial_date)

    # use a numerical propogator
    min_step = 0.001
    max_step = 1000.0
    position_tolerance = 10.0

    propagation_type = OrbitType.KEPLERIAN
    tolerances = NumericalPropagator.tolerances(position_tolerance, initialOrbit,
                                                propagation_type)

    integrator = DormandPrince853Integrator(min_step, max_step, 1e-5, 1e-10)

    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(propagation_type)

    # force model gravity field
    provider = GravityFieldFactory.getNormalizedProvider(10, 10)
    holmesFeatherstone = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), provider)

    # SRP
    ssc = IsotropicRadiationSingleCoefficient(100.0, 0.8)  # Spacecraft surface area (m^2), C_r absorbtion
    srp = SolarRadiationPressure(CelestialBodyFactory.getSun(), a, ssc)  # sun, semi-major Earth, spacecraft sensitivity

    propagator.addForceModel(holmesFeatherstone)
    propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getSun()))
    propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))
    # propagator.addForceModel(srp)

    handler = TutorialStepHandler()
    propagator.setMasterMode(60.0, handler)

    propagator.setInitialState(initial_state)
    final_state = propagator.propagate(initial_date.shiftedBy(10.0*24*60**2))   # TIme shift in seconds

    # o = OrbitType.KEPLERIAN.convertType(final_state.getOrbit())
    #
    # print('Final State: date: {}\na: {} \ne: {} \ni:{} \ntheta {}'.format(
    #     final_state.getDate(), o.getA(), o.getE(), degrees(o.getI()), degrees(o.getLv())))

    print("done")


def simple_keplarian(initialOrbit, initial_date):
    """
    :param initialOrbit: initial Keplarian orbit and central body
    :param initialDate: intial start date
    :return: plot xy orbit
    """
    utc = TimeScalesFactory.getUTC()

    # Simple extrapolation with Keplerian motion
    kepler = KeplerianPropagator(initialOrbit)
    # Set the propagator to slave mode (could be omitted as it is the default mode)
    # kepler.setSlaveMode()
    # Setup propagation time
    # Overall duration in seconds for extrapolation
    duration = 24 * 60.0 ** 2
    # Stop date
    finalDate = initial_date.shiftedBy(duration)
    # Step duration in seconds
    stepT = 30.0
    # Perform propagation
    # Extrapolation loop
    cpt = 1.0
    extrapDate = initial_date
    px, py = [], []

    max_check = 60.0
    threshold = 0.0010
    true_anomaly = radians(90.0)

    detect_ta_event = PositionAngleDetector(OrbitType.KEPLERIAN, PositionAngle.TRUE, true_anomaly)
    detect_ta_event = detect_ta_event.withHandler(AnomalyHandler().of_(PositionAngleDetector()))
    kepler.addEventDetector(detect_ta_event)

    kepler.propagate(finalDate)

    # while extrapDate.compareTo(finalDate) <= 0:
    #     currentState = kepler.propagate(extrapDate)
    #     print('step {}: time {} {}\n'.format(cpt, currentState.getDate(), currentState.getOrbit()))
    #     coord = currentState.getPVCoordinates().getPosition()
    #     px.append(coord.getX())
    #     py.append(coord.getY())
    #     # P[:,cpt]=[coord.getX coord.getY coord.getZ]
    #     extrapDate = AbsoluteDate(extrapDate, stepT, utc)
    #     cpt += 1
    # plt.plot(px, py)
    # plt.show()
    pass


class AnomalyHandler(PythonEventHandler):

    def eventOccured(self, sc_state, T, detector):
        # print('Reached target TA on')
        return EventHandler.Action.CONTINUE

    def resetState(self, detector, oldstate):
        return oldstate


class TutorialStepHandler(PythonOrekitFixedStepHandler):

    def init(self, s0, t):
        print('Orbial Elements')

    def handleStep(self, currentState, isLast):
        o = OrbitType.KEPLERIAN.convertType(currentState.getOrbit())
        print(o.getDate())
        print('a:{:5.3f}, e:{:5.3f}, i:{:5.3f}, theta:{:5.3f}'.format(o.getA(), o.getE(), o.getI(), o.getLv()))
        #         print(" %12.3f %10.8f %106f %10.6f %10.6f %10.6f'%o.getA(), o.getE(),
        #                                                                       o.getI(),o.getPerigeeArgument(),
        #                                                                       o.getRightAscensionOfAscendingNode(),
        #                                                                       o.getTrueAnomaly())
        if isLast:
            print('this was the last step ')




if __name__ == '__main__':
    main()
