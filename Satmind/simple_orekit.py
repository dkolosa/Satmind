import orekit

from math import radians, degrees
# import matplotlib.pyplot as plt

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

from org.orekit.orbits import OrbitType
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode import AbstractIntegrator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.orbits import OrbitType
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity import ThirdBodyAttraction
from org.orekit.utils import IERSConventions

from org.orekit.forces.radiation import SolarRadiationPressure
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient
from org.orekit.forces.radiation import RadiationSensitive

from org.orekit.bodies import CelestialBodyFactory

from org.orekit.python import PythonEventHandler, PythonOrekitFixedStepHandler




def simple_keplarian(initialOrbit, initialDate):
    """
    :param initialOrbit: initial Keplarian orbit and central body
    :param initialDate: intial start date
    :return: plot xy orbit
    """

    # Simple extrapolation with Keplerian motion
    kepler = KeplerianPropagator(initialOrbit)
    # Set the propagator to slave mode (could be omitted as it is the default mode)
    kepler.setSlaveMode()
    # Setup propagation time
    # Overall duration in seconds for extrapolation
    duration = 24 * 60.0 ** 2
    # Stop date
    finalDate = AbsoluteDate(initial_date, duration, utc)
    # Step duration in seconds
    stepT = 30.0
    # Perform propagation
    # Extrapolation loop
    cpt = 1.0
    extrapDate = initial_date
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
    pass

# simple_keplarian(initialOrbit, initialDate)



def main():

    a = 24396159.0  # semi major axis (m)
    e = 0.720  # eccentricity
    i = radians(10.0)  # inclination
    omega = radians(50.0)  # perigee argument
    raan = radians(150)  # right ascension of ascending node
    lM = 0.0  # mean anomaly

    # Set inertial frame
    inertialFrame = FramesFactory.getEME2000()

    # Initial date in UTC time scale
    utc = TimeScalesFactory.getUTC()
    initial_date = AbsoluteDate(2004, 1, 1, 23, 30, 00.000, utc)

    # Setup orbit propagator
    # gravitation coefficient
    mu = 3.986004415e+14

    # Orbit construction as Keplerian
    initialOrbit = KeplerianOrbit(a, e, i, omega, raan, lM,
                                  PositionAngle.MEAN, inertialFrame,
                                  initial_date, mu)

    initial_state = SpacecraftState(initialOrbit)

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
    # ssc = IsotropicRadiationSingleCoefficient(100.0, 0.8)  # Spacecraft surface area (m^2), C_r absorbtion
    # srp = SolarRadiationPressure(CelestialBodyFactory.getSun(), a, ssc)  # sun, semi-major Earth, spacecraft sensitivity

    propagator.addForceModel(holmesFeatherstone)
    # propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getSun()))
    # propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))
    # propagator.addForceModel(srp)

    propagator.setMasterMode(60.0, TutorialStepHandler())

    propagator.setInitialState(initial_state)

    # propagator.setEphemerisMode()

    finalstate = propagator.propagate(initial_date.shiftedBy(10.*24*60**2))   # TIme shift in seconds

    # ephemeris = propagator.getGeneratedEphemeris()
    # print('Ephemeris defined from {} to {}'.format(ephemeris.getMinDate(), ephemeris.getMaxDate()))
    # o = OrbitType.KEPLERIAN.convertType(final_state.getOrbit())

    # print('Final State: date: {}\na: {} \ne: {} \ni:{} \ntheta {}'.format(
    #     final_state.getDate(), o.getA(), o.getE(), degrees(o.getI()), degrees(o.getLv())))

    # print("done")


class TutorialStepHandler(PythonOrekitFixedStepHandler):

    def init(self, s0, t):
        # print("          date                a           e","           i         \u03c9          \u03a9","          \u03bd")
        pass

    def handleStep(self, currentState, isLast):
        o = OrbitType.KEPLERIAN.convertType(currentState.getOrbit())
        print('{} '.format(currentState.getDate()))
        # print("{} {} {} {} {} {} {}\n".format(currentState.getDate(), o.getA(), o.getE(),
        #                                                               o.getI(),o.getPerigeeArgument(),
        #                                                               o.getRightAscensionOfAscendingNode(),
        #                                                               o.getTrueAnomaly()))
        if isLast:
            print("last step")
            # System.out.println("this was the last step ")
            # System.out.println()


if __name__ == '__main__':
    main()