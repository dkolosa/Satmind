from math import radians

from Satmind.env_orekit import OrekitEnv
import unittest


class MyTestCase(unittest.TestCase):
    def test_no_thrust(self):
        """ Test for 0 thrust propagation"""

        env = OrekitEnv()
        year, month, day, hr, minute, sec = 2018, 8, 1, 9, 30, 0.0
        date = [year, month, day, hr, minute, sec]
        env.set_date(date)

        mass = 1000.0
        fuel_mass = 500.0
        duration = 24.0 * 60.0 ** 2

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
        stepT = 10.0

        thrust_mag = 0.0
        while env._extrap_Date.compareTo(final_date) <= 0:
            position = env.step(thrust_mag, stepT)
            env.shift_date(stepT)

        print("done")

        self.assertAlmostEqual(env._currentOrbit.getA(), env._orbit.getA(), delta=1000)

        self.assertAlmostEqual(env._currentOrbit.getE(), env._orbit.getE(), places=2)


if __name__ == '__main__':
    unittest.main()
