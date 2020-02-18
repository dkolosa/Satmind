from math import radians

from Satmind.env_orekit import OrekitEnv
import unittest
import pytest


class MyTestCase(unittest.TestCase):
    def test_no_thrust(self):
        """ Test for 0 thrust propagation"""

        env = OrekitEnv()
        done = False
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
            state, reward, done, _ = env.step(thrust_mag, stepT)
            env.shift_date(stepT)

        print("done")

        self.assertAlmostEqual(env._currentOrbit.getA(), env._orbit.getA(), delta=1000)

        self.assertAlmostEqual(env._currentOrbit.getE(), env._orbit.getE(), places=2)

def test_propagation():
    year, month, day, hr, minute, sec = 2018, 8, 1, 9, 30, 00.00
    date = [year, month, day, hr, minute, sec]

    mass = 100.0
    fuel_mass = 100.0
    duration = 24.0 * 60.0 ** 2 * 1

    # set the sc initial state
    a = 5500.0e3  # semi major axis (m)
    e = 0.1  # eccentricity
    i = 45.0  # inclination
    omega = 10.0  # perigee argument
    raan = 10.0  # right ascension of ascending node
    lM = 20.0  # mean anomaly
    state = [a, e, i, omega, raan, lM]

    # target state
    a_targ = 6600.0e3
    e_targ = e
    i_targ = 35.0
    omega_targ = omega
    raan_targ = raan
    lM_targ = lM
    state_targ = [a_targ, e_targ, i_targ, omega_targ, raan_targ, lM_targ]
    stepT = 1000.0

    env = OrekitEnv(state, state_targ, date, duration, mass, fuel_mass, stepT)

    env.render_target()
    fw = [0.1, 1.5, 0.1]
    i = []
    for f in fw:
        thrust_mag = np.array([0.0, 0.0, f])
        reward = []
        s = env.reset()
        while env._extrap_Date.compareTo(env.final_date) <= 0:
            position, r, done = env.step(thrust_mag)
        i.append(degrees(env._currentOrbit.getI()))
    assert i[0] == i[-1]

if __name__ == '__main__':
    # unittest.main()
