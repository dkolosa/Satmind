#!/bin/python3

import numpy as np

mu = 398600


def get_sma(state, thrust_s, duration):

    a, e, E = state
    da = 2*np.sqrt(a/mu)*thrust_s*((a*np.sqrt(1-e**2))/(1-e*np.cos(E)))
    return da * duration


def get_thrust(state, da, duration):
    a = state[0]
    return da/duration / (2*np.sqrt(a/mu)* a)


if __name__ == '__main__':
#TODO: given a circular orbit, determine the required time delta-a and thrust required

    a, e, E = 35_000.0, 0.0001, 0.0001
    state = [a, e, E]
    F_s = 0.10
    duration = 1*24*60**2
    da = 1000
    # print(get_sma(state, F_s, duration))
    thrust = get_thrust(state,da, duration)
    mass = 2000
    print('tangential thrust accl:{} m/s^2 \nthrust: {} N'.format(thrust, thrust * mass))
