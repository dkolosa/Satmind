#!/bin/python3

import numpy as np

mu = 398600.0


def get_sma(state, thrust_s, duration):

    a, e, E = state
    da = 2*np.sqrt(a/mu)*thrust_s*((a*np.sqrt(1-e**2))/(1-e*np.cos(E)))
    return da * duration


def get_time(state, thrust_s, da):
    a, e, E = state
    dt = da/(2*np.sqrt(a/mu)*thrust_s*((a*np.sqrt(1-e**2))/(1-e*np.cos(E))))
    return dt


def get_thrust(state, da, duration):
    a = state[0]
    return da/(duration*np.sqrt(a/mu)*a)


if __name__ == '__main__':

    a, e, E = 5500.0, 0.0001, 0.0001
    state = [a, e, E]
    F_s = 0.50
    duration = 1*24*60**2
    da = 1100.0
    # print(get_sma(state, F_s, duration))
    # thrust = get_thrust(state,da, duration)
    mass = 1500.0
    print(get_time(state, F_s, 25000.0))
    # print(f'sma: {}')
    # print('tangential thrust accl:{} m/s^2 \nthrust: {} N'.format(thrust, thrust * mass))
