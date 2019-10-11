#!~/anaconda3/bin/python

import numpy
import matplotlib.pyplot as plt

dt = numpy.load('rewards.npy')


plt.plot(dt)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.show()
