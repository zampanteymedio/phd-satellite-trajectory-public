import logging
import sys

import matplotlib.pyplot as plt

import gym_satellite_trajectory

logging.basicConfig(stream=sys.stdout)

plt.ioff()

gym_satellite_trajectory.register_environments()
