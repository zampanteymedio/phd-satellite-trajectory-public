import argparse
import sys
import matplotlib.image as mpimg

import gym
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import get_action_dim
from phd_satellite_trajectory.wrappers.normalize_observation_space import NormalizeObservationSpace


def main(argv):
    parser = argparse.ArgumentParser(description='Plot a real trajectory.')
    parser.add_argument('--file', '-f', metavar='F', type=str,
                        help='file containing the agent')
    args = parser.parse_args(argv)
    process(**vars(args))


def process(file):
    env = gym.make('PerigeeRaising-Continuous3D-v0')
    env.unwrapped._ref_sv[2] = 0.0
    env.unwrapped._ref_sv[3] = 0.0
    env.unwrapped._ref_sv[4] = 0.0
    env = NormalizeObservationSpace(env, lambda o: o / env.unwrapped.observation_space.high)
    env = Monitor(env)
    env.seed(42)
    agent = A2C.load(file)
    agent.policy.action_dist = SquashedDiagGaussianDistribution(get_action_dim(env.action_space))
    evaluate_policy(agent, env, n_eval_episodes=1)

    hist_sc_state = env.unwrapped.hist_sc_state
    hist_action = env.unwrapped.hist_action
    x = np.array(list(map(lambda sc_state: sc_state.getPVCoordinates().getPosition().getX(), hist_sc_state))) / 1000.0  # Convert to km
    y = np.array(list(map(lambda sc_state: sc_state.getPVCoordinates().getPosition().getY(), hist_sc_state))) / 1000.0  # Convert to km

    env2 = gym.make('PerigeeRaising-Continuous3D-v0')
    env2.unwrapped._ref_sv[0] = 11000000.0 / 1.05
    env2.unwrapped._ref_sv[1] = 0.05
    env2.unwrapped._ref_sv[2] = 0.0
    env2.unwrapped._ref_sv[3] = 0.0
    env2.unwrapped._ref_sv[4] = 0.0
    env2 = NormalizeObservationSpace(env2, lambda o: o / env2.unwrapped.observation_space.high)
    env2 = Monitor(env2)
    env2.seed(42)
    agent = A2C.load(file)
    agent.policy.action_dist = SquashedDiagGaussianDistribution(get_action_dim(env.action_space))
    evaluate_policy(agent, env2, n_eval_episodes=1)

    hist_sc_state2 = env2.unwrapped.hist_sc_state
    hist_action2 = env2.unwrapped.hist_action
    x2 = np.array(list(map(lambda sc_state: sc_state.getPVCoordinates().getPosition().getX(), hist_sc_state2))) / 1000.0  # Convert to km
    y2 = np.array(list(map(lambda sc_state: sc_state.getPVCoordinates().getPosition().getY(), hist_sc_state2))) / 1000.0  # Convert to km

    fig, axs = plt.subplots(1, 1, figsize=(4.8, 3.0))
    axs.set_xlim(-12000, 12000)
    axs.set_ylim(-12000, 12000)
    axs.grid(False)
    axs.plot(x, y, "k", zorder=2)
    l2, = axs.plot(x2, y2, zorder=1)
    l2.set_color("#777777")
    axs.legend(["Before", "After"], loc='upper right', frameon=False, bbox_to_anchor=(0.0, 1.0))
    im = mpimg.imread('earth.png')
    plt.imshow(im, extent=[-6400, 6400, -6400, 6400], interpolation="none")
    axs.set_aspect('equal')
    plt.text(11000, 0, "Pericenter")
    plt.text(-18500, 0, "Apocenter")
    plt.axis('off')
    plt.tight_layout()
    fig.savefig("orbit.pdf", format="pdf")
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1:])
