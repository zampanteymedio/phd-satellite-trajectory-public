import argparse
import sys

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
    env = gym.make('PerigeeRaising-Continuous3D-v0', use_perturbations=True, perturb_action=True)
    env = NormalizeObservationSpace(env, lambda o: o / env.unwrapped.observation_space.high)
    env = Monitor(env)
    env.seed(42)
    agent = A2C.load(file)
    agent.policy.action_dist = SquashedDiagGaussianDistribution(get_action_dim(env.action_space))
    evaluate_policy(agent, env, n_eval_episodes=1)

    hist_sc_state = env.unwrapped.hist_sc_state
    hist_action = env.unwrapped.hist_action
    time = np.array(list(map(lambda sc_state: sc_state.getDate().durationFrom(hist_sc_state[0].getDate()),
                             hist_sc_state))) / 3600.0  # Convert to hours
    a = np.array(list(map(lambda sc_state: sc_state.getA(), hist_sc_state))) / 1000.0  # Convert to km
    e = np.array(list(map(lambda sc_state: sc_state.getE(), hist_sc_state)))
    mass = np.array(list(map(lambda sc_state: sc_state.getMass(), hist_sc_state)))
    ra = a * (1.0 + e)
    rp = a * (1.0 - e)

    env2 = gym.make('PerigeeRaising-Continuous3D-v0')
    env2 = NormalizeObservationSpace(env2, lambda o: o / env2.unwrapped.observation_space.high)
    env2 = Monitor(env2)
    env2.seed(42)
    agent = A2C.load(file)
    agent.policy.action_dist = SquashedDiagGaussianDistribution(get_action_dim(env.action_space))
    evaluate_policy(agent, env2, n_eval_episodes=1)

    hist_sc_state2 = env2.unwrapped.hist_sc_state
    hist_action2 = env2.unwrapped.hist_action
    time2 = np.array(list(map(lambda sc_state: sc_state.getDate().durationFrom(hist_sc_state2[0].getDate()),
                             hist_sc_state2))) / 3600.0  # Convert to hours
    a2 = np.array(list(map(lambda sc_state: sc_state.getA(), hist_sc_state2))) / 1000.0  # Convert to km
    e2 = np.array(list(map(lambda sc_state: sc_state.getE(), hist_sc_state2)))
    mass2 = np.array(list(map(lambda sc_state: sc_state.getMass(), hist_sc_state2)))
    ra2 = a2 * (1.0 + e2)
    rp2 = a2 * (1.0 - e2)

    fig, axs = plt.subplots(1, 1, figsize=(4.8, 3.0))
    axs.ticklabel_format(axis='y', style='plain', useOffset=ra[0])
    axs.set_xlim(time[0], time[-1])
    axs.set_ylim(ra[0] - 20.0, ra[0] + 20.0)
    axs.grid(True)
    axs.set_xlabel("time (h)")
    axs.set_ylabel("ra (km)")
    l2, = axs.plot(time2, ra2, "--")
    l2.set_color("#777777")
    axs.plot(time, ra, "k")
    axs.legend(["Planned", "Real"], loc='upper left')
    plt.tight_layout()
    fig.savefig("real_ra.pdf", format="pdf")
    plt.close(fig)

    fig, axs = plt.subplots(1, 1, figsize=(4.8, 3.0))
    axs.ticklabel_format(axis='y', style='plain', useOffset=rp[0])
    axs.set_xlim(time[0], time[-1])
    axs.set_ylim(rp[0] - 5.0, rp[0] + 35.0)
    axs.grid(True)
    axs.set_xlabel("time (h)")
    axs.set_ylabel("rp (km)")
    l2, = axs.plot(time2, rp2, "--")
    l2.set_color("#777777")
    axs.plot(time, rp, "k")
    axs.legend(["Planned", "Real"], loc='upper left')
    plt.tight_layout()
    fig.savefig("real_rp.pdf", format="pdf")
    plt.close(fig)

    fig, axs = plt.subplots(1, 1, figsize=(4.8, 3.0))
    axs.ticklabel_format(axis='y', style='plain', useOffset=mass[0])
    axs.set_xlim(time[0], time[-1])
    axs.set_ylim(mass[0] - 0.04, mass[0])
    axs.grid(True)
    axs.set_xlabel("time (h)")
    axs.set_ylabel("mass (kg)")
    l2, = axs.plot(time2, mass2, "--")
    l2.set_color("#777777")
    axs.plot(time, mass, "k")
    axs.legend(["Planned", "Real"], loc='upper right')
    plt.tight_layout()
    fig.savefig("real_m.pdf", format="pdf")
    plt.close(fig)

    fig, axs = plt.subplots(1, 1, figsize=(4.8, 3.0))
    axs.ticklabel_format(axis='y', style='plain')
    axs.set_xlim(time[0], time[-1])
    axs.set_ylim(-1.3, 1.3)
    axs.grid(True)
    axs.set_xlabel("time (h)")
    axs.set_ylabel("action")
    l1, l2, l3 = axs.plot(time[0:-1], hist_action)
    l1.set_color("#000000")
    l2.set_color("#777777")
    l3.set_color("#BBBBBB")
    axs.legend(["Act1", "Act2", "Act3"], loc='upper left')
    plt.tight_layout()
    fig.savefig("real_action.pdf", format="pdf")
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1:])
