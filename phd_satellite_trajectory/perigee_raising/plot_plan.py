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
    parser = argparse.ArgumentParser(description='Plot a planned trajectory.')
    parser.add_argument('--file', '-f', metavar='F', type=str,
                        help='file containing the agent')
    args = parser.parse_args(argv)
    process(**vars(args))


def process(file):
    env = gym.make('PerigeeRaising-Continuous3D-v0')
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
    v = np.array(list(map(lambda sc_state: sc_state.getPVCoordinates().getVelocity().toArray(), hist_sc_state)))
    h = np.array(list(map(lambda sc_state: sc_state.getPVCoordinates().getMomentum().toArray(), hist_sc_state)))
    angle_f_v = list(map(lambda q:
                         np.degrees(np.arccos(
                             np.dot(q[0], q[1]) / np.linalg.norm(q[0]) / (np.linalg.norm(q[1]) + 1e-10)
                         )),
                         zip(v, hist_action)))
    hist_action_plane = list(map(lambda q: q[1] - np.dot(q[1], q[0]) * q[0] / (np.linalg.norm(q[0]) ** 2),
                                 zip(h, hist_action)))
    angle_fp_v = list(map(lambda q:
                          np.degrees(np.arccos(
                              np.dot(q[0], q[1] * [1, 1, 0]) / np.linalg.norm(q[0]) / (
                                      np.linalg.norm(q[1] * [1, 1, 0]) + 1e-10)
                          )),
                          zip(v, hist_action_plane)))

    fig, axs = plt.subplots(1, 1, figsize=(4.8, 3.0))
    axs.ticklabel_format(axis='y', style='plain', useOffset=ra[0])
    axs.set_xlim(time[0], time[-1])
    axs.set_ylim(ra[0] - 20.0, ra[0] + 20.0)
    axs.grid(True)
    axs.set_xlabel("time (h)")
    axs.set_ylabel("ra (km)")
    axs.plot(time, ra, "k")
    plt.tight_layout()
    fig.savefig("plan_ra.pdf", format="pdf")
    plt.close(fig)

    fig, axs = plt.subplots(1, 1, figsize=(4.8, 3.0))
    axs.ticklabel_format(axis='y', style='plain', useOffset=rp[0])
    axs.set_xlim(time[0], time[-1])
    axs.set_ylim(rp[0] - 5.0, rp[0] + 35.0)
    axs.grid(True)
    axs.set_xlabel("time (h)")
    axs.set_ylabel("rp (km)")
    axs.plot(time, rp, "k")
    plt.tight_layout()
    fig.savefig("plan_rp.pdf", format="pdf")
    plt.close(fig)

    fig, axs = plt.subplots(1, 1, figsize=(4.8, 3.0))
    axs.ticklabel_format(axis='y', style='plain', useOffset=mass[0])
    axs.set_xlim(time[0], time[-1])
    axs.set_ylim(mass[0] - 0.04, mass[0])
    axs.grid(True)
    axs.set_xlabel("time (h)")
    axs.set_ylabel("mass (kg)")
    axs.plot(time, mass, "k")
    plt.tight_layout()
    fig.savefig("plan_m.pdf", format="pdf")
    plt.close(fig)

    fig, axs = plt.subplots(1, 1, figsize=(4.8, 3.0))
    axs.ticklabel_format(axis='y', style='plain')
    axs.set_xlim(time[0], time[-1])
    axs.set_ylim(-1.3, 1.3)
    axs.grid(True)
    axs.set_xlabel("time (h)")
    axs.set_ylabel("action")
    l1, l2, l3 = axs.plot(time[0:-1], hist_action, "k")
    l1.set_color("#000000")
    l2.set_color("#777777")
    l3.set_color("#BBBBBB")
    axs.legend(["Act1", "Act2", "Act3"], loc='upper left')
    plt.tight_layout()
    fig.savefig("plan_action.pdf", format="pdf")
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1:])
