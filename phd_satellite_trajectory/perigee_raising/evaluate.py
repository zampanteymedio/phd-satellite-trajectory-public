import argparse
import os
import sys
import time

import gym
import matplotlib.pyplot as plt

from phd_satellite_trajectory.wrappers.perigee_raising_wrapper import wrap
from stable_baselines3.a2c import A2C
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import get_action_dim


def main(argv):
    parser = argparse.ArgumentParser(description='Run A2C on the perigee raising continuous 3D environment.')
    parser.add_argument('--file', '-f', metavar='F', type=str,
                        help='file name with the model to evaluate')
    parser.add_argument('--case_number', '-c', metavar='C', type=int, default=int(time.time()),
                        help='case number for the agent (seed for rng). Default: 0')
    parser.add_argument('--runs', '-r', metavar='R', type=int, default=10,
                        help='number of runs to evaluate the model with')
    parser.add_argument('--verbose', '-v', metavar='V', type=int, default=0,
                        help='verbose level in range [0, 2]. Default: 0')
    args = parser.parse_args(argv)
    process(**vars(args))


def process(file, case_number, runs, verbose):
    print(f"File: {file}")
    agent = load_agent(file)
    name = f"{file.split('/')[-1].split('.')[0]}"
    print(f"Name: {name}")
    agent.name = name
    env = Monitor(get_env(case_number))
    agent.policy.action_dist = SquashedDiagGaussianDistribution(get_action_dim(env.action_space))
    print(f"  --> Testing...")
    test_agent(agent, env, runs, verbose)


def load_agent(file):
    agent = A2C.load(file)
    return agent


def get_env(case_number):
    env = wrap(gym.make('PerigeeRaising-Continuous3D-v0'))
    env.seed(case_number)
    return env


def test_agent(agent, env, runs, verbose):
    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=runs,
                                              callback=EvaluateCallback(agent.name).callback)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


class EvaluateCallback:
    def __init__(self, name):
        self.folder = f"evals/{name}/plots"
        os.makedirs(self.folder, exist_ok=True)
        self.n = 0

    def callback(self, loc, glo):
        if loc['done']:
            print(self.n)
            figure = loc['env'].unwrapped.render('plot')
            figure.savefig(f"{self.folder}/{self.n}.svg", format="svg")
            plt.close(figure)
            self.n += 1


if __name__ == '__main__':
    main(sys.argv[1:])
