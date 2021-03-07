import argparse
import sys
import time

import gym
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

from phd_satellite_trajectory.wrappers.perigee_raising_wrapper import wrap
from stable_baselines3.a2c import A2C
from stable_baselines3.a2c.policies import MlpPolicy
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import FlattenExtractor


def main(argv):
    parser = argparse.ArgumentParser(description='Run A2C on the perigee raising continuous 3D environment.')
    parser.add_argument('--case_number', '-c', metavar='C', type=int, default=int(time.time()),
                        help='case number for the agent (seed for rng). Default: 0')
    parser.add_argument('--layers', '-l', metavar='L', type=int, nargs='+', default=[8],
                        help='internal layers to be used for DQN agent. Default: [8]')
    parser.add_argument('--steps', '-s', metavar='S', type=int, default=1000,
                        help='number of steps to train the agent for. Default: 1000')
    parser.add_argument('--envs', '-e', metavar='E', type=int, default=1,
                        help='number of environments to use to train the agent. Default: 100')
    parser.add_argument('--verbose', '-v', metavar='V', type=int, default=0,
                        help='verbose level in range [0, 2]. Default: 0')
    args = parser.parse_args(argv)
    process(**vars(args))


def process(layers, case_number, steps, envs, verbose):
    name = f"c3a_A2C_{str(layers).replace(' ', '')}_{case_number}"
    print(f"Case: {name}")
    env = make_vec_env('PerigeeRaising-Continuous3D-v0', n_envs=envs, wrapper_class=lambda x: wrap(x))
    agent = create_agent(env, name, case_number, layers, verbose)
    print(f"  --> Training...")
    train_agent(agent, name, steps=steps, callbacks=[])
    print(f"  --> Testing...")
    test_agent(agent)


def get_env(case_number=None):
    env = gym.make('PerigeeRaising-Continuous3D-v0')
    env.seed(case_number)
    return wrap(env)


def create_agent(env, name, case_number, layers, verbose):
    # return DQN.load("./saved_agents/d1a_DQN_0.zip", env=env)
    agent = A2C(
        policy=MlpPolicy,
        env=env,
        learning_rate=1.0e-3,
        n_steps=env.unwrapped.envs[0].unwrapped._max_steps,
        gamma=1.0,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        use_rms_prop=True,
        use_sde=False,
        sde_sample_freq=-1,
        normalize_advantage=False,
        tensorboard_log="./tblog",
        create_eval_env=True,
        policy_kwargs=dict(
            net_arch=[dict(vf=layers, pi=layers)],
            activation_fn=th.nn.LeakyReLU,
            ortho_init=True,
            log_std_init=1.0,
            full_std=True,
            sde_net_arch=None,
            use_expln=False,
            squash_output=True,
            features_extractor_class=FlattenExtractor,
            features_extractor_kwargs=dict(),
            normalize_images=False,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=dict(
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0,
                amsgrad=False,
            ),
        ),
        verbose=verbose,
        seed=case_number,
        device="cpu",
        _init_setup_model=True,
    )
    agent.name = name
    agent.policy.action_dist = SquashedDiagGaussianDistribution(get_action_dim(env.action_space))
    writer = SummaryWriter(log_dir=agent.tensorboard_log + "/" + agent.name + "_1")
    writer.add_graph(agent.policy, th.as_tensor(np.zeros((1, 8))).to(agent.policy.device))
    writer.close()
    return agent


def train_agent(agent, name, steps, callbacks):
    agent.learn(total_timesteps=steps,
                log_interval=10,
                tb_log_name=name,
                callback=callbacks,
                eval_env=Monitor(get_env()),
                n_eval_episodes=1,
                eval_freq=agent.env.unwrapped.envs[0].unwrapped._max_steps * 6,
                eval_log_path="./evals/" + name)
    agent.save("./saved_agents/" + name + ".zip")


def test_agent(agent):
    env = Monitor(get_env(42))  # We always test with seed 42 because it's the answer to... :-)
    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == '__main__':
    main(sys.argv[1:])
