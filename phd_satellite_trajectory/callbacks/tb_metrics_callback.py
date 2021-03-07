import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.vec_env import VecEnv


class TbMetricsCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TbMetricsCallback, self).__init__(verbose)
        self.unwrapped_training_env = None

    def _init_callback(self):
        self.unwrapped_training_env = self.training_env.unwrapped
        while isinstance(self.unwrapped_training_env, VecEnv):
            self.unwrapped_training_env = self.unwrapped_training_env.envs[0].unwrapped

    def _on_step(self):
        if self.locals.get('done') is not None or \
           (self.locals.get('dones') is not None and
            isinstance(self.locals.get('dones'), (list, np.ndarray)) and
            self.locals['dones'][-1]):
            figure = self.unwrapped_training_env.render('prev_plot')
            self.logger.record('custom/plot', Figure(figure, True), exclude=("stdout", "log", "json", "csv"))
            plt.close(figure)

        self.logger.record('custom/policy_0_bias', self.model.policy.mlp_extractor.policy_net._modules["0"].bias, exclude=("stdout", "log", "json", "csv"))
        self.logger.record('custom/policy_0_weight', self.model.policy.mlp_extractor.policy_net._modules["0"].weight, exclude=("stdout", "log", "json", "csv"))
        self.logger.record('custom/value_0_bias', self.model.policy.mlp_extractor.value_net._modules["0"].bias, exclude=("stdout", "log", "json", "csv"))
        self.logger.record('custom/value_0_weight', self.model.policy.mlp_extractor.value_net._modules["0"].weight, exclude=("stdout", "log", "json", "csv"))

        return True
