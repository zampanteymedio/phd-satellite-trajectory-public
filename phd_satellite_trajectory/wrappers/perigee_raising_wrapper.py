from gym.wrappers import TransformReward

from phd_satellite_trajectory.wrappers.normalize_observation_space import NormalizeObservationSpace


def wrap(env):
    # Applying normalisation of observations
    wrapper_observation = NormalizeObservationSpace(env, lambda o: o / env.unwrapped.observation_space.high)
    # Applying normalisation of rewards
    wrapper_reward = TransformReward(wrapper_observation, lambda r: 1.e0 * r)
    return wrapper_reward
