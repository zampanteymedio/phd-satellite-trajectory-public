from gym.spaces import Box
from gym.wrappers import TransformObservation


class NormalizeObservationSpace(TransformObservation):
    def __init__(self, env, f):
        super(NormalizeObservationSpace, self).__init__(env, f)
        if isinstance(self.observation_space, Box):
          self.observation_space = Box(
              low=self.f(self.observation_space.low),
              high=self.f(self.observation_space.high),
              shape=self.observation_space.shape,
              dtype=self.observation_space.dtype)
