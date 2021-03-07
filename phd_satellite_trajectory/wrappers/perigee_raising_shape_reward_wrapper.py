import numpy as np
from gym import RewardWrapper
from org.orekit.orbits import OrbitType
from org.orekit.orbits import PositionAngle
from org.orekit.propagation.events import EventsLogger
from org.orekit.propagation.events import PositionAngleDetector


class PerigeeRaisingShapeRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super(PerigeeRaisingShapeRewardWrapper, self).__init__(env)
        self._perigee_logger = None
        self._apogee_logger = None
        self._first_perigee = None
        self._last_perigee = None
        self._last_apogee_diff = 0.0
        self._first_mass = None
        self._last_mass = None

    def reset(self, **kwargs):
        obs = super(PerigeeRaisingShapeRewardWrapper, self).reset(**kwargs)

        self._perigee_logger = EventsLogger()
        perigee_detector = PositionAngleDetector(OrbitType.KEPLERIAN, PositionAngle.MEAN, 0.0)
        self.env.unwrapped.add_event_detector(self._perigee_logger.monitorDetector(perigee_detector))

        self._apogee_logger = EventsLogger()
        apogee_detector = PositionAngleDetector(OrbitType.KEPLERIAN, PositionAngle.MEAN, np.pi)
        self.env.unwrapped.add_event_detector(self._apogee_logger.monitorDetector(apogee_detector))

        self._first_perigee = None
        self._last_perigee = None
        self._last_apogee_diff = 0.0
        self._first_mass = None
        self._last_mass = None

        return obs

    def reward(self, reward):
        additional_reward = 0.0

        if self.mass_in_last_step is not None:
            if self._last_mass is not None:
                additional_reward += 1.0e5 * (self.mass_in_last_step - self._last_mass)
            self._last_mass = self.mass_in_last_step
            self._first_mass = self._last_mass if self._first_mass is None else self._first_mass

        if self.perigee_in_last_step is not None:
            if self._last_perigee is not None:
                additional_reward += self.perigee_in_last_step - self._last_perigee
            self._last_perigee = self.perigee_in_last_step
            self._first_perigee = self._last_perigee if self._first_perigee is None else self._first_perigee
            self._perigee_logger.clearLoggedEvents()

        if self.apogee_in_last_step is not None:
            apogee_target = 11000000.0
            apogee_diff = -abs(self.apogee_in_last_step - apogee_target)
            additional_reward += apogee_diff - self._last_apogee_diff
            self._last_apogee_diff = apogee_diff
            self._apogee_logger.clearLoggedEvents()

        if reward != 0.0:
            additional_reward -= (self._last_perigee - self._first_perigee) + self._last_apogee_diff + \
                                 1.0e5 * (self._last_mass - self._first_mass)

        # Mass is not used in shape rewards for now
        return reward + 0*additional_reward

    @property
    def perigee_in_last_step(self):
        return self._get_distance_from_logger(self._perigee_logger)

    @property
    def apogee_in_last_step(self):
        return self._get_distance_from_logger(self._apogee_logger)

    @property
    def mass_in_last_step(self):
        events = self._perigee_logger.getLoggedEvents()
        if events.size() > 0:
            return events.get(events.size() - 1).getState().getMass()
        events = self._apogee_logger.getLoggedEvents()
        if events.size() > 0:
            return events.get(events.size() - 1).getState().getMass()
        return None

    @staticmethod
    def _get_distance_from_logger(logger):
        events = logger.getLoggedEvents()
        if events.size() > 0:
            return events.get(events.size() - 1).getState().getPVCoordinates().getPosition().getNorm()
        return None
