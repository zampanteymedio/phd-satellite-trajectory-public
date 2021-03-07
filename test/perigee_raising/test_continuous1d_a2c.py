import os
import unittest

from phd_satellite_trajectory.perigee_raising.continuous1d_a2c import main


class TestContinuous1dA2C(unittest.TestCase):
    def test_main(self):
        os.chdir('workspace')
        try:
            main([])
        finally:
            os.chdir('..')
