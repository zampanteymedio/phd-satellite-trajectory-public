import os
import unittest

from phd_satellite_trajectory.perigee_raising.continuous3d_a2c import main


class TestContinuous3dA2C(unittest.TestCase):
    def test_main(self):
        os.chdir('workspace')
        try:
            main([])
        finally:
            os.chdir('..')
