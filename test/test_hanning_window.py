import unittest
from numpy import hanning

from lauscher.audiowaves import MonoAudioWave
from lauscher.transformations import HanningWindow


class TestHanningWindow(unittest.TestCase):
    def test_default_windowsize(self):
        num_samples = 1000
        one_track = MonoAudioWave([1] * num_samples, 44000)

        hanned = one_track.transform(HanningWindow())

        for idx, value in enumerate(hanned.samples):  # pylint: disable=no-member
            self.assertAlmostEqual(hanning(num_samples)[idx], value, places=2)
