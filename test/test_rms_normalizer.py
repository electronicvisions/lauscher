import unittest
import numpy as np

from lauscher.audiowaves import MonoAudioWave
from lauscher.transformations import RmsNormalizer


def rms(values):
    return np.sqrt(np.mean(values ** 2))


class TestRMSNormalizer(unittest.TestCase):
    def test_rms_from_above(self):
        sine_track = MonoAudioWave(np.sin(np.linspace(0, 1000, 10000)),
                                   44000)

        sine_track = sine_track.transform(RmsNormalizer(0.5))
        self.assertAlmostEqual(rms(sine_track.samples), 0.5, places=1)

    def test_peak_from_above(self):
        sine_track = MonoAudioWave(np.sin(np.linspace(0, 1000, 10000)),
                                   44000)

        # Normalize to 3dB -> factor 2 amplitude increase
        sine_track = sine_track.transform(RmsNormalizer(2))
        self.assertAlmostEqual(rms(sine_track.samples), 2, places=1)
