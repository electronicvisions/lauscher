import unittest
import numpy as np

from lauscher.audiowaves import MonoAudioWave
from lauscher.transformations import PeakNormalizer


class TestPeakNormalizer(unittest.TestCase):
    def test_peak_from_below(self):
        sine_track = MonoAudioWave(np.sin(np.linspace(0, 1000, 10000)),
                                   44000)

        # Normalized initialization
        self.assertAlmostEqual(max(abs(sine_track.samples)), 1, places=1)

        # Reduce to half-scale
        sine_track.samples /= 2
        self.assertAlmostEqual(max(abs(sine_track.samples)), 0.5, places=1)

        # Normalize again
        sine_track = sine_track.transform(PeakNormalizer())
        self.assertAlmostEqual(max(abs(sine_track.samples)), 1, places=1)

    def test_peak_from_above(self):
        sine_track = MonoAudioWave(np.sin(np.linspace(0, 1000, 10000)),
                                   44000)

        # Normalized initialization
        self.assertAlmostEqual(max(abs(sine_track.samples)), 1, places=1)

        # Amplify to double-cale
        sine_track.samples = sine_track.samples * 2
        self.assertAlmostEqual(max(abs(sine_track.samples)), 2, places=1)

        # Normalize again
        sine_track = sine_track.transform(PeakNormalizer())
        self.assertAlmostEqual(max(abs(sine_track.samples)), 1, places=1)
