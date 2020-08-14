import unittest
import numpy as np

from lauscher.audiowaves import MonoAudioWave


class TestAudioWave(unittest.TestCase):
    def test_rms_calculation(self):
        sine_track = MonoAudioWave(np.sin(np.linspace(0, 1000, 10000)),
                                   44000)

        # Full-scale sine has -3db RMS
        self.assertAlmostEqual(sine_track.get_rms(), -3, places=1)

        # Half signal is a 6dB decrease
        sine_track.samples = sine_track.samples / 2
        self.assertAlmostEqual(sine_track.get_rms(), -9, places=1)
