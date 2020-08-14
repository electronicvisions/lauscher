import unittest
import numpy as np

from lauscher.spike_train import SpikeTrain


class TestSpiketrain(unittest.TestCase):
    def test_from_dense(self):
        data = np.asarray([[1, 0, 0, 1, 0],
                           [0, 1, 0, 1, 0],
                           [0, 0, 1, 1, 0]])
        sample_rate = 2

        result = SpikeTrain.from_dense(data, sample_rate)

        label_time_tuples = set(zip(result.spike_labels, result.spike_times))

        self.assertSetEqual(
            {(0, 0.0),
             (1, 0.5),
             (2, 1.0),
             (0, 1.5),
             (1, 1.5),
             (2, 1.5)},
            label_time_tuples)
