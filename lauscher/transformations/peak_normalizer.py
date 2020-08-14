import numpy as np

from lauscher.abstract import Transformation
from lauscher.audiowaves import MonoAudioWave


class PeakNormalizer(Transformation):
    def __call__(self, data: MonoAudioWave) -> MonoAudioWave:
        assert isinstance(data, MonoAudioWave)

        data.samples = data.samples / np.max(np.abs(data.samples))
        return data
