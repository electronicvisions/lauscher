import numpy as np
from lauscher.abstract import Transformation
from lauscher.audiowaves import MonoAudioWave


class RmsNormalizer(Transformation):
    def __init__(self, level: float):
        super().__init__()
        self.level: float = level

    def __call__(self, data: MonoAudioWave) -> MonoAudioWave:
        assert isinstance(data, MonoAudioWave)

        data.samples *= self.level / np.sqrt(np.mean(data.samples ** 2))
        return data
