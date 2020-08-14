from abc import ABCMeta, abstractmethod
from typing import Optional
import numpy as np

from lauscher.abstract import Transformation
from lauscher.audiowaves import MonoAudioWave


class Window(Transformation, metaclass=ABCMeta):
    def __init__(self, rampup_time: Optional[float] = None,
                 rampdown_time: Optional[float] = None):
        """
        :param rampup_time: Time in seconds from the start of the signal
                            during which the window function is applied.
                            Defaults to half the track.
        :param rampdown_time: Time in seconds to the end of the signal
                              during which the window function is applied.
                              Defaults to half the track.
        """
        super().__init__()
        self.rampup_time = rampup_time
        self.rampdown_time = rampdown_time

    def __call__(self, data: MonoAudioWave) -> MonoAudioWave:
        assert isinstance(data, MonoAudioWave)

        if self.rampup_time is None:
            rampup_samples = data.num_samples // 2
        else:
            rampup_samples = data.get_sample_id(self.rampup_time)

        if self.rampdown_time is None:
            rampdown_samples = data.num_samples // 2
        else:
            rampdown_samples = data.get_sample_id(self.rampdown_time)

        rampup_window = self._window(rampup_samples * 2)[:rampup_samples]
        rampdown_window = self._window(rampdown_samples * 2)[:rampdown_samples]

        data.samples[:rampup_samples] = rampup_window \
                                        * data.samples[:rampup_samples]
        data.samples[-rampdown_samples:] = rampdown_window[::-1] * \
                                           data.samples[-rampdown_samples:]
        return data

    @abstractmethod
    def _window(self, num_samples: int):
        raise NotImplementedError


class HanningWindow(Window):
    def _window(self, num_samples: int):
        return np.hanning(num_samples)
