from typing import List

import numpy as np
from matplotlib.axes import Axes
import soundfile

from lauscher.abstract import SampledTimeSeries, Exportable


class MonoAudioWave(SampledTimeSeries, Exportable):
    def __init__(self, samples: List[float], sample_rate: int):
        super().__init__([samples], sample_rate)

    @property
    def samples(self):
        return self.channels[0]

    @samples.setter
    def samples(self, values: np.ndarray):
        self.channels[0] = values

    def get_rms(self) -> float:
        """
        Compute the RMS value of the given track relative to 0dBfs
        """
        rms_float = np.sqrt(np.mean(np.square(self.samples)))
        rms_db = np.log10(rms_float) * 20
        return rms_db

    def export(self, path: str):
        soundfile.write(path, self.samples, self.sample_rate)

    def plot(self, axis: Axes):
        axis.plot(self.times, self.samples)


class FileMonoAudioWave(MonoAudioWave):
    def __init__(self, file_path: str):
        self._soundfile: soundfile.SoundFile = soundfile.SoundFile(file_path)
        if self._soundfile.channels > 1:
            raise RuntimeError("Only mono audio files are supported!")

        super().__init__(self.raw_amplitudes, self._soundfile.samplerate)

    @property
    def raw_amplitudes(self) -> np.array:
        self._soundfile.seek(0)
        return self._soundfile.read()
