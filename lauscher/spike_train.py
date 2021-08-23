from __future__ import annotations

from os import makedirs
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from lauscher.abstract import Transformable, Exportable, Plottable


class SpikeTrain(Transformable, Exportable, Plottable):
    def __init__(self):
        super().__init__()
        self._data = NotImplemented

    @property
    def spike_labels(self):
        return self._data[1]

    @property
    def spike_times(self):
        return self._data[0]

    def export(self, path: str):
        makedirs(Path(path).parent, exist_ok=True)
        np.savez(path, self._data)

    def plot(self, axis: Axes):
        axis.plot(self.spike_times, self.spike_labels,
                  ls="none", marker=".", color="black")

        axis.set_xlabel("Time")
        axis.set_ylabel("Label")

    @classmethod
    def from_dense(cls, channel_time_matrix: np.ndarray,
                   sample_rate: int) -> SpikeTrain:
        spikes = np.array(np.where(channel_time_matrix.T), dtype=np.double)
        spikes[0, :] = spikes[0, :] / sample_rate

        result = cls()
        result._data = spikes

        return result
