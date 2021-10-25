from __future__ import annotations

from os import makedirs
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from lauscher.abstract import Transformable, Exportable, Plottable


class SpikeTrain(Transformable, Exportable, Plottable):
    def __init__(self, times: Union[None,list,numpy.ndarray] = None, units:Union[None,list,numpy.ndarray] = None):
        super().__init__()
        self._times = times
        self._units = units

    @property
    def spike_units(self):
        return self._units

    @property
    def spike_times(self):
        return self._times

    def export(self, path: str):
        makedirs(Path(path).parent, exist_ok=True)
        np.savez(path, times=self._times, units=self._units)

    def plot(self, axis: Axes):
        axis.plot(self.spike_times, self.spike_units,
                  ls="none", marker=".", color="black")

        axis.set_xlabel("Time")
        axis.set_ylabel("Label")

