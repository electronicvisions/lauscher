from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Union, List
import numpy as np
from matplotlib.axes import Axes

from lauscher.helpers import Timer


class Transformable(metaclass=ABCMeta):
    """
    Classes that can be transformed with a :class:`Transformation`.
    """

    def transform(self, transformation: Transformation) -> Transformable:
        """
        Transform this instance by the given :class:`Transformation`.
        :param transformation: Transformation to be applied
        :return: Transformed instance
        """
        with Timer(transformation.__class__.__name__):
            return transformation(self)


class Transformation(metaclass=ABCMeta):
    """
    Class that can perform transformations on a given :class:`Transformable`.
    """

    @abstractmethod
    def __call__(self, input_data: Transformable) -> Transformable:
        """
        Execute the transformation.
        :param input_data: Transformable instance to be transformed
        :return: Transformation result
        """
        raise NotImplementedError


class Exportable(metaclass=ABCMeta):
    """
    Classes that can be exported.
    """

    @abstractmethod
    def export(self, path: str):
        """
        Export the data in this instance to a given path.
        :param path: Path the data exported to
        """
        raise NotImplementedError


class Plottable(metaclass=ABCMeta):
    """
    Classes that can be plotted.
    """

    @abstractmethod
    def plot(self, axis: Axes):
        """
        Plot the data contained in this instance.
        :param axis: Axis object to be plotted in
        """
        raise NotImplementedError


class SampledTimeSeries(Transformable, Plottable, metaclass=ABCMeta):
    """
    Time series data that uses an equally spaced sample grid.
    """

    def __init__(self, samples: Union[List[List[float]], np.ndarray],
                 sample_rate: int):
        self.channels = np.asarray(samples, dtype=np.double)
        self.sample_rate = sample_rate
        self.times = self.get_time(np.arange(self.num_samples))

    def get_time(self,
                 sample: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Calculate the time (in seconds) of a given (list of) sample point(s).
        :param sample: Sample point(s) time is calculated for
        :return: (List of) time values
        """
        return sample / self.sample_rate

    def get_sample_id(self, time: float) -> int:
        """
        Calculate the sample id of a given time (in seconds).
        :param time: Time the sample point is calculated for
        :return: Best-matching Sample id
        """
        return int(round(time * self.sample_rate))

    @property
    def num_samples(self):
        """
        :return: Number of samples in this instance.
        """
        return len(self.channels[0])

    @property
    def num_channels(self):
        """
        :return: Number of channels in this instance.
        """
        return len(self.channels)

    def plot(self, axis: Axes):
        for i in range(self.num_channels):
            axis.plot(self.times, self.channels[i], alpha=0.1)
