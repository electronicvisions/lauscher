from __future__ import annotations

from functools import partial
from multiprocessing import Pool
import numpy as np
import numba

from lauscher.abstract import Transformation
from lauscher.firing_probability import FiringProbability
from lauscher.helpers import CommandLineArguments
from lauscher.spike_train import SpikeTrain


class BushyCell(Transformation):
    # Model parameters have well-defined short names
    # pylint: disable=invalid-name

    def __init__(self,
                 n_convergence: int = 40,
                 tau_mem: float = 1e-3,
                 tau_syn: float = 5e-4,
                 tau_refrac: float = 1e-3,
                 weight: float = 13e3):
        # Signature is given by model parameters
        # pylint: disable=too-many-arguments

        super().__init__()
        self.n_convergence = n_convergence
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.tau_refrac = tau_refrac
        self.weight = weight / float(self.n_convergence)

    @staticmethod
    @numba.jit(nopython=True)
    def _correct(spikes, refrac_samples):
        for i in range(spikes.shape[1]):
            last = -refrac_samples
            for j in range(spikes.shape[0]):
                if spikes[j, i]:
                    if j - last < refrac_samples:
                        spikes[j, i] = 0
                    else:
                        last = j
        return spikes

    def _sample(self, stimulus, fs):
        spikes = np.random.rand(stimulus.size,
                                self.n_convergence) < stimulus[:, None]
        return self._correct(spikes, self.tau_refrac * fs)

    def _lif(self, stimuli, fs):
        dt = 1  / float(fs)

        vm = np.zeros(stimuli.shape[0])
        spikes = np.zeros(stimuli.shape[0])
        isyn = np.zeros(stimuli.shape)

        refrac_counter = 0
        n_refrac_samples = self.tau_refrac * fs

        for step in range(stimuli.shape[0]):
            if step > 0:
                if refrac_counter <= 0:
                    if vm[step - 1] < 1.0:
                        vm[step] = vm[step - 1] * np.exp(-dt / self.tau_mem)
                    else:
                        refrac_counter = n_refrac_samples
                        vm[step] = 0.0
                        spikes[step - 1] = 1.0
                isyn[step, :] = isyn[step - 1] * np.exp(-dt / self.tau_syn)
            isyn[step, :] += stimuli[step, :]
            if refrac_counter <= 0:
                vm[step] += np.sum(isyn[step, :]) * self.weight * dt
            if vm[step] > 1.0:
                vm[step] = 1.0

            refrac_counter -= 1

        return spikes

    def __call__(self, data: FiringProbability) -> SpikeTrain:
        assert isinstance(data, FiringProbability)

        stimuli = np.ndarray((data.num_channels, data.num_samples,
                              self.n_convergence))
        for i in range(data.num_channels):
            stimuli[i] = self._sample(data.channels[i], data.sample_rate)

        with Pool(CommandLineArguments().num_concurrent_jobs) as workers:
            spike_matrix = workers.map(partial(self._lif, fs=data.sample_rate),
                                       stimuli)

        return SpikeTrain.from_dense(np.array(spike_matrix), data.sample_rate)
