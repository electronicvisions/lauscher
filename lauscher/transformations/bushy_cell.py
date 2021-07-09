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
        dt = float(1.0/fs)

        spike_times = [] 
        vm = 0.0
        isyn = 0.0

        refrac_counter = 0
        n_refrac_samples = self.tau_refrac * fs

        scl_mem = np.exp(-dt / self.tau_mem)
        scl_syn = np.exp(-dt / self.tau_syn)
        for step in range(stimuli.shape[0]):
            if refrac_counter <= 0:
                if vm < 1.0:
                    new_vm = vm * scl_mem
                else: # emit spike
                    refrac_counter = n_refrac_samples
                    new_vm = 0.0
                    spike_times.append(step-1)
            new_isyn = isyn * scl_syn + stimuli[step]
            if refrac_counter <= 0:
                new_vm += new_isyn * self.weight * dt
            
            vm = new_vm
            refrac_counter -= 1
            isyn = new_isyn

        return spike_times

    def __call__(self, data: FiringProbability) -> SpikeTrain:
        assert isinstance(data, FiringProbability)
        np.random.seed(123) # TODO remove after optimization

        stimuli = []
        for i in range(data.num_channels):
            stimuli.append(self._sample(data.channels[i], data.sample_rate))
        renewal_spikes = np.sum(np.array(stimuli,dtype=np.bool),axis=2) # We can do this because we use the same weight for all inputs
        print(renewal_spikes.shape)
        print(renewal_spikes.dtype)

        with Pool(CommandLineArguments().num_concurrent_jobs) as workers:
            lif_spike_times = workers.map(partial(self._lif, fs=data.sample_rate),
                                       renewal_spikes)

        times = []
        units = []
        for i,ts in enumerate(lif_spike_times):
            if len(ts):
                times.extend(ts)
                units.append(i*np.ones(len(ts),dtype=int))
        times = np.array(times,dtype=np.float)/data.sample_rate
        units = np.concatenate(units)

        # Here we sort the spikes in time
        # TODO remove this sorting step when we have a clean vectorized implementation of _lif
        idx = np.argsort(times)
        times = times[idx]
        units = units[idx]

        return SpikeTrain(times,units)
