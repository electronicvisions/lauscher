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

        times = [] 
        units = [] 

        nb_cells = stimuli.shape[0]
        refrac_counter = np.zeros(nb_cells)
        vm = np.zeros(nb_cells)
        isyn = np.zeros(nb_cells)

        n_refrac_samples = self.tau_refrac * fs

        scl_mem = np.exp(-dt / self.tau_mem)
        scl_syn = np.exp(-dt / self.tau_syn)
        for step in range(stimuli.shape[1]):
            spiked = np.logical_and(vm>=1.0,refrac_counter<=0)
            new_vm = vm * scl_mem
            refrac_counter[spiked] = n_refrac_samples
            new_vm[spiked] = 0.0
            new_isyn = isyn * scl_syn + stimuli[:,step]
            active = (refrac_counter <= 0)
            new_vm[active] += new_isyn[active] * self.weight * dt
            
            vm = new_vm
            isyn = new_isyn
            refrac_counter -= 1

            ids = np.where(spiked)[0]
            if len(ids):
                units.append(ids)
                times.append(step/fs*np.ones(len(ids),dtype=np.int))

        times, units = np.concatenate(times),np.concatenate(units)
        return times, units 

    def __call__(self, data: FiringProbability) -> SpikeTrain:
        assert isinstance(data, FiringProbability)
        np.random.seed(123) # TODO remove after optimization

        stimuli = []
        for i in range(data.num_channels):
            stimuli.append(self._sample(data.channels[i], data.sample_rate))
        renewal_spikes = np.sum(np.array(stimuli,dtype=np.bool),axis=2) # We can do this because we use the same weight for all inputs

        times,units = self._lif(renewal_spikes, data.sample_rate)

        return SpikeTrain(times,units)
