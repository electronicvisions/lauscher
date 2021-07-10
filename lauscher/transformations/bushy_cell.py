from __future__ import annotations

from functools import partial
from multiprocessing import Pool
import numpy as np
import numba
from numpy.random import SeedSequence, default_rng

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
        # Alternative attempt, but slower 
        # for t,j in np.argwhere(spikes):
        #     end = min(t+refrac_samples,spikes.shape[0]-1)
        #     spikes[t+1:end,j] = 0
        return spikes

    def _sample(self, data, channel, rngs=None):
        stimulus = data.channels[channel]
        if rngs is None:
            spikes = np.random.rand(stimulus.size, self.n_convergence) < stimulus[:, None]
        else:
            spikes = rngs[channel].random((stimulus.size, self.n_convergence)) < stimulus[:, None]
        return np.sum(self._correct(spikes, self.tau_refrac * data.sample_rate), axis=1, dtype=np.float32)

    def _lif(self, stimuli, fs, indices=None):
        dt = float(1.0/fs)

        if indices is None:
            indices = np.arange(len(stimuli))
        stim = stimuli[indices]
        
        nb_cells = stim.shape[0]
        refrac_counter = np.zeros(nb_cells)
        vm = np.zeros(nb_cells)
        isyn = np.zeros(nb_cells)

        n_refrac_samples = self.tau_refrac * fs

        scl_mem = np.exp(-dt / self.tau_mem)
        scl_syn = np.exp(-dt / self.tau_syn)

        times = [] 
        units = [] 
        for step in range(stim.shape[1]):
            spiked = np.logical_and(vm>=1.0,refrac_counter<=0)
            new_vm = vm * scl_mem
            refrac_counter[spiked] = n_refrac_samples
            new_vm[spiked] = 0.0
            new_isyn = isyn * scl_syn + stim[:,step]
            active = (refrac_counter <= 0)
            new_vm[active] += new_isyn[active] * self.weight * dt
            
            vm = new_vm
            isyn = new_isyn
            refrac_counter -= 1

            ids = np.where(spiked)[0]
            if len(ids):
                units.append( indices[ids] )
                times.append(step/fs*np.ones(len(ids),dtype=np.int))

        times, units = np.concatenate(times), np.concatenate(units)
        return times, units 

    def __call__(self, data: FiringProbability) -> SpikeTrain:
        assert isinstance(data, FiringProbability)
        np.random.seed(123) # TODO remove after optimization

        # Simulate renewal processes
        compat_mode = False
        if compat_mode:
            renewal_spikes = np.empty(data.channels.shape, dtype=np.int)
            for i in range(data.num_channels):
                renewal_spikes[i] = self._sample(data, i)
        else:
            # using the parallel strategy here yields different results due to random number generation
            ss = SeedSequence(np.random.randint(1e9))
            child_seeds = ss.spawn(data.num_channels)
            random_streams = [default_rng(s) for s in child_seeds]
            with Pool(CommandLineArguments().num_concurrent_jobs) as workers:
                renewal_spikes = workers.map(partial(self._sample, data, rngs=random_streams), np.arange(data.num_channels) )
            renewal_spikes = np.array(renewal_spikes)
     
        # Simulate LIF dynamics
        chunk_size=100 # TODO tune this empirical parameter 
        if data.num_channels>chunk_size:
            print("Using chunked strategy")
            # Split work in chunks
            chunks = np.array_split(np.arange(data.num_channels), data.num_channels//chunk_size)
            with Pool(CommandLineArguments().num_concurrent_jobs) as workers:
                results = workers.map(partial(self._lif, renewal_spikes, data.sample_rate), chunks)

            # combine results
            times = np.concatenate([ ts for ts,us in results ])
            units = np.concatenate([ us for ts,us in results ])

            # sort spikes in time
            idx = np.argsort(times)
            times = times[idx]
            units = units[idx]
        else:
            times,units = self._lif(renewal_spikes, data.sample_rate)

        return SpikeTrain(times,units)
