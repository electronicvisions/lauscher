from functools import partial
from multiprocessing import Pool
import numpy as np
from scipy import special

from lauscher.abstract import Transformation
from lauscher.audiowaves import MonoAudioWave
from lauscher.helpers import CommandLineArguments
from lauscher.membranevelocity import MembraneVelocity


class BasilarMembrane(Transformation):
    # Model parameters have well-defined short names
    # pylint: disable=invalid-name

    # The best readable equation indentation differs from PEP8
    # pylint: disable=bad-continuation

    def __init__(self,
                 channels: int = 700,
                 a: int = 3500,
                 alpha: float = 3.0,
                 rho: float = 1.0,
                 c: float = 3.5,
                 c0: float = 10e8,
                 de: float = 0.15,
                 h: float = 0.1,
                 m: float = 0.05):
        # Signature is given by model parameters
        # pylint: disable=too-many-arguments

        super().__init__()
        self.channels = channels
        self.ch = np.linspace(0, c, self.channels)
        self.a = a
        self.alpha = alpha
        self.rho = rho
        self.c0 = c0
        self.de = de
        self.h = h
        self.m = m
        self.r = self.de * np.sqrt(self.c0 * self.m)

    def _s(self, c):
        return self.c0 * np.exp(-self.alpha * c) - self.a

    def _r(self, c):
        return self.r * np.exp(-self.alpha * c / 2.)

    def _xi(self, omega, c):
        return self._s(c) - self.m * omega ** 2 - 1j * omega * self._r(c)

    def _g(self, omega, c):
        return omega * np.sqrt(self.rho / (self.h * self._xi(omega, c)))

    def _p(self, omega, c):
        G = 2. * self._g(omega, 0.0) / self.alpha - 1 \
                / (self.alpha * (np.sqrt(self.h * 2.
                                 * (self.m * omega ** 2 + self.a)))) \
                * (2. * 1j * np.sqrt(2.) * omega
                * (np.log((-self.alpha * np.sqrt(-self.a +self.c0 - omega
                        * (self.m * omega +1j * self.r))
                + 1j * self.alpha
                * (2. * self.a + omega
                   * (2. * self.m * omega + 1j * self.r))
                / (2. * np.sqrt(self.a + self.m * omega ** 2)))
                / (0.5 * self.alpha * np.exp(self.alpha * c / 2.)
                   * (-2. * np.sqrt(-self.a
                                    + self.c0 * np.exp(-self.alpha * c)
                                    - self.m * omega ** 2
                                    - 1j * np.exp(-self.alpha * c / 2.)
                                         * omega * self.r)
                     + 1j * (2. * self.a+omega
                             * (2. * self.m * omega
                                + 1j * np.exp(-self.alpha * c / 2.)
                               * self.r))
                     / np.sqrt(self.a + self.m * omega ** 2))))))
        return np.sqrt(G / self._g(omega, c)) * special.hankel1(0, G)

    def _v(self, omega, c):
        return 2. * omega / self._xi(omega, c) * self._p(omega, c) \
                * self._z(omega) / self._p(omega, 0.0)

    def _z(self, omega):
        f = omega / (2. * np.pi)
        return np.sqrt(2. * self.c0 / self.h) \
                * (1j * special.j0(4. * np.pi * f / self.alpha
                                   * np.sqrt(2. / (self.h * self.c0)))
                + special.y0(4. * np.pi * f / self.alpha
                             * np.sqrt(2 / (self.h * self.c0)))) \
                / (special.j1(4. * np.pi * f / self.alpha
                              * np.sqrt(2. / (self.h * self.c0)))
                - 1j * special.y1(4. * np.pi * f / self.alpha
                                  * np.sqrt(2. / (self.h * self.c0))))

    def process_single_channel(self, c, x, stim_fft):
        return np.real(np.fft.ifft(self._v(x, c) * stim_fft))

    def __call__(self, data: MonoAudioWave) -> MembraneVelocity:
        """
        Calculate hydrodynamic shallow water basilar membrane response

        References:
        Sieroka, N., Dosch, H.G., Rupp, A. (July 2006). Semirealistic models of
        the chochlea. Acoustical society of America 120(1) 297
        """
        assert isinstance(data, MonoAudioWave)

        nsamples = data.num_samples // 2 + 1
        sample_rate = data.sample_rate // 2
        x = 2 * np.pi * np.linspace(1, sample_rate, nsamples)

        # calculate fft of stimulus
        stim_fft = np.fft.fft(data.samples)[:nsamples]

        # apply model matrix
        with Pool(CommandLineArguments().num_concurrent_jobs) as workers:
            samples = workers.map(partial(self.process_single_channel,
                                          x=x,
                                          stim_fft=stim_fft),
                                  self.ch)

        return MembraneVelocity(np.array(samples), sample_rate)
