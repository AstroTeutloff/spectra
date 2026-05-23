"""
Package for PYPEIT 1D spectra.

@author: Felix Teutloff
@date: 10-2025
@version: 0.1.0
"""

from os.path import isfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from astropy import units as u
from astropy.io import fits

from spectra.Spectrum import Spectrum


class PYPEITSpectrum():
    """
    Class that houses commonly used methods for analysing spectra.
    """

    WAVE_UNIT = u.angstrom
    FLUX_UNIT = u.dimensionless_unscaled  # Counts

    def __init__(
        self,
        filename: str,
    ):
        """
        Constructor for a SDSSSpectrum.

        Parameters:
        -----------

            filename: str; Fits file that contains the spectral data.
        """

        if not isfile(filename):
            raise FileNotFoundError(f"File `{filename}` does not exist!")

        with fits.open(filename) as hdul:
            self.header = hdul[0].header
            wave = hdul[1].data.OPT_WAVE
            flux = hdul[1].data.OPT_COUNTS
            flux_err = hdul[1].data.OPT_COUNTS_SIG
            self.spectrum = Spectrum(
                wave * self.WAVE_UNIT,
                flux * self.FLUX_UNIT,
                flux_err * self.FLUX_UNIT
            )

            sky_wave = hdul[1].data.OPT_WAVE
            sky_flux = hdul[1].data.OPT_COUNTS
            sky_flux_err = hdul[1].data.OPT_COUNTS_SIG
            self.spectrum_sky = Spectrum(
                sky_wave * self.WAVE_UNIT,
                sky_flux * self.FLUX_UNIT,
                sky_flux_err * self.FLUX_UNIT
            )

    def plot_spectrum(
        self,
        ax: Axes | None = None,
        show_uncertainty: bool = False,
        **plot_kwargs
    ) -> Axes:
        """
        Method for plotting the spectrum.

        Parameters:
        -----------
            ax: plt.Axes object; The plotting axis to use. If not declared in
            show_uncertainty: bool; Show show uncertainty bars for flux.
            plot_kwargs; Further keywords are passed to the call of plt.plot as
                keyword arguments

        Returns:
        --------
            Axes; The axes object that was either put in, or created for
                the plot.

        """

        if ax is None:
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(111)

            self.spectrum.plot(
                ax,
                show_uncertainty=show_uncertainty,
                **plot_kwargs
            )

        ax.set_xlabel(r"Wavelength $\lambda$ [$\AA$]")
        ax.set_ylabel(
            r"Flux $F_{\lambda}$ [$10^{-17} \mathrm{erg/s/cm^2/\AA}$]"
        )

        return ax

    @staticmethod
    def __spec_from_hdu(hdu) -> tuple[Spectrum, Spectrum]:
        """
        Private method to give spectral data back, organized and nicely with
        units.

        Parameters:
        -----------
            hdu: Fits header-data unit

        Returns:
        --------
            tuple[Spectrum, Spectrum]; The science and the sky spectrum.
        """
        data = hdu.data

        loglam = data.LOGLAM
        ivar = data.IVAR

        # Masking and setting up units. With new astropy versions not necessary
        # anymore.
        wave = 10**(loglam) * SDSSSpectrum.WAVE_UNIT
        flux = data.FLUX * SDSSSpectrum.FLUX_UNIT
        fsky = data.SKY * SDSSSpectrum.FLUX_UNIT
        eflux = np.sqrt(1./ivar) * SDSSSpectrum.FLUX_UNIT

        spectrum = Spectrum(
            wave,
            flux,
            eflux
        )

        sky = Spectrum(
            wave,
            fsky,
        )

        return (spectrum, sky)


if __name__ == "__main__":
    print(__doc__)
