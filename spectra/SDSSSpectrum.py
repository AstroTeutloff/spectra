"""
Package for SDSS(-V) spectra.

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


class SDSSSpectrum():
    """
    Class that houses commonly used methods for analysing spectra.
    """

    WAVE_UNIT = u.angstrom
    FLUX_UNIT = u.erg * (u.s ** -1) * (u.cm ** -2) * (u.angstrom ** -1)

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

            # These have to be defined out of the loop, otherwise they get
            # overwritten.
            self.spectra = []
            self.spectra_sky = []
            self.spectra_header = []

            for i, hdu in enumerate(hdul):

                # skip primary HDU ( TODO: find out why)
                if i == 0:
                    continue

                header = hdu.header
                extname = header["EXTNAME"]

                if "SPALL" in extname:

                    self.spall = hdu.data

                    self.gaia_ids = self.spall.GAIA_ID
                    self.n_exp = self.spall.NEXP
                    self.t_exp_tot = self.spall.EXPTIME

                if "COADD" in extname:
                    # Setting up coadded spectrum
                    self.coadd_header = header
                    self.coadd, self.coadd_sky = SDSSSpectrum.__spec_from_hdu(
                        hdu)

                if "MJD_EXP" in extname:
                    # Setting up the constituent spectra
                    spec, sky = SDSSSpectrum.__spec_from_hdu(hdu)
                    self.spectra.append(spec)
                    self.spectra_sky.append(spec)
                    self.spectra_header.append(hdu.header)

    def plot_spectrum(
        self,
        ax: Axes | None = None,
        show_coadd: bool = True,
        show_individual: bool = False,
        show_uncertainty: bool = False,
        **plot_kwargs
    ) -> Axes:
        """
        Method for plotting the spectrum.

        Parameters:
        -----------
            ax: plt.Axes object; The plotting axis to use. If not declared in
            show_coadd: bool; Show the coadded spectrum.
            show_individual: bool; Show the individual spectra
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

        if show_coadd:
            self.coadd.plot(
                ax,
                label="COADD",
                show_uncertainty=show_uncertainty,
                **plot_kwargs
            )

        if show_individual:
            for header, spectrum in zip(self.spectra_header, self.spectra):
                spectrum.plot(
                    ax,
                    label=header["EXTNAME"],
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
