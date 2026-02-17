"""
Package for SDSS(-V) spectra.

@author: Felix Teutloff
@date: 10-2025
@version: 0.0.1
"""

from os.path import isfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

from astropy import units as u
from astropy.io import fits
from astropy.table import QTable
from astropy.utils.masked import Masked

from simplespec.BaseSpectrum import BaseSpectrum

class SDSSSpectrum(BaseSpectrum):
    """
    Class that houses commonly used methods for analysing spectra.
    """

    WAVE_UNIT = u.angstrom
    FLUX_UNIT = u.erg * (u.s **-1) * (u.cm **-2) * (u.angstrom **-1)

    def __init__(
        self,
        filename : str,
        mask_blue: int| None = 100,
        mask_red: int| None = 100
    ):
        """
        Constructor for a SDSSSpectrum.

        Parameters:
        -----------
            
            filename: str; Fits file that contains the spectral data.
            mask_blue: int|None; Mask n points on blue end of spectrum. Default
                is 100.
            mask_red: int|None; Mask n points on red end of spectrum. Default
                is 100.

        """

        if not isfile(filename):
            raise FileNotFoundError(f"File `{filename}` does not exist!")

        with fits.open(filename) as hdul:

            # These have to be defined out of the loop, otherwise they get
            # overwritten.
            self.spectra = []
            self.spectra_header = []

            for i, hdu in enumerate(hdul):

                #skip primary HDU ( TODO: find out why)
                if i == 0: continue

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
                    self.coadd = SDSSSpectrum.__spec_from_hdu(hdu)

                    # Masking blue and red end, because they are very noisy.
                    if isinstance(mask_blue, int) and mask_blue > 0:
                        self.coadd["lambda"][:mask_blue] = np.ma.masked
                    if isinstance(mask_red, int) and mask_red > 0:
                        self.coadd["lambda"][-mask_red:] = np.ma.masked

                if "MJD_EXP" in extname:
                    # Setting up the constituent spectra
                    self.spectra.append(SDSSSpectrum.__spec_from_hdu(hdu))
                    self.spectra_header.append(hdu.header)

                    # Masking blue and red end, because they are very noisy.
                    # (In the individual spectra too.)
                    for spectrum in self.spectra:
                        if isinstance(mask_blue, int) and mask_blue > 0:
                            spectrum["lambda"][:mask_blue] = np.ma.masked
                        if isinstance(mask_red, int) and mask_red > 0:
                            spectrum["lambda"][-mask_red:] = np.ma.masked


    def plot_spectrum(
        self,
        ax: plt.Axes = None,
        show_coadd: bool = True,
        show_individual: bool = False,
        show_uncertainty: bool = False,
        **plot_kwargs
    ) -> plt.Axes:
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
            plt.Axes; The axes object that was either put in, or created for
                the plot.

        """

        if ax is None:
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(111)

        if show_coadd:
            # Plotting coadded spectrum
            yerr = self.coadd["phi_err"] if show_uncertainty else None

            ax.errorbar(
                self.coadd["lambda"],
                self.coadd["phi"],
                yerr = yerr,
                label = "COADD",
                **plot_kwargs
            )

        if show_individual:
            for header, spectrum in zip(self.spectra_header, self.spectra):
                yerr = spectrum["phi_err"] if show_uncertainty else None

                ax.errorbar(
                    spectrum["lambda"],
                    spectrum["phi"],
                    yerr = yerr,
                    label = header["EXTNAME"],
                    **plot_kwargs
                )

        ax.set_xlabel(r"Wavelength $\lambda$ [$\AA$]")
        ax.set_ylabel(
            r"Flux $F_{\lambda}$ [$10^{-17} \mathrm{erg/s/cm^2/\AA}$]"
        )

        return ax


    @classmethod
    def fit_continuum(
        cls,
        spectrum:QTable,
        penalty:float
    ) -> BSpline:
        """
        Perform a continuum fit on a specific spectrum (i.e. coadd or
        individual).

        Parameters:
        -----------
            spectrum: QTable; The spectrum that is to be fitted to.
            penalty: float; Penalty value for the spline fit. A lower value
            fits the data closer.

        Returns:
        --------
            BSpline: The fitted spline.

        TODO:
        -----
            Include a linelist to cut out from the continuum before fitting.
        """
        return super()._fit_continuum(
            spectrum["lambda"], spectrum["phi"], penalty
        )


    @staticmethod
    def __spec_from_hdu(hdu) -> QTable:
        """
        Private method to give spectral data back, organized and nicely with
        units.

        Parameters:
        -----------
            hdu: Fits header-data unit

        Returns:
        --------
            QTable: Table of spectrum data.
        """
        data = hdu.data

        loglam = data.LOGLAM
        ivar = data.IVAR

        # Masking and setting up units. With new astropy versions not necessary
        # anymore.
        wave =  Masked(10**(loglam) * SDSSSpectrum.WAVE_UNIT, mask = len(data) * [False])
        flux =  Masked(data.FLUX * SDSSSpectrum.FLUX_UNIT, mask = len(data) * [False])
        fsky =  Masked(data.SKY * SDSSSpectrum.FLUX_UNIT, mask = len(data) * [False])
        eflux = Masked(np.sqrt(1./ivar) * SDSSSpectrum.FLUX_UNIT, mask = len(data) * [False])

        data_table = QTable(
            [wave, flux, eflux, fsky],
            names = ["lambda", "phi", "phi_err", "phi_sky"],
            masked = True
        )

        return data_table


if __name__ == "__main__":
    print(__doc__)
