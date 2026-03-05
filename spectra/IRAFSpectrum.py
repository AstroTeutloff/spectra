"""
Package for IRAF spectra.

@author: Felix Teutloff
@date: 02-2026
@version: 0.1.0
"""

from warnings import warn
from os.path import isfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from astropy import units as u
from astropy.io import fits

from spectra.Spectrum import Spectrum


class IRAFSpectrum():
    """
    Class that houses commonly used methods for analysing spectra.
    """

    WAVE_UNIT = u.angstrom
    FLUX_UNIT = u.dimensionless_unscaled

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
            if len(hdul) > 1:
                warn(
                    "HDUL has more than 1 element. " +
                    "Only first one will be accessed."
                )

            self.spectrum, self.spectrum_sky = self.__spec_from_hdu(hdul[0])
            self.header = hdul[0].header

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
            label=None,
            show_uncertainty=show_uncertainty,
            **plot_kwargs
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
            spectrum, sky; tuple[Spectrum, Spectrum]; Science and sky spectra
        """
        data = hdu.data
        header = hdu.header

        wave_offset = header["CRVAL1"]
        wave_resolution = IRAFSpectrum.__get_resolution(header)
        pixels_specaxis = header["NAXIS1"]
        pixels_offset = header["CRPIX1"]

        wave_space = wave_offset + wave_resolution * \
            (np.arange(0, pixels_specaxis, 1) - pixels_offset)

        flux_space = IRAFSpectrum.__get_flux(data, pixels_specaxis)
        # Masking and setting up units. With new astropy versions not necessary
        # anymore.

        wave = wave_space * IRAFSpectrum.WAVE_UNIT
        flux = flux_space[0] * IRAFSpectrum.FLUX_UNIT
        flux_sky = flux_space[1] * IRAFSpectrum.FLUX_UNIT

        spectrum = Spectrum(
            wave,
            flux
        )

        sky = Spectrum(
            wave,
            flux_sky
        )

        return (spectrum, sky)

    @staticmethod
    def __get_resolution(header) -> float:
        possible_keywords = ["CRDELTA1", "CD1_1"]
        resolution = None
        try:
            while resolution is None:
                resolution = header.get(possible_keywords.pop(0), None)
        except IndexError as ie:
            raise IndexError(
                "Non of the known keywords found a resolution value."
            ) from ie

        return resolution

    @staticmethod
    def __get_flux(data, spec_size) -> list[np.ndarray]:
        spectra = []
        if len(data) == spec_size:
            return data

        for individual in data:
            # unwrap if is wrapped
            while len(individual) == 1:
                individual = individual[0]

            if (lenindiv := len(individual)) != spec_size:
                raise IndexError(
                    "Length of individual spectrum is mismatched with " +
                    "expected spec length " +
                    f"(expected: {spec_size}, got: {lenindiv})")

            spectra.append(individual)

        return spectra


if __name__ == "__main__":
    print(__doc__)
