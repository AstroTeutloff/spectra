"""
Dataclass for storing spectra.

@author: Felix Teutloff
@date: 02-2026
@version: 0.1
"""

from __future__ import annotations
from dataclasses import dataclass
from astropy import units as u, constants as c
from astropy.table import QTable
from matplotlib.axes import Axes
import numpy as np
from warnings import warn


@dataclass(frozen=True)
class Spectrum:
    """
    Dataclass that represents a single spectrum.
    This spectrum is immutable. You cannot set fields after constructing.

    Available fields are:
        - wave: u.Quantity; The wavelength space for the spectrum
        - flux: u.Quantity; The flux at each wavelength bin
        - flux_err: u.Quantity | None; The flux uncertainty/error at each
            wavelength bin
    """

    wave: u.Quantity
    flux: u.Quantity
    flux_err: u.Quantity | None = None

    def __post_init__(self):
        if (
            not isinstance(self.wave, u.Quantity) or
            not isinstance(self.flux, u.Quantity)
        ):
            raise TypeError("wave or flux not of type u.Quantity!")

        if len(self.wave) != len(self.flux):
            raise ValueError("Lengths of wave and flux do not match!")

        # Return early if flux_err is none.
        if self.flux_err is None:
            return

        if len(self.flux) != len(self.flux_err):
            raise ValueError("Lengths of flux and flux_err do not match!")

        try:
            _ = self.flux_err.to(self.flux.unit)
        except u.UnitConversionError as uce:
            raise u.UnitConversionError(
                "Units of flux and flux_err not compatible!"
            ) from uce

    def __add__(self, other: Spectrum) -> Spectrum:
        """
        Adds the flux of two spectra together to make a new `coadded` spectrum.

        NOTE:
        -----
            As of version 0.1, this only interpolates the flux valus of addend
            1 at the wavelength values of addend 2. This will only be a
            reasonable approximation if both spectra are nearly the same
            resolution!
        """
        warn(
            "This currently does not convolve the spectra! Only use it for " +
            "similarly resolved spectra!"
        )

        # Interpolate values down from higher to lower res spectrum
        flux_interp = np.interp(self.wave, other.wave, other.flux)
        fluxerr_interp = np.interp(self.wave, other.wave, other.flux_err)

        # Add the errors in quadrature if both exist, otherwise just make error None.
        if self.flux_err is not None and other.flux_err is not None:
            flux_err_quadrature = np.sqrt(
                self.flux_err ** 2 + fluxerr_interp ** 2
            )
        else:
            flux_err_quadrature = None

        return Spectrum(
            self.wave,
            self.flux + flux_interp,
            flux_err_quadrature
        )

    def __mul__(self, factor: float) -> Spectrum:
        """
        Multiplies the flux of the spectrum by a floating point number.
        """
        return Spectrum(
            self.wave,
            self.flux * factor,
            self.flux_err * factor if self.flux_err is not None else None
        )

    def __rmul__(self, factor: float) -> Spectrum:
        """
        Allow commutativity in multiplication with scalar factor.
        """
        return self.__mul__(factor)

    def redshift(self, velocity: u.Quantity) -> Spectrum:
        """
        Redshifts the spectrum wavelengths.

        Parameters:
        -----------
            velocity: u.Quantity; Radial velocity of the source

        Returns:
        --------
            Spectrum; Spectrum in which the wavelengths are shifted by a radial
            velocity factor.
        """
        redshift_factor = 1 + velocity/c.c
        redshifted_wave = self.wave * redshift_factor

        return Spectrum(
            redshifted_wave,
            self.flux,
            self.flux_err
        )

    def plot(
        self,
        ax: Axes,
        label: str | None = None,
        show_uncertainty: bool = True,
        **plot_kwargs
    ) -> None:
        yerr = self.flux_err if show_uncertainty else None

        ax.errorbar(
            self.wave,
            self.flux,
            yerr=yerr,
            label=label,
            **plot_kwargs
        )

    def write(
        self,
        **write_kwargs
    ) -> None:
        """
        Writes the spectrum to a file, using the astropy table writer(s).

        Parameters:
        -----------
            **write_kwargs: all keyword arguments passed to this function get
                passed to the astropy Table.write() method.
        """

        if self.flux_err is not None:
            QTable(
                [self.wave, self.flux, self.flux_err],
                names=["wave", "flux", "flux_err"]
            ).write(**write_kwargs)

        else:
            QTable(
                [self.wave, self.flux],
                names=["wave", "flux"]
            ).write(**write_kwargs)

        return None
