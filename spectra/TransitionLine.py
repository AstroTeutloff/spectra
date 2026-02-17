"""
Dataclass for storing Transition lines.

@author: Felix Teutloff
@date: 02-2026
@version: 0.1
"""

from __future__ import annotations
from dataclasses import dataclass
from astropy import units as u, constants as c
from astropy.modeling.models import Voigt1D
from matplotlib.axes import Axes
import numpy as np


def rydberg(base: np.int64, to: np.int64) -> u.Quantity:
    """
    Returns transition wavelength between two states.

    Parameters:
    -----------
        base: np.int64; Base state
        to: np.int64; Final state
        Both can be np.inf to compute ionisation wavelength.

    Returns:
    --------
        wave_transit: u.Quantity; The transition wavelength.

    NOTE:
    -----
        The values calculated are not the same as the ones from Wikipedia. I
            assume, this is because of the small difference in the Rydberg
            constant.
    """
    if base == to:
        return 0 * u.angstrom

    state_difference = (base ** (-2) - to**(-2))
    one_over_lambda = c.Ryd * state_difference
    return (one_over_lambda**(-1)).to(u.angstrom)


@dataclass(frozen=True)
class TransitionLine:
    """
    Dataclass that combines transition line data.
    Transition line parameters are immutable. You cannot set field values after
    constructing.
    Encoded are:
        - id: str; An ID, e.g. H_α (for balmer3-2)
        - wave_0: u.Quantity; A transition wavelength
        - wave_delta: u.Quantity; A line width
        - line_depth: float = 0.; A line depth
    """
    id: str
    wave_0: u.Quantity
    wave_delta: u.Quantity
    line_depth: float = 0.

    def __post_init__(self):
        if (
            not isinstance(self.wave_0, u.Quantity) or
            not isinstance(self.wave_delta, u.Quantity)
        ):
            raise TypeError("wl_transit or wl_width not of type u.Quantity!")

        try:
            self.wave_0.to(self.wave_delta.unit)
        except u.UnitConversionError as uce:
            raise u.UnitConversionError(
                "Units of wl_transit and wl_width are not compatible!"
            ) from uce

    def covers(self, wavelength: u.Quantity):
        """
        Checks if a given wavelength is `inside` the transition line regime.
        Calculation is done by |wl-wl_transit| < wl_width.

        Parameters:
        -----------
            wavelength: u.Quantity; A length Quantity. If it is not of
                comparable units to fields of TransitionLine, it will
                throw an Error.
        Returns:
        --------
            bool: Whether or not the given length is inside the transition
                regime.
        """

        return abs(wavelength - self.wave_0) < self.wave_delta

    def __add__(
        self,
        wavelength: u.Quantity
    ) -> TransitionLine:
        """
        Shifts the transition line by a certain value.

        Parameters:
        -----------
            wavelength: u.Quantity; The amount, by which the line is shifted.
            This value has to be comparable to wl_transit, otherwise it
            will raise an Error.

        Returns:
        --------
            TransitionLine; New TransitionLine object with shifted transition
                wavelength, but same other values.
        """

        new_tl = TransitionLine(
            self.id,
            self.wave_0 + wavelength,
            self.wave_delta,
            self.line_depth
        )
        return new_tl

    def fit_voigt(
        self,
        x_data: u.Quantity,
        y_data: u.Quantity,
        y_err: u.Quantity | None = None,
        maxiter: int = 200
    ) -> Voigt1D:
        """
        Fits a 1D Voigt profile to x and ydata.
        Priors for the line data are taken from the classes' fields.

        Parameters:
        -----------
            x_data: u.Quantity; The data, that makes up the x values.
            y_data: u.Quantity; The data, that makes up the y values.
            y_err: Error values for the y data. Is used as weights in the line
                fit through `weight = 1/y_err`. If not declared, fit will not
                be weighted.
            maxiter: int; How many iterations of the fit should be performed.

        Returns:
        --------
            fitted_voigt: Voigt1D; The parameters for the best fit to the data.
        """
        raise NotImplementedError

    def plot(self, ax: Axes):
        raise NotImplementedError
