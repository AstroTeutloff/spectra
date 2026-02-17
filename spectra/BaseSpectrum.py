"""
Base class for spectra.

@author: Felix Teutloff
@date: 10-2025
@version: 0.1
"""

import abc
from scipy.interpolate import make_smoothing_spline, BSpline

class BaseSpectrum(metaclass=abc.ABCMeta):
    """
    Base Class for spectra.
    """

    @staticmethod
    def _fit_continuum(x, y, penalty) -> BSpline:
        """
        Perform a continuum fit by fitting a smoothing spline to the data.

        Parameters:
        -----------
            x: np.ndarray; 1d array of `x` values.
            y: np.ndarray; 1d array of `y` values.
            penalty: float; Penalty parameter that is used for smooting. A
                value of 0 perfectly constructs a spline to the data. As the
                value goes to infinity, a perfect constant is constructed. For
                details see implementation in scipy:
                https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
                A value of `None` makes the function optimize the value for it
                itself.

        Returns:
        --------
            spl: BSpline; A bspline object, a proxy for the spectral continuum.
        """

        return make_smoothing_spline(x, y, lam = penalty)

if __name__ == "__main__":
    print(__doc__)
