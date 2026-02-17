from astropy import units as u, constants as c
import unittest
import os
import sys
from spectra.Spectrum import Spectrum
from dataclasses import FrozenInstanceError
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


class TestSpectrum(unittest.TestCase):

    def __dummy_spectrum(self):
        return Spectrum(
            np.arange(3000, 6000) * u.angstrom,
            np.arange(3000, 6000) * u.electron,
            np.arange(3000, 6000) * u.electron,
        )

    def test_constructor(self):
        # Happy path
        self.assertTrue(self.__dummy_spectrum())

        # Try to change a field after the fact
        with self.assertRaises(FrozenInstanceError):
            self.__dummy_spectrum().wave = 0. * u.meter

        # Create some failing instances
        with self.assertRaises(TypeError):
            Spectrum(0., 0. * u.meter)
        with self.assertRaises(TypeError):
            Spectrum(0. * u.meter, 0.)

        with self.assertRaises(ValueError):
            Spectrum([0]*u.meter, [0, 1] * u.meter)
        with self.assertRaises(ValueError):
            Spectrum([0]*u.meter, [0]*u.meter, [0, 1] * u.meter)

        with self.assertRaises(u.UnitConversionError):
            Spectrum([0] * u.meter, [0]*u.meter, [1]*u.electron)

    def test_add(self):
        dummy_spectrum = self.__dummy_spectrum()
        added_spectrum = dummy_spectrum + dummy_spectrum
        # TODO: Do more specific test!

    def test_mul(self):
        dummy_spectrum = self.__dummy_spectrum()
        multiplied_spectrum = dummy_spectrum * 2
        # TODO: Do more specific test!

    def test_redshift(self):
        dummy_spectrum = self.__dummy_spectrum()
        dummy_spectrum.redshift(c.c)
        # TODO: Do more specific test!
