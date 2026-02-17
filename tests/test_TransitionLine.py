import numpy as np
import astropy.units as u
import unittest
import os
import sys
import spectra.TransitionLine as tl
from dataclasses import FrozenInstanceError

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


class TestTimeseries(unittest.TestCase):

    def test_rydberg(self):
        self.assertEqual(int(tl.rydberg(2, 3).value), 6561)

    def test_constructor(self):
        # Test construction
        self.assertTrue(self.__give_halpha())

        # Test for changing something after construction
        with self.assertRaises(FrozenInstanceError):
            self.__give_halpha().id = "other ID"

        # Test that inconsistent units are not allowed
        with self.assertRaises(u.UnitConversionError):
            tl.TransitionLine(
                "some ID",
                0 * u.meter,
                1 * u.second,
            )

        with self.assertRaises(TypeError):
            tl.TransitionLine(
                "some ID",
                0.,
                1 * u.meter
            )

    def test_covers(self):
        h_alpha = self.__give_halpha()

        self.assertFalse(h_alpha.covers(1 * u.parsec))
        self.assertTrue(h_alpha.covers(6550 * u.angstrom))

    def test_add(self):
        h_alpha = self.__give_halpha()

        self.assertNotEqual(h_alpha + 5 * u.angstrom, h_alpha)

        with self.assertRaises(u.UnitConversionError):
            h_alpha + 5 * u.second

    def __give_halpha(self):
        return tl.TransitionLine(
            "H_alpha",
            tl.rydberg(2, 3),
            30 * u.angstrom,
            1.
        )
