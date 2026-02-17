# SPECTRA
A tiny (and growing, depending on my needs) python module for analyses of
Spectra in python.

## INSTALLATION
You can install the package via `pip`:
```
pip install git+https://github.com/AstroTeutloff/simplespec.git
```
or by cloning the repository:
```bash
git clone https://github.com/AstroTeutloff/simplespec.git
cd simplespec
pip install .
```

## USE
The module is split up into "named" Spectra and the dataclass `Spectrum`. The
named spectra are instantiated from a file and usually hold not just the
spectral data (i.e. wavelengths, fluxes, flux uncertainties) in fields usually
called `spectrum`, or `spectra` (check the implementation on the case-by-case),
but also logistical information (e.g. the FITS header).

Explicit calculations are not performed with the named spectra classes, the
`Spectrum` class has a suite of commonly used methods for spectra (redshift,
adding two spectra, multiplying by a scalar factor)

The `TransitionLine` dataclass is also there, I'm not super sure what to do
with it yet, but its tested and can be used for visualisation purposes.

### PLOTTING
The named classes have a `plot_spectrum()` method.
```python
import matplotlib.pyplot as plt
import spectra as sp
sdss_obj = sp.SDSSSpectrum("YOUR_SDSS_SPEC.fits")
_ = sdss_obj.plot_spectrum()
plt.show()
```


## NOTABLE CHANGES
- As of v0.1.0:
  - `SDSSSpectrum` no longer supports fitting the continuum. If need arises, it will be implemented in the `Spectrum` dataclass itself.
  - `SDSSSpectrum` no longer supports masking the Spectrum edges. This _might_ be refactored into the `Spectrum` dataclass.
  - `SDSSSpectrum` no longer has QTables as its `spectra` and `coadd` fields, these are now `Spectrum` objects.
  - `BaseSpectrum` no longer exists, because I didn't have a reason to keep it around.

## CONTRIBUTION
If you have ideas and/or improvements, feel free to open a PR! :)

---
[brainmade](brainmade.org)
