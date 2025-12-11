from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import ovro_lwa_portal


def test_accessor_registration() -> None:
    ds = xr.Dataset(
        {"SKY": (("m", "l"), np.random.rand(10, 10))},
        coords={"l": np.arange(10), "m": np.arange(10)},
    )
    assert hasattr(ds, "radport")


def test_accessor_validation_missing_coords() -> None:
    ds = xr.Dataset(
        {"data": (("x", "y"), np.random.rand(10, 10))},
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    with pytest.raises(AttributeError, match="must have.*l.*m.*coordinates"):
        _ = ds.radport


def test_accessor_validation_partial_coords() -> None:
    ds = xr.Dataset(
        {"data": (("m", "x"), np.random.rand(10, 10))},
        coords={"m": np.arange(10), "x": np.arange(10)},
    )
    with pytest.raises(AttributeError, match="Missing.*l"):
        _ = ds.radport


def test_accessor_plot_basic(mocker) -> None:
    mock_show = mocker.patch("matplotlib.pyplot.show")

    ds = xr.Dataset(
        {"SKY": (("m", "l"), np.random.rand(10, 10))},
        coords={"l": np.arange(10), "m": np.arange(10)},
    )

    ds.radport.plot()
    mock_show.assert_called_once()


def test_accessor_plot_with_time_freq(mocker) -> None:
    mock_show = mocker.patch("matplotlib.pyplot.show")

    ds = xr.Dataset(
        {"SKY": (("time", "frequency", "m", "l"), np.random.rand(2, 3, 10, 10))},
        coords={
            "time": np.array([1.0, 2.0]),
            "frequency": np.array([50e6, 60e6, 70e6]),
            "l": np.arange(10),
            "m": np.arange(10),
        },
    )

    ds.radport.plot()
    mock_show.assert_called_once()


def test_accessor_plot_with_wcs(mocker) -> None:
    mock_show = mocker.patch("matplotlib.pyplot.show")

    wcs_header = """WCSAXES =                    2
CRPIX1  =                  512
CRPIX2  =                  512
CDELT1  =           -0.0166667
CDELT2  =            0.0166667
CUNIT1  = 'deg'
CUNIT2  = 'deg'
CTYPE1  = 'RA---SIN'
CTYPE2  = 'DEC--SIN'
CRVAL1  =                  0.0
CRVAL2  =                 90.0
"""

    ds = xr.Dataset(
        {"SKY": (("m", "l"), np.random.rand(10, 10))},
        coords={"l": np.arange(10), "m": np.arange(10)},
    )
    ds.attrs["fits_wcs_header"] = wcs_header

    ds.radport.plot()
    mock_show.assert_called_once()
