from __future__ import annotations

import xarray as xr


@xr.register_dataset_accessor("radport")
class RadportAccessor:
    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj
        self._validate()

    def _validate(self) -> None:
        required_coords = {"l", "m"}
        missing = required_coords - set(self._obj.coords)
        if missing:
            raise AttributeError(
                f"Dataset must have {required_coords} coordinates for OVRO-LWA data. "
                f"Missing: {missing}"
            )

    def plot(self) -> None:
        import matplotlib.pyplot as plt
        from astropy.wcs import WCS

        ds = self._obj

        if "time" in ds.dims and ds.sizes["time"] > 1:
            ds = ds.isel(time=0)
        if "frequency" in ds.dims and ds.sizes["frequency"] > 1:
            ds = ds.isel(frequency=0)

        data_var = next(iter(ds.data_vars))
        data = ds[data_var]

        wcs_header_str = None
        if "wcs_header_str" in ds:
            wcs_header_str = ds["wcs_header_str"].values
            if isinstance(wcs_header_str, bytes):
                wcs_header_str = wcs_header_str.decode("utf-8")
        elif "fits_wcs_header" in ds.attrs:
            wcs_header_str = ds.attrs["fits_wcs_header"]
        elif "fits_wcs_header" in data.attrs:
            wcs_header_str = data.attrs["fits_wcs_header"]

        if wcs_header_str:
            wcs = WCS(wcs_header_str)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection=wcs)
            im = ax.imshow(data.values, origin="lower", cmap="viridis")
            ax.set_xlabel("Right Ascension")
            ax.set_ylabel("Declination")
            ax.grid(color="white", ls="dotted", alpha=0.5)
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(data.values, origin="lower", cmap="viridis")
            ax.set_xlabel("l")
            ax.set_ylabel("m")

        plt.colorbar(im, ax=ax, label=data.attrs.get("units", "Intensity"))
        title_parts = [data_var]
        if "time" in ds.coords:
            title_parts.append(f"time={ds.time.values}")
        if "frequency" in ds.coords:
            freq_mhz = float(ds.frequency.values) / 1e6
            title_parts.append(f"freq={freq_mhz:.1f} MHz")
        plt.title(" | ".join(title_parts))
        plt.tight_layout()
        plt.show()
