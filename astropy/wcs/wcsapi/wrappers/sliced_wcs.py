import numbers
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from astropy.utils.decorators import lazyproperty
from astropy.wcs.wcsapi.low_level_api import (
    BaseLowLevelWCS,
    _WorldAxisClass,
    _WorldAxisComponent,
)

from .base import BaseWCSWrapper

__all__ = ["SlicedLowLevelWCS", "sanitize_slices"]


def sanitize_slices(slices: Any, ndim: int) -> list[slice | int]:
    """
    Given a slice as input sanitise it to an easier to parse format.format.

    This function returns a list ``ndim`` long containing slice objects (or ints).
    """
    if not isinstance(slices, (tuple, list)):  # We just have a single int
        slices = (slices,)

    if len(slices) > ndim:
        raise ValueError(
            f"The dimensionality of the specified slice {slices} can not be greater "
            f"than the dimensionality ({ndim}) of the wcs."
        )

    if any(np.iterable(s) for s in slices):
        raise IndexError(
            "This slice is invalid, only integer or range slices are supported."
        )

    slices = list(slices)

    if Ellipsis in slices:
        if slices.count(Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        # Replace the Ellipsis with the correct number of slice(None)s
        e_ind = slices.index(Ellipsis)
        slices.remove(Ellipsis)
        n_e = ndim - len(slices)
        for i in range(n_e):
            ind = e_ind + i
            slices.insert(ind, slice(None))

    for i in range(ndim):
        if i < len(slices):
            slc = slices[i]
            if isinstance(slc, slice):
                if slc.step and slc.step != 1:
                    raise IndexError("Slicing WCS with a step is not supported.")
            elif not isinstance(slc, numbers.Integral):
                raise IndexError("Only integer or range slices are accepted.")
        else:
            slices.append(slice(None))

    return slices


def combine_slices(slice1: Any, slice2: Any) -> slice | int:
    """
    Given two slices that can be applied to a 1-d array, find the resulting
    slice that corresponds to the combination of both slices. We assume that
    slice2 can be an integer, but slice1 cannot.
    """
    if isinstance(slice1, slice) and slice1.step is not None:
        raise ValueError("Only slices with steps of 1 are supported")

    if isinstance(slice2, slice) and slice2.step is not None:
        raise ValueError("Only slices with steps of 1 are supported")

    if isinstance(slice2, numbers.Integral):
        if slice1.start is None:
            return cast(int, slice2)
        else:
            return slice2 + slice1.start

    if slice1.start is None:
        if slice1.stop is None:
            return slice2
        else:
            if slice2.stop is None:
                return slice(slice2.start, slice1.stop)
            else:
                return slice(slice2.start, min(slice1.stop, slice2.stop))
    else:
        if slice2.start is None:
            start = slice1.start
        else:
            start = slice1.start + slice2.start
        if slice2.stop is None:
            stop = slice1.stop
        else:
            if slice1.start is None:
                stop = slice2.stop
            else:
                stop = slice2.stop + slice1.start
            if slice1.stop is not None:
                stop = min(slice1.stop, stop)
    return slice(start, stop)


class SlicedLowLevelWCS(BaseWCSWrapper):
    """
    A Low Level WCS wrapper which applies an array slice to a WCS.

    This class does not modify the underlying WCS object and can therefore drop
    coupled dimensions as it stores which pixel and world dimensions have been
    sliced out (or modified) in the underlying WCS and returns the modified
    results on all the Low Level WCS methods.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS to slice.
    slices : `slice` or `tuple` or `int`
        A valid array slice to apply to the WCS.

    """

    _slices_array: list[Any]
    _slices_pixel: list[Any]
    _pixel_keep: np.ndarray
    _world_keep: np.ndarray

    def __init__(self, wcs: BaseLowLevelWCS, slices: Any) -> None:
        slices = sanitize_slices(slices, wcs.pixel_n_dim)

        if isinstance(wcs, SlicedLowLevelWCS):
            # Here we combine the current slices with the previous slices
            # to avoid ending up with many nested WCSes
            self._wcs = wcs._wcs
            slices_original = wcs._slices_array.copy()
            for ipixel in range(wcs.pixel_n_dim):
                ipixel_orig = wcs._wcs.pixel_n_dim - 1 - wcs._pixel_keep[ipixel]
                ipixel_new = wcs.pixel_n_dim - 1 - ipixel
                slices_original[ipixel_orig] = combine_slices(
                    slices_original[ipixel_orig], slices[ipixel_new]
                )
            self._slices_array = slices_original
        else:
            self._wcs = wcs
            self._slices_array = slices

        self._slices_pixel = self._slices_array[::-1]

        # figure out which pixel dimensions have been kept, then use axis correlation
        # matrix to figure out which world dims are kept
        self._pixel_keep = np.nonzero(
            [
                not isinstance(self._slices_pixel[ip], numbers.Integral)
                for ip in range(self._wcs.pixel_n_dim)
            ]
        )[0]

        # axis_correlation_matrix[world, pixel]
        self._world_keep = np.nonzero(
            self._wcs.axis_correlation_matrix[:, self._pixel_keep].any(axis=1)
        )[0]

        if len(self._pixel_keep) == 0 or len(self._world_keep) == 0:
            raise ValueError(
                "Cannot slice WCS: the resulting WCS should have "
                "at least one pixel and one world dimension."
            )

    @lazyproperty
    def dropped_world_dimensions(self) -> dict[str, Any]:
        """
        Information describing the dropped world dimensions.
        """
        world_coords = self._pixel_to_world_values_all(*[0] * len(self._pixel_keep))
        dropped_info: defaultdict[str, Any] = defaultdict(list)

        for i in range(self._wcs.world_n_dim):
            if i in self._world_keep:
                continue

            if "world_axis_object_classes" not in dropped_info:
                dropped_info["world_axis_object_classes"] = {}

            wao_classes = self._wcs.world_axis_object_classes
            wao_components = self._wcs.world_axis_object_components

            dropped_info["value"].append(world_coords[i])
            dropped_info["world_axis_names"].append(self._wcs.world_axis_names[i])
            dropped_info["world_axis_physical_types"].append(
                self._wcs.world_axis_physical_types[i]
            )
            dropped_info["world_axis_units"].append(self._wcs.world_axis_units[i])
            dropped_info["world_axis_object_components"].append(wao_components[i])
            dropped_info["world_axis_object_classes"].update(
                dict(
                    filter(lambda x: x[0] == wao_components[i][0], wao_classes.items())
                )
            )
            dropped_info["serialized_classes"] = self.serialized_classes
        return dict(dropped_info)

    @property
    def pixel_n_dim(self) -> int:
        return len(self._pixel_keep)

    @property
    def world_n_dim(self) -> int:
        return len(self._world_keep)

    @property
    def world_axis_physical_types(self) -> Sequence[str | None]:
        return [self._wcs.world_axis_physical_types[int(i)] for i in self._world_keep]

    @property
    def world_axis_units(self) -> Sequence[str]:
        return [self._wcs.world_axis_units[int(i)] for i in self._world_keep]

    @property
    def pixel_axis_names(self) -> Sequence[str]:
        return [self._wcs.pixel_axis_names[int(i)] for i in self._pixel_keep]

    @property
    def world_axis_names(self) -> Sequence[str]:
        return [self._wcs.world_axis_names[int(i)] for i in self._world_keep]

    def _pixel_to_world_values_all(self, *pixel_arrays: Any) -> Any:
        pixel_arrays = tuple(map(np.asanyarray, pixel_arrays))
        pixel_arrays_new = []
        ipix_curr = -1
        for ipix in range(self._wcs.pixel_n_dim):
            if isinstance(self._slices_pixel[ipix], numbers.Integral):
                pixel_arrays_new.append(self._slices_pixel[ipix])
            else:
                ipix_curr += 1
                if self._slices_pixel[ipix].start is not None:
                    pixel_arrays_new.append(
                        pixel_arrays[ipix_curr] + self._slices_pixel[ipix].start
                    )
                else:
                    pixel_arrays_new.append(pixel_arrays[ipix_curr])

        return self._wcs.pixel_to_world_values(*np.broadcast_arrays(*pixel_arrays_new))

    def pixel_to_world_values(self, *pixel_arrays: ArrayLike) -> Any:
        world_arrays: Any = self._pixel_to_world_values_all(*pixel_arrays)

        # Detect the case of a length 0 array
        if isinstance(world_arrays, np.ndarray) and not world_arrays.shape:
            return world_arrays

        if self._wcs.world_n_dim > 1:
            # Select the dimensions of the original WCS we are keeping.
            world_arrays = [world_arrays[iw] for iw in self._world_keep]
            # If there is only one world dimension (after slicing) we shouldn't return a tuple.
            if self.world_n_dim == 1:
                world_arrays = world_arrays[0]

        return world_arrays

    def world_to_pixel_values(self, *world_arrays: ArrayLike) -> Any:
        sliced_out_world_coords = self._pixel_to_world_values_all(
            *[0] * len(self._pixel_keep)
        )

        world_values = tuple(map(np.asanyarray, world_arrays))
        world_arrays_new: list[Any] = []
        iworld_curr = -1
        for iworld in range(self._wcs.world_n_dim):
            if iworld in self._world_keep:
                iworld_curr += 1
                world_arrays_new.append(world_values[iworld_curr])
            else:
                world_arrays_new.append(sliced_out_world_coords[iworld])

        pixel_values: Any = self._wcs.world_to_pixel_values(
            *np.broadcast_arrays(*world_arrays_new)
        )
        pixel_values = (
            list(pixel_values) if self._wcs.pixel_n_dim > 1 else [pixel_values]
        )

        for ipixel in range(self._wcs.pixel_n_dim):
            if (
                isinstance(self._slices_pixel[ipixel], slice)
                and self._slices_pixel[ipixel].start is not None
            ):
                pixel_values[ipixel] -= self._slices_pixel[ipixel].start

        # Detect the case of a length 0 array
        if isinstance(pixel_values, np.ndarray) and not pixel_values.shape:
            return pixel_values
        pixel = tuple(pixel_values[ip] for ip in self._pixel_keep)
        return pixel[0] if self.pixel_n_dim == 1 else pixel

    @property
    def world_axis_object_components(self) -> Sequence[_WorldAxisComponent]:
        return [
            self._wcs.world_axis_object_components[int(idx)] for idx in self._world_keep
        ]

    @property
    def world_axis_object_classes(self) -> Mapping[str, _WorldAxisClass]:
        keys_keep = [item[0] for item in self.world_axis_object_components]
        return dict(
            [
                item
                for item in self._wcs.world_axis_object_classes.items()
                if item[0] in keys_keep
            ]
        )

    @property
    def array_shape(self) -> tuple[int, ...] | None:
        if self._wcs.array_shape:
            return np.broadcast_to(0, self._wcs.array_shape)[
                tuple(self._slices_array)
            ].shape
        return None

    @property
    def pixel_shape(self) -> tuple[int, ...] | None:
        if self.array_shape:
            return tuple(self.array_shape[::-1])
        return None

    @property
    def pixel_bounds(self) -> Sequence[tuple[float, float] | None] | None:
        if self._wcs.pixel_bounds is None:
            return None

        bounds: list[Any] = []
        for idx in self._pixel_keep:
            if self._slices_pixel[idx].start is None:
                bounds.append(self._wcs.pixel_bounds[int(idx)])
            else:
                imin, imax = cast(
                    "tuple[float, float]", self._wcs.pixel_bounds[int(idx)]
                )
                start = self._slices_pixel[idx].start
                bounds.append((imin - start, imax - start))

        return tuple(bounds)

    @property
    def axis_correlation_matrix(self) -> np.ndarray:
        return self._wcs.axis_correlation_matrix[self._world_keep][:, self._pixel_keep]
