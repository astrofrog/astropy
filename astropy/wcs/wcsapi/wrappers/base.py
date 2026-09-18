import abc
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from astropy.wcs.wcsapi import BaseLowLevelWCS, wcs_info_str
from astropy.wcs.wcsapi.low_level_api import (
    _WorldAxisClass,
    _WorldAxisComponent,
)


class BaseWCSWrapper(BaseLowLevelWCS, metaclass=abc.ABCMeta):
    """
    A base wrapper class for things that modify Low Level WCSes.

    This wrapper implements a transparent wrapper to many of the properties,
    with the idea that not all of them would need to be overridden in your
    wrapper, but some probably will.

    Parameters
    ----------
    wcs : `astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS object to wrap
    """

    def __init__(self, wcs: BaseLowLevelWCS, *args: Any, **kwargs: Any) -> None:
        self._wcs: BaseLowLevelWCS = wcs

    @property
    def pixel_n_dim(self) -> int:
        return self._wcs.pixel_n_dim

    @property
    def world_n_dim(self) -> int:
        return self._wcs.world_n_dim

    @property
    def world_axis_physical_types(self) -> Sequence[str | None]:
        return self._wcs.world_axis_physical_types

    @property
    def world_axis_units(self) -> Sequence[str]:
        return self._wcs.world_axis_units

    @property
    def world_axis_object_components(self) -> Sequence[_WorldAxisComponent]:
        return self._wcs.world_axis_object_components

    @property
    def world_axis_object_classes(self) -> Mapping[str, _WorldAxisClass]:
        return self._wcs.world_axis_object_classes

    @property
    def pixel_shape(self) -> tuple[int, ...] | None:
        return self._wcs.pixel_shape

    @property
    def pixel_bounds(self) -> Sequence[tuple[float, float] | None] | None:
        return self._wcs.pixel_bounds

    @property
    def pixel_axis_names(self) -> Sequence[str]:
        return self._wcs.pixel_axis_names

    @property
    def world_axis_names(self) -> Sequence[str]:
        return self._wcs.world_axis_names

    @property
    def axis_correlation_matrix(self) -> np.ndarray:
        return self._wcs.axis_correlation_matrix

    @property
    def serialized_classes(self) -> bool:
        return self._wcs.serialized_classes

    @abc.abstractmethod
    def pixel_to_world_values(self, *pixel_arrays: ArrayLike) -> Any:
        pass

    @abc.abstractmethod
    def world_to_pixel_values(self, *world_arrays: ArrayLike) -> Any:
        pass

    def __repr__(self) -> str:
        return f"{object.__repr__(self)}\n{str(self)}"

    def __str__(self) -> str:
        return wcs_info_str(self)
