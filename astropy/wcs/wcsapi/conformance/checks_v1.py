"""
Checks for the requirements of version 1 of the shared WCS API, which is the
version defined in APE 14.

The checks follow the order in which the API is described in APE 14, and are
split into the properties and methods of the low-level API, and the methods of
the high-level API. A few of the checks cover additions that were made to the
API in astropy after APE 14 was accepted, which is indicated where relevant.
"""

import numbers
from collections.abc import Mapping

import numpy as np

from astropy import units as u
from astropy.wcs.wcsapi.high_level_api import (
    BaseHighLevelWCS,
    default_order,
    high_level_objects_to_values,
)
from astropy.wcs.wcsapi.high_level_wcs_wrapper import HighLevelWCSWrapper
from astropy.wcs.wcsapi.low_level_api import BaseLowLevelWCS, validate_physical_types
from astropy.wcs.wcsapi.utils import deserialize_class

from .core import CheckCollection, as_tuple, require

__all__ = []

checks = CheckCollection(version=1)

LOW_LEVEL_PROPERTIES = [
    "pixel_n_dim",
    "world_n_dim",
    "array_shape",
    "pixel_shape",
    "pixel_bounds",
    "world_axis_physical_types",
    "world_axis_units",
    "axis_correlation_matrix",
    "serialized_classes",
    "world_axis_object_components",
    "world_axis_object_classes",
]

LOW_LEVEL_METHODS = [
    "pixel_to_world_values",
    "array_index_to_world_values",
    "world_to_pixel_values",
    "world_to_array_index_values",
]

HIGH_LEVEL_METHODS = [
    "pixel_to_world",
    "array_index_to_world",
    "world_to_pixel",
    "world_to_array_index",
]


def is_integer(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def expected_array_index(pixel):
    # Array indices are the pixel coordinates rounded to the nearest integer,
    # in reverse order.
    return tuple(int(np.floor(p + 0.5)) for p in pixel)[::-1]


def object_classes(low_level_wcs):
    """
    The classes of the high-level objects, in the order in which the objects
    should be returned by the high-level API.
    """
    classes = low_level_wcs.world_axis_object_classes
    result = []
    for name in default_order(low_level_wcs.world_axis_object_components):
        if low_level_wcs.serialized_classes:
            result.append(deserialize_class(classes[name], construct=False)[0])
        else:
            result.append(classes[name][0])
    return result


def high_level_wcs_for(wcs):
    """
    An object exposing the high-level API for a given WCS object.

    For objects that only implement the low-level API, we use the wrapper from
    astropy, so that the checks of the high-level API test whether the
    information provided by the low-level object is sufficient to build and
    interpret high-level objects.
    """
    if isinstance(wcs, BaseHighLevelWCS):
        return wcs
    return HighLevelWCSWrapper(wcs)


def world_objects_as_tuple(world_objects, n_objects):
    # Methods of the high-level API return a single object instead of a tuple
    # when all the world coordinates are represented by one object.
    if n_objects == 1 and not isinstance(world_objects, (tuple, list)):
        return (world_objects,)
    return tuple(world_objects)


# LOW-LEVEL API: BASE CLASS AND REQUIRED ATTRIBUTES


@checks.register()
def check_low_level_base_class(wcs):
    require(
        isinstance(wcs, BaseLowLevelWCS),
        "the WCS should either inherit from BaseLowLevelWCS or be registered "
        "with it using BaseLowLevelWCS.register",
    )


def has_attribute(wcs, name):
    # We cannot use hasattr since this only returns False for an
    # AttributeError, and a property that exists can raise other exceptions
    # if the WCS is not valid, which is for other checks to report.
    try:
        getattr(wcs, name)
    except AttributeError:
        return False
    except Exception:
        return True
    return True


@checks.register()
def check_low_level_attributes(wcs):
    missing = [
        name
        for name in LOW_LEVEL_PROPERTIES + LOW_LEVEL_METHODS
        if not has_attribute(wcs, name)
    ]
    require(
        len(missing) == 0,
        f"the WCS does not define the following attributes: {missing}",
    )
    not_callable = [
        name for name in LOW_LEVEL_METHODS if not callable(getattr(wcs, name))
    ]
    require(
        len(not_callable) == 0,
        f"the following should be methods but cannot be called: {not_callable}",
    )


# LOW-LEVEL API: PROPERTIES


@checks.register()
def check_dimensions(wcs):
    for name in ("pixel_n_dim", "world_n_dim"):
        value = getattr(wcs, name)
        require(
            is_integer(value) and value > 0,
            f"{name} should be a positive integer but is {value!r}",
        )


@checks.register()
def check_array_shape(wcs):
    array_shape = wcs.array_shape
    pixel_shape = wcs.pixel_shape
    require(
        (array_shape is None) == (pixel_shape is None),
        "array_shape and pixel_shape should either both be None or both be set",
    )
    if array_shape is None:
        return
    for name, shape in (("array_shape", array_shape), ("pixel_shape", pixel_shape)):
        require(
            len(shape) == wcs.pixel_n_dim,
            f"{name} should have {wcs.pixel_n_dim} elements but has {len(shape)}",
        )
        require(
            all(is_integer(n) and n >= 0 for n in shape),
            f"{name} should contain non-negative integers but is {tuple(shape)}",
        )
    require(
        tuple(array_shape) == tuple(pixel_shape)[::-1],
        f"array_shape {tuple(array_shape)} should be the reverse of pixel_shape "
        f"{tuple(pixel_shape)}",
    )


@checks.register()
def check_pixel_bounds(wcs):
    pixel_bounds = wcs.pixel_bounds
    if pixel_bounds is None:
        return
    require(
        len(pixel_bounds) == wcs.pixel_n_dim,
        f"pixel_bounds should have {wcs.pixel_n_dim} elements but has "
        f"{len(pixel_bounds)}",
    )
    for bounds in pixel_bounds:
        # APE 14 requires a (min, max) tuple for each pixel dimension, and
        # astropy additionally allows None for dimensions without bounds.
        require(
            bounds is None
            or (
                len(bounds) == 2
                and all(isinstance(b, numbers.Real) for b in bounds)
                and bounds[0] <= bounds[1]
            ),
            f"pixel_bounds should contain (min, max) tuples but contains {bounds!r}",
        )


@checks.register()
def check_world_axis_physical_types(wcs):
    physical_types = wcs.world_axis_physical_types
    require(
        len(physical_types) == wcs.world_n_dim,
        f"world_axis_physical_types should have {wcs.world_n_dim} elements but "
        f"has {len(physical_types)}",
    )
    require(
        all(p is None or isinstance(p, str) for p in physical_types),
        "world_axis_physical_types should contain strings or None but is "
        f"{physical_types!r}",
    )
    try:
        validate_physical_types(physical_types)
    except ValueError as exc:
        raise AssertionError(str(exc)) from None


@checks.register()
def check_world_axis_units(wcs):
    units = wcs.world_axis_units
    require(
        len(units) == wcs.world_n_dim,
        f"world_axis_units should have {wcs.world_n_dim} elements but has {len(units)}",
    )
    for unit in units:
        require(
            isinstance(unit, str),
            f"world_axis_units should contain strings but contains {unit!r}",
        )
        # APE 14 recommends but does not require the VOUnit standard, so we
        # only check that the string can be interpreted as a unit.
        try:
            u.Unit(unit)
        except ValueError:
            raise AssertionError(
                f"world_axis_units contains {unit!r} which is not a valid unit"
            ) from None


@checks.register()
def check_axis_names(wcs):
    # These properties are not part of APE 14 and were added to the API in
    # astropy afterwards, with default implementations in BaseLowLevelWCS.
    for attribute, n_dim in (
        ("world_axis_names", wcs.world_n_dim),
        ("pixel_axis_names", wcs.pixel_n_dim),
    ):
        names = getattr(wcs, attribute)
        require(
            len(names) == n_dim,
            f"{attribute} should have {n_dim} elements but has {len(names)}",
        )
        require(
            all(isinstance(name, str) for name in names),
            f"{attribute} should contain strings but is {names!r}",
        )


@checks.register()
def check_axis_correlation_matrix(wcs):
    matrix = np.asarray(wcs.axis_correlation_matrix)
    expected_shape = (wcs.world_n_dim, wcs.pixel_n_dim)
    require(
        matrix.shape == expected_shape,
        f"axis_correlation_matrix should have shape {expected_shape} but has "
        f"shape {matrix.shape}",
    )
    require(
        matrix.dtype == bool,
        f"axis_correlation_matrix should contain booleans but has dtype {matrix.dtype}",
    )


@checks.register()
def check_serialized_classes(wcs):
    require(
        isinstance(wcs.serialized_classes, bool),
        f"serialized_classes should be a boolean but is {wcs.serialized_classes!r}",
    )


@checks.register()
def check_world_axis_object_components(wcs):
    components = wcs.world_axis_object_components
    require(
        len(components) == wcs.world_n_dim,
        f"world_axis_object_components should have {wcs.world_n_dim} elements but "
        f"has {len(components)}",
    )
    classes = wcs.world_axis_object_classes
    for component in components:
        require(
            len(component) == 3,
            "world_axis_object_components should contain tuples with three "
            f"elements but contains {component!r}",
        )
        name, key, attribute = component
        require(
            isinstance(name, str) and name in classes,
            f"world_axis_object_components refers to {name!r} which is not a "
            "key in world_axis_object_classes",
        )
        require(
            isinstance(key, str) or is_integer(key),
            "the second element of each tuple in world_axis_object_components "
            f"should be a keyword argument name or a positional index but is {key!r}",
        )
        # APE 14 requires the name of a property, and astropy additionally
        # allows a function which is called with the high-level object.
        require(
            isinstance(attribute, str) or callable(attribute),
            "the third element of each tuple in world_axis_object_components "
            f"should be a string or a callable but is {attribute!r}",
        )


@checks.register()
def check_world_axis_object_classes(wcs):
    serialized = wcs.serialized_classes
    classes = wcs.world_axis_object_classes
    require(
        isinstance(classes, Mapping), "world_axis_object_classes should be a dictionary"
    )
    for name, definition in classes.items():
        require(
            isinstance(name, str),
            f"the keys of world_axis_object_classes should be strings but {name!r} "
            "is not",
        )
        # APE 14 defines three elements, and astropy allows a fourth optional
        # element which is a function used to construct the object.
        require(
            isinstance(definition, (tuple, list)) and len(definition) in (3, 4),
            f"world_axis_object_classes[{name!r}] should be a tuple with three or "
            f"four elements but is {definition!r}",
        )
        klass, args, kwargs = definition[:3]
        # Implementations should either always or never use the serialized
        # form, as indicated by serialized_classes.
        if serialized:
            require(
                isinstance(klass, str),
                f"the class in world_axis_object_classes[{name!r}] should be given "
                f"as a string since serialized_classes is True, but is {klass!r}",
            )
            try:
                deserialize_class(tuple(definition[:3]), construct=False)
            except Exception as exc:
                raise AssertionError(
                    f"world_axis_object_classes[{name!r}] could not be "
                    f"deserialized: {exc}"
                ) from None
        else:
            require(
                isinstance(klass, type),
                f"the class in world_axis_object_classes[{name!r}] should be a "
                f"class since serialized_classes is False, but is {klass!r}",
            )
        require(
            isinstance(args, (tuple, list)),
            f"the second element of world_axis_object_classes[{name!r}] should be "
            f"a tuple of positional arguments but is {args!r}",
        )
        require(
            isinstance(kwargs, Mapping),
            f"the third element of world_axis_object_classes[{name!r}] should be "
            f"a dictionary of keyword arguments but is {kwargs!r}",
        )
        if len(definition) == 4:
            require(
                callable(definition[3]),
                f"the fourth element of world_axis_object_classes[{name!r}] should "
                f"be a callable but is {definition[3]!r}",
            )


# LOW-LEVEL API: METHODS


@checks.register(needs_pixel_points=True)
def check_pixel_to_world_values(wcs, pixel_points, rtol):
    for pixel in pixel_points:
        world = wcs.pixel_to_world_values(*pixel)
        if wcs.world_n_dim > 1:
            require(
                isinstance(world, (tuple, list)) and len(world) == wcs.world_n_dim,
                f"pixel_to_world_values should return {wcs.world_n_dim} values",
            )
        for value in as_tuple(world, wcs.world_n_dim):
            require(
                np.shape(value) == (),
                "pixel_to_world_values should return scalars for scalar input "
                f"but returned a value with shape {np.shape(value)}",
            )
            require(
                np.issubdtype(np.asarray(value).dtype, np.number),
                "pixel_to_world_values should return numerical values but "
                f"returned {value!r}",
            )


@checks.register(needs_pixel_points=True)
def check_world_to_pixel_values(wcs, pixel_points, rtol):
    for pixel in pixel_points:
        world = as_tuple(wcs.pixel_to_world_values(*pixel), wcs.world_n_dim)
        pixel_out = wcs.world_to_pixel_values(*world)
        if wcs.pixel_n_dim > 1:
            require(
                isinstance(pixel_out, (tuple, list))
                and len(pixel_out) == wcs.pixel_n_dim,
                f"world_to_pixel_values should return {wcs.pixel_n_dim} values",
            )
        pixel_out = as_tuple(pixel_out, wcs.pixel_n_dim)
        for value in pixel_out:
            require(
                np.shape(value) == (),
                "world_to_pixel_values should return scalars for scalar input "
                f"but returned a value with shape {np.shape(value)}",
            )
        require(
            np.allclose(pixel_out, pixel, rtol=rtol, atol=rtol),
            f"converting pixel coordinates {pixel} to world coordinates and back "
            f"gives {tuple(float(p) for p in pixel_out)}",
        )


@checks.register(needs_pixel_points=True)
def check_array_input(wcs, pixel_points, rtol):
    # We check one-dimensional arrays as well as two-dimensional arrays, in
    # which the positions are repeated along the first dimension.
    pixel_1d = [np.array(values) for values in zip(*pixel_points)]
    pixel_2d = [np.tile(values, (3, 1)) for values in pixel_1d]
    for pixel_arrays in (pixel_1d, pixel_2d):
        shape = pixel_arrays[0].shape
        world_arrays = as_tuple(
            wcs.pixel_to_world_values(*pixel_arrays), wcs.world_n_dim
        )
        for world_array in world_arrays:
            require(
                np.shape(world_array) == shape,
                "pixel_to_world_values should return arrays with the same shape "
                f"as the input, but returned shape {np.shape(world_array)} for "
                f"input shape {shape}",
            )
        pixel_arrays_out = as_tuple(
            wcs.world_to_pixel_values(*world_arrays), wcs.pixel_n_dim
        )
        for pixel_array in pixel_arrays_out:
            require(
                np.shape(pixel_array) == shape,
                "world_to_pixel_values should return arrays with the same shape "
                f"as the input, but returned shape {np.shape(pixel_array)} for "
                f"input shape {shape}",
            )
        for index, pixel in enumerate(pixel_points):
            world = as_tuple(wcs.pixel_to_world_values(*pixel), wcs.world_n_dim)
            require(
                np.allclose(
                    [np.asarray(w)[..., index].ravel()[0] for w in world_arrays],
                    world,
                    rtol=rtol,
                    atol=0,
                ),
                "pixel_to_world_values gives different results for scalar and "
                f"array input at pixel coordinates {pixel}",
            )
            require(
                np.allclose(
                    [np.asarray(p)[..., index].ravel()[0] for p in pixel_arrays_out],
                    pixel,
                    rtol=rtol,
                    atol=rtol,
                ),
                "world_to_pixel_values gives different results for scalar and "
                f"array input at pixel coordinates {pixel}",
            )


@checks.register(needs_pixel_points=True)
def check_array_index_to_world_values(wcs, pixel_points, rtol):
    for pixel in pixel_points:
        index = expected_array_index(pixel)
        pixel = index[::-1]
        world = as_tuple(wcs.pixel_to_world_values(*pixel), wcs.world_n_dim)
        world_from_index = as_tuple(
            wcs.array_index_to_world_values(*index), wcs.world_n_dim
        )
        require(
            np.allclose(world_from_index, world, rtol=rtol, atol=0),
            f"array_index_to_world_values{index} should give the same result as "
            f"pixel_to_world_values{pixel}",
        )


@checks.register(needs_pixel_points=True)
def check_world_to_array_index_values(wcs, pixel_points, rtol):
    for pixel in pixel_points:
        world = as_tuple(wcs.pixel_to_world_values(*pixel), wcs.world_n_dim)
        index = as_tuple(wcs.world_to_array_index_values(*world), wcs.pixel_n_dim)
        require(
            len(index) == wcs.pixel_n_dim,
            f"world_to_array_index_values should return {wcs.pixel_n_dim} values "
            f"but returned {len(index)}",
        )
        require(
            all(np.issubdtype(np.asarray(i).dtype, np.integer) for i in index),
            "world_to_array_index_values should return integers",
        )
        expected = expected_array_index(pixel)
        require(
            tuple(int(i) for i in index) == expected,
            "world_to_array_index_values should return the pixel coordinates "
            f"rounded to the nearest integer and in reverse order, that is "
            f"{expected} for pixel coordinates {pixel}, but returned "
            f"{tuple(int(i) for i in index)}",
        )


@checks.register(needs_pixel_points=True)
def check_axis_correlation_matrix_dependencies(wcs, pixel_points, rtol):
    # The matrix is allowed to include dependencies that do not exist, but
    # should not leave out any that do, so we change each pixel coordinate in
    # turn and check that all the world coordinates that change are marked as
    # depending on that pixel coordinate.
    matrix = np.asarray(wcs.axis_correlation_matrix)
    for pixel in pixel_points:
        world = np.array(as_tuple(wcs.pixel_to_world_values(*pixel), wcs.world_n_dim))
        for ipix in range(wcs.pixel_n_dim):
            shifted = list(pixel)
            shifted[ipix] += 0.5
            world_shifted = np.array(
                as_tuple(wcs.pixel_to_world_values(*shifted), wcs.world_n_dim)
            )
            changed = ~np.isclose(world_shifted, world, rtol=rtol, atol=0)
            missing = np.nonzero(changed & ~matrix[:, ipix])[0]
            require(
                len(missing) == 0,
                f"world axes {missing.tolist()} depend on pixel axis {ipix} but "
                "this is not indicated in axis_correlation_matrix",
            )


# HIGH-LEVEL API


@checks.register(high_level=True)
def check_high_level_attributes(wcs):
    if not isinstance(wcs, BaseHighLevelWCS):
        # The high-level API is optional for objects that implement the
        # low-level API, since these can be wrapped to provide it.
        return
    missing = [
        name for name in HIGH_LEVEL_METHODS if not callable(getattr(wcs, name, None))
    ]
    require(
        len(missing) == 0, f"the WCS does not define the following methods: {missing}"
    )
    require(
        isinstance(wcs.low_level_wcs, BaseLowLevelWCS),
        "low_level_wcs should return an object that implements the low-level API",
    )


@checks.register(needs_pixel_points=True, high_level=True)
def check_pixel_to_world(wcs, pixel_points, rtol):
    high_level_wcs = high_level_wcs_for(wcs)
    low_level_wcs = high_level_wcs.low_level_wcs
    expected = object_classes(low_level_wcs)
    for pixel in pixel_points:
        world_objects = high_level_wcs.pixel_to_world(*pixel)
        if len(expected) > 1:
            require(
                isinstance(world_objects, (tuple, list))
                and len(world_objects) == len(expected),
                f"pixel_to_world should return {len(expected)} objects",
            )
        world_objects = world_objects_as_tuple(world_objects, len(expected))
        # The objects should be in the order in which their names first
        # appear in world_axis_object_components.
        require(
            all(isinstance(w, k) for w, k in zip(world_objects, expected)),
            "pixel_to_world should return objects of type "
            f"{[k.__name__ for k in expected]} in that order, but returned "
            f"{[type(w).__name__ for w in world_objects]}",
        )
        # The values extracted from the objects using the properties given in
        # world_axis_object_components should be the low-level world values.
        values = high_level_objects_to_values(
            *world_objects, low_level_wcs=low_level_wcs
        )
        world = as_tuple(
            low_level_wcs.pixel_to_world_values(*pixel), low_level_wcs.world_n_dim
        )
        require(
            np.allclose(values, world, rtol=rtol, atol=0),
            "the values obtained from the objects returned by pixel_to_world "
            "using world_axis_object_components do not match the values "
            f"returned by pixel_to_world_values for pixel coordinates {pixel}",
        )


@checks.register(needs_pixel_points=True, high_level=True)
def check_world_object_conversion(wcs, pixel_points, rtol):
    # The classes should be able to carry out conversions when an instance is
    # passed as the first argument to the class.
    high_level_wcs = high_level_wcs_for(wcs)
    expected = object_classes(high_level_wcs.low_level_wcs)
    world_objects = world_objects_as_tuple(
        high_level_wcs.pixel_to_world(*pixel_points[0]), len(expected)
    )
    for world_object, klass in zip(world_objects, expected):
        try:
            converted = klass(world_object)
        except Exception as exc:
            raise AssertionError(
                f"{klass.__name__} cannot be initialized with an instance of "
                f"{klass.__name__} as its first argument: {exc}"
            ) from None
        require(
            isinstance(converted, klass),
            f"initializing {klass.__name__} with an instance of {klass.__name__} "
            f"returned an instance of {type(converted).__name__}",
        )


@checks.register(needs_pixel_points=True, high_level=True)
def check_world_to_pixel(wcs, pixel_points, rtol):
    high_level_wcs = high_level_wcs_for(wcs)
    low_level_wcs = high_level_wcs.low_level_wcs
    n_objects = len(object_classes(low_level_wcs))
    for pixel in pixel_points:
        world_objects = world_objects_as_tuple(
            high_level_wcs.pixel_to_world(*pixel), n_objects
        )
        pixel_out = as_tuple(
            high_level_wcs.world_to_pixel(*world_objects), low_level_wcs.pixel_n_dim
        )
        require(
            len(pixel_out) == low_level_wcs.pixel_n_dim,
            f"world_to_pixel should return {low_level_wcs.pixel_n_dim} values but "
            f"returned {len(pixel_out)}",
        )
        require(
            np.allclose(pixel_out, pixel, rtol=rtol, atol=rtol),
            f"converting pixel coordinates {pixel} to world objects and back "
            f"gives {tuple(float(p) for p in pixel_out)}",
        )


@checks.register(needs_pixel_points=True, high_level=True)
def check_array_index_to_world(wcs, pixel_points, rtol):
    high_level_wcs = high_level_wcs_for(wcs)
    low_level_wcs = high_level_wcs.low_level_wcs
    n_objects = len(object_classes(low_level_wcs))
    for pixel in pixel_points:
        index = expected_array_index(pixel)
        pixel = index[::-1]
        values = [
            high_level_objects_to_values(
                *world_objects_as_tuple(world_objects, n_objects),
                low_level_wcs=low_level_wcs,
            )
            for world_objects in (
                high_level_wcs.array_index_to_world(*index),
                high_level_wcs.pixel_to_world(*pixel),
            )
        ]
        require(
            np.allclose(values[0], values[1], rtol=rtol, atol=0),
            f"array_index_to_world{index} should give the same result as "
            f"pixel_to_world{pixel}",
        )


@checks.register(needs_pixel_points=True, high_level=True)
def check_world_to_array_index(wcs, pixel_points, rtol):
    high_level_wcs = high_level_wcs_for(wcs)
    low_level_wcs = high_level_wcs.low_level_wcs
    n_objects = len(object_classes(low_level_wcs))
    for pixel in pixel_points:
        world_objects = world_objects_as_tuple(
            high_level_wcs.pixel_to_world(*pixel), n_objects
        )
        index = as_tuple(
            high_level_wcs.world_to_array_index(*world_objects),
            low_level_wcs.pixel_n_dim,
        )
        require(
            all(np.issubdtype(np.asarray(i).dtype, np.integer) for i in index),
            "world_to_array_index should return integers",
        )
        expected = expected_array_index(pixel)
        require(
            tuple(int(i) for i in index) == expected,
            "world_to_array_index should return the pixel coordinates rounded "
            f"to the nearest integer and in reverse order, that is {expected} "
            f"for pixel coordinates {pixel}, but returned "
            f"{tuple(int(i) for i in index)}",
        )
