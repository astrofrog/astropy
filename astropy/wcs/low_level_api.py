import abc

import numpy as np

from .. import units as u

__all__ = ['BaseLowLevelWCS', 'FITSLowLevelWCS']


class BaseLowLevelWCS(metaclass=abc.ABCMeta):
    """
    Abstract base class for the low-level WCS interface described in APE 14
    (https://doi.org/10.5281/zenodo.1188875)
    """

    @abc.abstractproperty
    def pixel_n_dim(self):
        """
        The number of axes in the pixel coordinate system.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def world_n_dim(self):
        """
        The number of axes in the world coordinate system.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def world_axis_physical_types(self):
        """
        An iterable of strings describing the physical type for each world axis.

        These should be names from the VO UCD1+ controlled Vocabulary
        (http://www.ivoa.net/documents/latest/UCDlist.html). If no matching UCD
        type exists, this can instead be ``"custom:xxx"``, where ``xxx`` is an
        arbitrary string.  Alternatively, if the physical type is
        unknown/undefined, an element can be `None`.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def world_axis_units(self):
        """
        An iterable of strings given the units of the world coordinates for each
        axis.

        The strings should follow the recommended VOUnit standard (though as
        noted in the VOUnit specification document, units that do not follow
        this standard are still allowed, but just not recommended).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def pixel_to_world_values(self, *pixel_arrays):
        """
        Convert pixel coordinates to world coordinates.

        This method takes ``pixel_n_dim`` scalars or arrays as input, and pixel
        coordinates should be zero-based. Returns ``world_n_dim`` scalars or
        arrays in units given by ``world_axis_units``. Note that pixel
        coordinates are assumed to be 0 at the center of the first pixel in each
        dimension.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def world_to_pixel_values(self, *world_arrays):
        """
        Convert world coordinates to pixel coordinates.

        This method takes ``world_n_dim`` scalars or arrays as input in units
        given by ``world_axis_units``. Returns ``pixel_n_dim`` scalars or
        arrays. Note that pixel coordinates are assumed to be 0 at the center of
        the first pixel in each dimension.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def world_axis_object_components(self):
        """
        A list with ``world_n_dim`` elements, where each element is a tuple with
        two items:

        * The first is a name for the world object this world array
          corresponds to, which *must* match the string names used in
          `world_axis_object_classes`. Note that names might appear twice
          because two world arrays might correspond to a single world object
          (e.g. a celestial coordinate might have both "ra" and "dec"
          arrays, which correspond to a single sky coordinate object).

        * The second element is either a string keyword argument name or a
          positional index for the corresponding class from
          ``world_axis_object_classes``

        See https://doi.org/10.5281/zenodo.1188875 for examples.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def world_axis_object_classes(self):
        """
        A dictionary with each key being a string key from
        ``world_axis_object_components``, and each value being a tuple with
        two elements:

        * The first element of the tuple must be a string specifying the
          fully-qualified name of a class, which will specify the actual
          Python object to be created.

        * The second tuple element must be a
          dictionary with the keyword arguments required to initialize the
          class.

        See https://doi.org/10.5281/zenodo.1188875 for examples.
        """
        raise NotImplementedError()

    # The following three properties have default fallback implementations, so
    # they are not abstract.

    @property
    def pixel_shape(self):
        """
        The shape of the data that the WCS applies to as a tuple of
        length ``pixel_n_dim``.

        If the WCS is valid in the context of a dataset with a particular
        shape, then this property can be used to store the shape of the
        data. This can be used for example if implementing slicing of WCS
        objects. This is an optional property, and it should return `None`
        if a shape is not known or relevant.
        """
        return None

    @property
    def pixel_bounds(self):
        """
        The bounds (in pixel coordinates) inside which the WCS is defined,
        as a list with ``pixel_n_dim`` ``(min, max)`` tuples.

        WCS solutions are sometimes only guaranteed to be accurate within a
        certain range of pixel values, for example when definining a WCS
        that includes fitted distortions. This is an optional property, and
        it should return `None` if a shape is not known or relevant.
        """
        return None

    @property
    def axis_correlation_matrix(self):
        """
        Returns an ``(world_n_dim, pixel_n_dim)`` matrix that indicates using
        booleans whether a given world coordinate depends on a given pixel
        coordinate.

        This defaults to a matrix where all elements are True in the absence of
        any further information. For completely independent axes, the diagonal
        would be `True` and all other entries `False`.
        """
        return np.ones((self.world_n_dim, self.pixel_n_dim), dtype=bool)


# Mapping from CTYPE axis name to UCD1

CTYPE_TO_UCD1 = {

    # Celestial coordinates
    'RA': 'pos.eq.ra',
    'DEC': 'pos.eq.dec',
    'GLON': 'pos.galactic.lon',
    'GLAT': 'pos.galactic.lat',
    'ELON': 'pos.ecliptic.lon',
    'ELAT': 'pos.ecliptic.lat',
    'TLON': 'pos.bodyrc.lon',
    'TLAT': 'pos.bodyrc.lat',
    'HPLT': 'custom:pos.helioprojective.lat',
    'HPLN': 'custom:pos.helioprojective.lon',

    # Spectral coordinates (WCS paper 3)
    'FREQ': 'em.freq',  # Frequency
    'ENER': 'em.energy',  # Energy
    'WAVN': 'em.wavenumber',  # Wavenumber
    'WAVE': 'em.wl',  # Vacuum wavelength
    'VRAD': 'spect.dopplerVeloc.radio',  # Radio velocity
    'VOPT': 'spect.dopplerVeloc.opt',  # Optical velocity
    'ZOPT': 'src.redshift',  # Redshift
    'AWAV': 'em.wl',  # Air wavelength
    'VELO': 'spect.dopplerVeloc',  # Apparent radial velocity
    'BETA': None,  # Beta factor (v/c)

    # Time coordinates (https://www.aanda.org/articles/aa/pdf/2015/02/aa24653-14.pdf)
    'TIME': 'time',
    'TAI': 'time',
    'TT': 'time',
    'TDT': 'time',
    'ET': 'time',
    'IAT': 'time',
    'UT1': 'time',
    'UTC': 'time',
    'GMT': 'time',
    'GPS': 'time',
    'TCG': 'time',
    'TCB': 'time',
    'TDB': 'time',
    'LOCAL': 'time'

    # UT() is handled separately in world_axis_physical_types

}


class FITSLowLevelWCS(BaseLowLevelWCS):
    """
    A wrapper around the :class:`astropy.wcs.WCS` class that provides the
    low-level WCS API from APE 14.
    """

    def __init__(self, wcs):
        self._wcs = wcs

    @property
    def pixel_n_dim(self):
        return self._wcs.naxis

    @property
    def world_n_dim(self):
        return len(self._wcs.wcs.ctype)

    @property
    def world_axis_physical_types(self):
        types = []
        for axis_type in self._wcs.axis_type_names:
            if axis_type.startswith('UT('):
                types.append('time')
            else:
                types.append(CTYPE_TO_UCD1.get(axis_type, None))
        return types

    @property
    def world_axis_units(self):
        units = []
        for unit in self._wcs.wcs.cunit:
            if unit is None:
                unit = ''
            elif isinstance(unit, u.Unit):
                unit = unit.to_string(format='vounit')
            else:
                try:
                    unit = u.Unit(unit).to_string(format='vounit')
                except u.UnitsError:
                    unit = ''
            units.append(unit)
        return units

    @property
    def axis_correlation_matrix(self):

        # If there are any distortions present, we assume that there may be
        # correlations between all axes. Maybe if some distortions only apply
        # to the image plane we can improve this?
        for distortion_attribute in ('sip', 'det2im1', 'det2im2'):
            if getattr(self._wcs, distortion_attribute):
                return np.ones((self.n_world, self.n_pixel), dtype=bool)

        # Assuming linear world coordinates along each axis, the correlation
        # matrix would be given by whether or not the PC matrix is zero
        matrix = self._wcs.wcs.get_pc() != 0

        # We now need to check specifically for celestial coordinates since
        # these can assume correlations because of spherical distortions. For
        # each celestial coordinate we copy over the pixel dependencies from
        # the other celestial coordinates.
        celestial = (self._wcs.wcs.axis_types // 1000) % 10 == 2
        celestial_indices = np.nonzero(celestial)[0]
        for world1 in celestial_indices:
            for world2 in celestial_indices:
                if world1 != world2:
                    matrix[world1] |= matrix[world2]
                    matrix[world2] |= matrix[world1]

        return matrix

    def pixel_to_world_values(self, *pixel_arrays):
        return self.all_pixel_to_world(*pixel_arrays, 0)

    def world_to_pixel_values(self, *world_arrays):
        return self.all_world_to_pixel(*world_arrays, 0)

    @property
    def world_axis_object_components(self):
        return _get_components_and_classes(self._wcs)[0]

    @property
    def world_axis_object_classes(self):
        return _get_components_and_classes(self._wcs)[1]


def _get_components_and_classes(wcs):

    # The aim of this function is to return whatever is needed for
    # world_axis_object_components and world_axis_object_classes. It's easier
    # to figure it out in one go and then return the values and let the
    # properties return part of it.

    from .utils import wcs_to_celestial_frame

    from ..coordinates.attributes import Attribute, QuantityAttribute, TimeAttribute

    components = [None] * wcs.naxis
    classes = {}

    # Let's start off by checking whether the WCS has a pair of celestial
    # components

    if wcs.has_celestial:

        frame = wcs_to_celestial_frame(wcs)

        if frame is not None:

            kwargs = {}

            # TODO: don't need to list attributes that have default values

            for name, attr in frame.frame_attributes.items():
                value = attr.__get__(frame)

                # Don't use isinstance as we don't want to match subclasses

                if type(attr) is Attribute:
                    kwargs[name] = value
                elif type(attr) is QuantityAttribute:
                    # TODO: update APE14 to allow tuple to mean nested classes
                    kwargs[name] = ('astropy.units.Quantity',
                                    {'value': value.value,
                                     'unit': value.unit.to_string('vounit')})
                elif type(attr) is TimeAttribute:
                    kwargs[name] = ('astropy.time.Time',
                                    {'val': value.value,
                                     'format': value.format,
                                     'scale': value.scale})
                else:
                    raise NotImplementedError("Don't yet know how to serialize {0}".format(type(attr)))

        classes['celestial'] = ('astropy.coordinates.SkyCoord', kwargs)

        components[wcs.wcs.lng] = ('celestial', 0)
        components[wcs.wcs.lat] = ('celestial', 1)

    # Fallback: for any remaining components that haven't been identified, just
    # return Quantity as the class to use

    for i in range(wcs.naxis):
        if components[i] is None:
            name = wcs.axis_type_names[i].lower()
            while name in classes:
                name += "_"
            classes[name] = ('astropy.units.Quantity', {'unit': wcs.wcs.cunit[i]})
            components[i] = (name, 0)

    return components, classes
