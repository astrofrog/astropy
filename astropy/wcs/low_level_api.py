import abc

import numpy as np

__all__ = ['BaseLowLevelWCS', 'FITSWCSLowLevelWCS']


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




class FITSWCSLowLevelWCS(BaseLowLevelWCS):
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
    def pixel_shape(self):
        return None

    @property
    def pixel_bounds(self):
        return None

    @property
    def world_axis_physical_types(self):
        raise NotImplementedError()

    @property
    def world_axis_units(self):
        return self._wcs.wcs.cunit

    @property
    def axis_correlation_matrix(self):

        # If there are any distortions present, we assume that there may be
        # correlations between all axes. Maybe if some distortions only apply
        # to the image plane we can improve this
        for distortion_attribute in ('sip', 'det2im1', 'det2im2'):
            if getattr(self._wcs, distortion_attribute):
                return np.ones((self.n_world, self.n_pixel), dtype=bool)

        # We flip the PC matrix with [::-1] because WCS and numpy index conventions
        # are reversed.
        pc = np.array(self._wcs.wcs.get_pc()[::-1, ::-1])
        axes = self._wcs.get_axis_types()[::-1]

        # There might be smarter ways to do this with matrix arithmetic
        for world in range(self.n_world):
            for pix in range(self.n_pixel):
                matrix = pc != 0

        # We now need to check specifically for celestial coordinates since
        # these can assume correlations because of spherical distortions.
        for world1 in range(self.n_world):
            if axes[world1]['coordinate_type'] == 'celestial':
                for world2 in range(self.n_world):
                    if world1 != world2:
                        if axes[world2]['coordinate_type'] == 'celestial':
                            matrix[world1] |= matrix[world2]
                            matrix[world2] |= matrix[world1]

        return matrix

    def pixel_to_world_values(self, *pixel_arrays):
        return self.all_pixel_to_world(*pixel_arrays, 0)

    def world_to_pixel_values(self, *world_arrays):
        return self.all_world_to_pixel(*world_arrays, 0)

    @property
    def world_axis_object_components(self):
        raise NotImplementedError()

    @property
    def world_axis_object_classes(self):
        raise NotImplementedError()
