# Licensed under a 3-clause BSD style license - see LICENSE.rst


# This file defines the AngleFormatterLocator class which is a class that
# provides both a method for a formatter and one for a locator, for a given
# label spacing. The advantage of keeping the two connected is that we need to
# make sure that the formatter can correctly represent the spacing requested and
# vice versa. For example, a format of dd:mm cannot work with a tick spacing
# that is not a multiple of one arcminute.

import re
import warnings

import numpy as np

from matplotlib import rcParams

from ... import units as u
from ...units import UnitsError
from ...coordinates import Angle

DMS_RE = re.compile('^dd(:mm(:ss(.(s)+)?)?)?$')
HMS_RE = re.compile('^hh(:mm(:ss(.(s)+)?)?)?$')
DDEC_RE = re.compile('^d(.(d)+)?$')
DMIN_RE = re.compile('^m(.(m)+)?$')
DSEC_RE = re.compile('^s(.(s)+)?$')
SCAL_RE = re.compile('^x(.(x)+)?$')


class BaseFormatterLocator:
    """
    A joint formatter/locator
    """

    def __init__(self, values=None, number=None, spacing=None, format=None):

        if (values, number, spacing).count(None) < 2:
            raise ValueError("At most one of values/number/spacing can be specifed")

        if values is not None:
            self.values = values
        elif number is not None:
            self.number = number
        elif spacing is not None:
            self.spacing = spacing
        else:
            self.number = 5

        self.format = format

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        if not isinstance(values, u.Quantity) or (not values.ndim == 1):
            raise TypeError("values should be an astropy.units.Quantity array")
        self._number = None
        self._spacing = None
        self._values = values

    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, number):
        self._number = number
        self._spacing = None
        self._values = None

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, spacing):
        self._number = None
        self._spacing = spacing
        self._values = None

    def minor_locator(self, spacing, frequency, value_min, value_max):
        if self.values is not None:
            return [] * self._unit

        minor_spacing = spacing.value / frequency
        values = self._locate_values(value_min, value_max, minor_spacing)
        index = np.where((values % frequency) == 0)
        index = index[0][0]
        values = np.delete(values, np.s_[index::frequency])
        return values * minor_spacing * self._unit

    @property
    def format_unit(self):
        return self._format_unit

    @format_unit.setter
    def format_unit(self, unit):
        self._format_unit = u.Unit(unit)

    @staticmethod
    def _locate_values(value_min, value_max, spacing):
        imin = np.ceil(value_min / spacing)
        imax = np.floor(value_max / spacing)
        values = np.arange(imin, imax + 1, dtype=int)
        return values


class AngleFormatterLocator(BaseFormatterLocator):
    """
    A joint formatter/locator
    """

    def __init__(self, values=None, number=None, spacing=None, format=None,
                 unit=None, decimal=None, format_unit=None):

        if unit is None:
            unit = u.degree

        if format_unit is None:
            format_unit = unit

        if format_unit not in (u.degree, u.hourangle, u.hour):
            if decimal is False:
                raise UnitsError("Units should be degrees or hours when using non-decimal (sexagesimal) mode")

        self._unit = unit
        self._format_unit = format_unit or unit
        self._decimal = decimal
        self._sep = None
        super().__init__(values=values, number=number, spacing=spacing,
                         format=format)

    @property
    def decimal(self):
        decimal = self._decimal
        if self.format_unit not in (u.degree, u.hourangle, u.hour):
            if self._decimal is None:
                decimal = True
            elif self._decimal is False:
                raise UnitsError("Units should be degrees or hours when using non-decimal (sexagesimal) mode")
        elif self._decimal is None:
            decimal = False
        return decimal


    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, spacing):
        if spacing is not None and (not isinstance(spacing, u.Quantity) or
                                    spacing.unit.physical_type != 'angle'):
            raise TypeError("spacing should be an astropy.units.Quantity "
                            "instance with units of angle")
        self._number = None
        self._spacing = spacing
        self._values = None

    @property
    def sep(self):
        return self._sep

    @sep.setter
    def sep(self, separator):
        self._sep = separator

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, value):

        self._format = value

        if value is None:
            return

        if DMS_RE.match(value) is not None:
            self._decimal = False
            self._format_unit = u.degree
            if '.' in value:
                self._precision = len(value) - value.index('.') - 1
                self._fields = 3
            else:
                self._precision = 0
                self._fields = value.count(':') + 1
        elif HMS_RE.match(value) is not None:
            self._decimal = False
            self._format_unit = u.hourangle
            if '.' in value:
                self._precision = len(value) - value.index('.') - 1
                self._fields = 3
            else:
                self._precision = 0
                self._fields = value.count(':') + 1
        elif DDEC_RE.match(value) is not None:
            self._decimal = True
            self._format_unit = u.degree
            self._fields = 1
            if '.' in value:
                self._precision = len(value) - value.index('.') - 1
            else:
                self._precision = 0
        elif DMIN_RE.match(value) is not None:
            self._decimal = True
            self._format_unit = u.arcmin
            self._fields = 1
            if '.' in value:
                self._precision = len(value) - value.index('.') - 1
            else:
                self._precision = 0
        elif DSEC_RE.match(value) is not None:
            self._decimal = True
            self._format_unit = u.arcsec
            self._fields = 1
            if '.' in value:
                self._precision = len(value) - value.index('.') - 1
            else:
                self._precision = 0
        else:
            raise ValueError("Invalid format: {0}".format(value))

        if self.spacing is not None and self.spacing < self.base_spacing:
            warnings.warn("Spacing is too small - resetting spacing to match format")
            self.spacing = self.base_spacing

        if self.spacing is not None:

            ratio = (self.spacing / self.base_spacing).decompose().value
            remainder = ratio - np.round(ratio)

            if abs(remainder) > 1.e-10:
                warnings.warn("Spacing is not a multiple of base spacing - resetting spacing to match format")
                self.spacing = self.base_spacing * max(1, round(ratio))

    @property
    def base_spacing(self):

        if self.decimal:

            spacing = self._format_unit / (10. ** self._precision)

        else:

            if self._fields == 1:
                spacing = 1. * u.degree
            elif self._fields == 2:
                spacing = 1. * u.arcmin
            elif self._fields == 3:
                if self._precision == 0:
                    spacing = 1. * u.arcsec
                else:
                    spacing = u.arcsec / (10. ** self._precision)

        if self._format_unit is u.hourangle:
            spacing *= 15

        return spacing

    def locator(self, value_min, value_max):

        if self.values is not None:

            # values were manually specified
            return self.values, 1.1 * u.arcsec

        else:

            # In the special case where value_min is the same as value_max, we
            # don't locate any ticks. This can occur for example when taking a
            # slice for a cube (along the dimension sliced).
            if value_min == value_max:
                return [] * u.deg, 0 * u.arcsec

            if self.spacing is not None:

                # spacing was manually specified
                spacing_deg = self.spacing.to_value(u.degree)

            elif self.number is not None:

                # number of ticks was specified, work out optimal spacing

                # first compute the exact spacing
                dv = abs(float(value_max - value_min)) / self.number * u.degree

                if self.format is not None and dv < self.base_spacing:
                    # if the spacing is less than the minimum spacing allowed by the format, simply
                    # use the format precision instead.
                    spacing_deg = self.base_spacing.to_value(u.degree)
                else:
                    # otherwise we clip to the nearest 'sensible' spacing
                    if self._format_unit is u.degree:
                        from .utils import select_step_degree
                        spacing_deg = select_step_degree(dv).to_value(u.degree)
                    else:
                        from .utils import select_step_hour
                        spacing_deg = select_step_hour(dv).to_value(u.degree)

            # We now find the interval values as multiples of the spacing and
            # generate the tick positions from this.
            values = self._locate_values(value_min, value_max, spacing_deg)
            return values * spacing_deg * u.degree, spacing_deg * u.degree

    def formatter(self, values, spacing, format='auto'):

        if not isinstance(values, u.Quantity) and values is not None:
            raise TypeError("values should be a Quantities array")

        if len(values) > 0:
            decimal = self.decimal
            unit = self._format_unit

            if self.format is None:
                if decimal:
                    # Here we assume the spacing can be arbitrary, so for example
                    # 1.000223 degrees, in which case we don't want to have a
                    # format that rounds to degrees. So we find the number of
                    # decimal places we get from representing the spacing as a
                    # string in the desired units. The easiest way to find
                    # the smallest number of decimal places required is to
                    # format the number as a decimal float and strip any zeros
                    # from the end. We do this rather than just trusting e.g.
                    # str() because str(15.) == 15.0. We format using 10 decimal
                    # places by default before stripping the zeros since this
                    # corresponds to a resolution of less than a microarcecond,
                    # which should be sufficient.
                    spacing = spacing.to_value(unit)
                    fields = 0
                    precision = len("{0:.10f}".format(spacing).replace('0', ' ').strip().split('.', 1)[1])
                else:
                    spacing = spacing.to_value(unit / 3600)
                    if spacing >= 3600:
                        fields = 1
                        precision = 0
                    elif spacing >= 60:
                        fields = 2
                        precision = 0
                    elif spacing >= 1:
                        fields = 3
                        precision = 0
                    else:
                        fields = 3
                        precision = -int(np.floor(np.log10(spacing)))
            else:
                fields = self._fields
                precision = self._precision

            if decimal:
                sep = None
                fmt = None

                decimal = False
                sep = 'fromunit'
                if format == 'latex' or (format == 'auto' and rcParams['text.usetex']):
                    fmt = 'latex'
                else:
                    fmt = 'unicode'

            elif self._sep is not None:
                sep = self._sep
                fmt = None
            else:
                sep = 'fromunit'
                if unit == u.degree:
                    if format == 'latex' or (format == 'auto' and rcParams['text.usetex']):
                        fmt = 'latex'
                    else:
                        sep = ('\xb0', "'", '"')
                        fmt = None
                else:
                    if format == 'ascii':
                        fmt = None
                    elif format == 'latex' or (format == 'auto' and rcParams['text.usetex']):
                        fmt = 'latex'
                    else:
                        # Here we still use LaTeX but this is for Matplotlib's
                        # LaTeX engine - we can't use fmt='latex' as this
                        # doesn't produce LaTeX output that respects the fonts.
                        sep = (r'$\mathregular{^h}$', r'$\mathregular{^m}$', r'$\mathregular{^s}$')
                        fmt = None

            angles = Angle(values)
            string = angles.to_string(unit=unit,
                                      precision=precision,
                                      decimal=decimal,
                                      fields=fields,
                                      sep=sep,
                                      format=fmt).tolist()

            return string
        else:
            return []


class ScalarFormatterLocator(BaseFormatterLocator):
    """
    A joint formatter/locator
    """

    def __init__(self, values=None, number=None, spacing=None, format=None, unit=None):
        if unit is not None:
            self._unit = unit
            self._format_unit = unit
        elif spacing is not None:
            self._unit = spacing.unit
            self._format_unit = spacing.unit
        elif values is not None:
            self._unit = values.unit
            self._format_unit = values.unit
        super().__init__(values=values, number=number, spacing=spacing,
                         format=format)

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, spacing):
        if spacing is not None and not isinstance(spacing, u.Quantity):
            raise TypeError("spacing should be an astropy.units.Quantity instance")
        self._number = None
        self._spacing = spacing
        self._values = None

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, value):

        self._format = value

        if value is None:
            return

        if SCAL_RE.match(value) is not None:
            if '.' in value:
                self._precision = len(value) - value.index('.') - 1
            else:
                self._precision = 0

            if self.spacing is not None and self.spacing < self.base_spacing:
                warnings.warn("Spacing is too small - resetting spacing to match format")
                self.spacing = self.base_spacing

            if self.spacing is not None:

                ratio = (self.spacing / self.base_spacing).decompose().value
                remainder = ratio - np.round(ratio)

                if abs(remainder) > 1.e-10:
                    warnings.warn("Spacing is not a multiple of base spacing - resetting spacing to match format")
                    self.spacing = self.base_spacing * max(1, round(ratio))

        elif not value.startswith('%'):
            raise ValueError("Invalid format: {0}".format(value))

    @property
    def base_spacing(self):
        return self._format_unit / (10. ** self._precision)

    def locator(self, value_min, value_max):

        if self.values is not None:

            # values were manually specified
            return self.values, 1.1 * self._unit
        else:

            # In the special case where value_min is the same as value_max, we
            # don't locate any ticks. This can occur for example when taking a
            # slice for a cube (along the dimension sliced).
            if value_min == value_max:
                return [] * self._unit, 0 * self._unit

            if self.spacing is not None:

                # spacing was manually specified
                spacing = self.spacing.to_value(self._unit)

            elif self.number is not None:

                # number of ticks was specified, work out optimal spacing

                # first compute the exact spacing
                dv = abs(float(value_max - value_min)) / self.number

                if self.format is not None and (not self.format.startswith('%')) and dv < self.base_spacing.value:
                    # if the spacing is less than the minimum spacing allowed by the format, simply
                    # use the format precision instead.
                    spacing = self.base_spacing.to_value(self._unit)
                else:
                    from .utils import select_step_scalar
                    spacing = select_step_scalar(dv)

            # We now find the interval values as multiples of the spacing and
            # generate the tick positions from this

            values = self._locate_values(value_min, value_max, spacing)
            return values * spacing * self._unit, spacing * self._unit

    def formatter(self, values, spacing, format='auto'):

        if len(values) > 0:
            if self.format is None:
                if spacing.value < 1.:
                    precision = -int(np.floor(np.log10(spacing.value)))
                else:
                    precision = 0
            elif self.format.startswith('%'):
                return [(self.format % x.value) for x in values]
            else:
                precision = self._precision

            return [("{0:." + str(precision) + "f}").format(x.to_value(self._format_unit)) for x in values]

        else:
            return []
