

.. plot::
   :include-source:

    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.modeling import models, fitting
    from astropy.utils.exceptions import AstropyUserWarning

    # Generate fake data
    rng = np.random.default_rng(0)
    y, x = np.mgrid[:128, :128]
    z = 2. * x ** 2 - 0.5 * x ** 2 + 1.5 * x * y - 1.
    z += rng.normal(0., 0.1, z.shape) * 50000.

    # Fit the data using astropy.modeling
    p_init = models.Polynomial2D(degree=2)
    fit_p = fitting.LMLSQFitter()

    p = fit_p(p_init, x, y, z)
