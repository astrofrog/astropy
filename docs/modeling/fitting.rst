
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

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.filterwarnings('ignore', message='Model is linear in parameters',
                                category=AstropyUserWarning)
        p = fit_p(p_init, x, y, z)

    # Plot the data with the best-fit model
    fig, axs = plt.subplots(figsize=(8, 2.5), ncols=3)
    ax1 = axs[0]
    ax1.imshow(z, origin='lower', interpolation='nearest', vmin=-1e4, vmax=5e4)
    ax1.set_title("Data")
    ax2 = axs[1]
    ax2.imshow(p(x, y), origin='lower', interpolation='nearest', vmin=-1e4,
               vmax=5e4)
    ax2.set_title("Model")
    ax3 = axs[2]
    ax3.imshow(z - p(x, y), origin='lower', interpolation='nearest', vmin=-1e4,
               vmax=5e4)
    ax3.set_title("Residual")
