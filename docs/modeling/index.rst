

.. plot::
   :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares

    rng = np.random.default_rng(0)
    y, x = np.mgrid[:128, :128]
    z = 2.0 * x**2 - 0.5 * y**2 + 1.5 * x * y - 1.0
    z += rng.normal(0.0, 0.1, z.shape) * 50000.

    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()

    # z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f

    def poly2d(params, x, y):
        a, b, c, d, e, f = params
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

    def residuals(params, x, y, z):
        return poly2d(params, x, y) - z

    p0 = np.zeros(6)

    res = least_squares(residuals, p0, args=(x_flat, y_flat, z_flat))
