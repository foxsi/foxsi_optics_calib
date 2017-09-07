from foxsi_optics_calib.psf import psf2d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import astropy.units as u

angle = (90 * u.deg).to('rad').value
amplitude = 1
psf = psf2d(amplitude1=amplitude, x_stddev1=1, y_stddev1=1*5,
                       amplitude2=amplitude/10., x_stddev2=5, y_stddev2=5*5,
                       amplitude3=amplitude/20., x_stddev3=10, y_stddev3=10*5,
                       x_mean=0, y_mean=0, theta=angle, offset=0.0)


x, y = np.meshgrid(*[np.arange(v)-50 for v in (100, 100)])

plt.imshow(psf(x, y), origin='lower', vmin=1e-4, cmap=plt.cm.viridis, norm=LogNorm())
plt.show()