import matplotlib.pyplot as plt
import astropy.units as u
from foxsi_optics_calib.ccd.ccd import find_ccd_image, CCDFitImage

foxsi_ccd_image = find_ccd_image(0 * u.arcsec, 0 * u.arcsec)
fit_ccd_image = CCDFitImage(foxsi_ccd_image)

fig = plt.figure(figsize=(10, 4))

ax = fig.add_subplot(1, 3, 1, projection=foxsi_ccd_image.w)
fit_ccd_image.plot_image(ax=ax, title='On-axis [data]')

ax = fig.add_subplot(1, 3, 2, projection=foxsi_ccd_image.w)
fit_ccd_image.plot_fit(ax=ax, title='On-axis [fit]')

ax = fig.add_subplot(1, 3, 3, projection=foxsi_ccd_image.w)
fit_ccd_image.plot_fit_and_image(ax=ax, title='On-axis [data, fit(contours)]')

print("center = {0}, {1}".format(fit_ccd_image.fit_func.x_mean, fit_ccd_image.fit_func.y_mean))

plt.savefig('figure_fitting_psf.pdf')
