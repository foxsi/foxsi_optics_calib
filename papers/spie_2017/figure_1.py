import matplotlib.pyplot as plt
import astropy.units as u

from foxsi_optics_calib.ccd.ccd import find_ccd_image

fig = plt.figure(figsize=(13, 4))

foxsi_ccd_image = find_ccd_image(0 * u.arcsec, 0 * u.arcsec)

ax = fig.add_subplot(1, 3, 1, projection=foxsi_ccd_image.w)
foxsi_ccd_image.plot_image(ax=ax, title='On-axis', colorbar=True)

ax = fig.add_subplot(1, 3, 2)
foxsi_ccd_image.plot_cut(ax=ax, direction='x', title='X')

ax = fig.add_subplot(1, 3, 3)
foxsi_ccd_image.plot_cut(ax=ax, direction='y', title='Y')

plt.savefig('figure_1.pdf')
