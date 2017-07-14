import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import astropy.units as u
from foxsi_optics_calib.ccd.ccd import find_ccd_image

fig = plt.figure(figsize=(16, 4))

offaxis_angles = [0, 3, 5, 7, 9] * u.arcmin

for i, offaxis_angle in enumerate(offaxis_angles):
    foxsi_ccd_image = find_ccd_image(offaxis_angle, 0 * u.arcsec)
    ax = fig.add_subplot(1, len(offaxis_angles), i+1, projection=foxsi_ccd_image.w)
    ax.coords.grid(color='white', alpha=0.5, linestyle='solid')
    im, cont = foxsi_ccd_image.plot_image(ax=ax, title=offaxis_angle)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('normalized DN')
cbar.add_lines(cont)

plt.savefig('figure_2.pdf')
print('Done')
