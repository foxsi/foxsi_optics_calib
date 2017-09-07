import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from foxsi_optics_calib.ccd.ccd import find_ccd_image, CCDFitImage

# find the fwhm at all polar angles and average them for each offaxis angle.

offaxis_angles = [-9, -7, -5, -3, -2, -1.5, -1, 0.5, 0, 0.5, 1, 1.5, 2, 3, 5, 7, 9] * u.arcmin
polar_angles = [0, 45, 90, 135] * u.deg

fwhm_x = np.zeros_like(offaxis_angles.value)
fwhm_y = np.zeros_like(offaxis_angles.value)

for i, this_angle in enumerate(offaxis_angles):
    these_fwhm_x = np.zeros_like(polar_angles.value)
    these_fwhm_y = np.zeros_like(polar_angles.value)
    for j, this_polar_angle in enumerate(polar_angles):
        print("offaxis={0}, polar={1}".format(this_angle, this_polar_angle))
        foxsi_ccd_image = find_ccd_image(this_angle, this_polar_angle)
        fit_ccd_image = CCDFitImage(foxsi_ccd_image)
        these_fwhm_x[j] = fit_ccd_image.fwhm[0]
        these_fwhm_y[j] = fit_ccd_image.fwhm[1]
        # create a plot for each fit
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 3, 1, projection=fit_ccd_image.w)
        fit_ccd_image.plot_image(ax=ax1)
        ax2 = fig.add_subplot(1, 3, 2, projection=fit_ccd_image.w)
        fit_ccd_image.plot_fit(ax=ax2)
        ax3 = fig.add_subplot(1, 3, 3, projection=fit_ccd_image.w)
        fit_ccd_image.plot_fit_and_image(ax=ax3)
        plt.savefig("fit_{0}arcmin_{1}arcmin.pdf".format(this_angle.value, this_polar_angle.value))
        plt.close()
    fwhm_x[i] = np.nanmean(these_fwhm_x)
    fwhm_y[i] = np.nanmean(these_fwhm_y)

print(fwhm_y)
print(fwhm_x)
plt.plot(offaxis_angles, fwhm_x, label='X')
plt.plot(offaxis_angles, fwhm_y, label='Y')
plt.plot(offaxis_angles, np.sqrt(fwhm_x ** 2 + fwhm_y ** 2), label='effective')
plt.xlabel('Offaxis angle [arcmin]')
plt.ylabel('FWHM [arcsec]')
plt.legend()
plt.savefig('fwhm_vs_offaxis1d.pdf')
