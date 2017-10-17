import astropy.units as u
from foxsi_optics_calib.ccd.ccd import AndorCCDPsfImage
import matplotlib.pyplot as plt

focal_length = 2 * u.m

f = '/Users/schriste/Desktop/2017-Oct/X4/CCD/rawdata/PSF/FOXSI3_X4_25mmz_CCD_10kV_0p02mA_+1arcminX_+1arcminY.fits'

# for just a general image with no assumptions
psf_image = AndorCCDPsfImage(f, focal_length)

fig = plt.figure(figsize=(10, 4))

ax = fig.add_subplot(1, 1, 1, projection=psf_image.wcs)
psf_image.plot_image(ax=ax, title='On-axis [data]')

plt.savefig('plotting_psf_file.pdf')
