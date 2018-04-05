import astropy.units as u
from foxsi_optics_calib.ccd.ccd import AndorCCDPsfImage
import matplotlib.pyplot as plt

focal_length = 2 * u.m

f = '/Users/Kamilobu/Desktop/X10Test/FOXSI3_X10-Test_CCD_T2Sx6_10kV_0p02mA_0mmZ.fits'
d = '/Users/Kamilobu/Desktop/X10Test/Dark1_FOXSI3_X10-Test_CCD_T2Sx6_10kV_0p02mA_+15mmZ.fits'

# for just a general image with no assumptions
psf_image = AndorCCDPsfImage(f, d, focal_length)

fig = plt.figure(figsize=(10, 4))

ax = fig.add_subplot(1, 1, 1, projection=psf_image.wcs)
psf_image.plot_image(ax=ax, title='On-axis [data]')

plt.savefig('plotting_psf_file.pdf')
