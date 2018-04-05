
import astropy.units as u
from foxsi_optics_calib.ccd.ccd import AndorCCDImage, AndorCCDPsfImage, AndorCCDPsfFitImage

focal_length = 2 * u.m

f = '/Users/Kamilobu/Desktop/X10Test/FOXSI3_X10-Test_CCD_T2Sx6_10kV_0p02mA_0mmZ.fits'
d = '/Users/Kamilobu/Desktop/X10Test/Dark1_FOXSI3_X10-Test_CCD_T2Sx6_10kV_0p02mA_+15mmZ.fits'

# for just a general image with no assumptions
img = AndorCCDImage(f, d, focal_length)

print(img)

psf_img = AndorCCDPsfImage(f, d, focal_length)

print(psf_img)

psffit_img = AndorCCDPsfFitImage(f, d, focal_length)

print(psffit_img)
