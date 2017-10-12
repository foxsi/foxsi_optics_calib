
import astropy.units as u
from foxsi_optics_calib.ccd.ccd import AndorCCDImage, AndorCCDPsfImage, AndorCCDPsfFitImage

focal_length = 2 * u.m

f = '/Users/schriste/Desktop/2017-Oct/X4/CCD/rawdata/PSF/FOXSI3_X4_25mmz_CCD_10kV_0p02mA_+1arcminX_+1arcminY.fits'

# for just a general image with no assumptions
img = AndorCCDImage(f, focal_length)

print(img)

psf_img = AndorCCDPsfImage(f, focal_length)

print(psf_img)

psffit_img = AndorCCDPsfFitImage(f, focal_length)

print(psffit_img)