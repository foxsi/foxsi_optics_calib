"""

"""
import astropy.units as u
import numpy as np
import h5py

# FOXSI-R Constants
FOCAL_LENGTH = 2 * u.m
PIXEL_NUMBER = 128
STRIP_PITCH = {'cdte': 60 * u.micron, 'si': 75 * u.micron}

# CCD Parameters
CCD_PIXEL_PITCH = 13.5 * u.micron

HDF_CCD_FILE = '/Users/schriste/Documents/FOXSI-R/Optics PSF Calibration/foxsi2_ccd_corrected.hdf5'

print("Loading file {0}".format(HDF_CCD_FILE))
hdf_ccd = h5py.File(HDF_CCD_FILE, 'r+')
ccd_images = hdf_ccd['X2/ccd_images']
ccd_polar_angles = u.Quantity(hdf_ccd['meta/polar_angle'][...], hdf_ccd['meta/polar_angle'].attrs['units'])
ccd_offaxis_angles = u.Quantity(hdf_ccd['meta/offaxis_angle'][...], hdf_ccd['meta/offaxis_angle'].attrs['units'])
ccd_nimages = ccd_images.shape[0]


@u.quantity_input(pixel_pitch=u.mm)
def plate_scale(pixel_pitch):
    """Given a pixel pitch provide the plate scale."""
    return np.arctan(pixel_pitch / FOCAL_LENGTH).to(u.arcsec)