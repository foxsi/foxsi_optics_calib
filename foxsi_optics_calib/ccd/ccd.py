"""

"""
import h5py
from astropy.wcs import WCS
import astropy.units as u
import numpy as np

import foxsi_optics_calib


class CCDImage():
    """

    """
    def __init__(self, index, shift=False):
        # save the image
        self.im = foxsi_optics_calib.ccd_images[index, :, :]
        # define the wcs system
        self.w = WCS(naxis=2)
        self.w.wcs.crpix = np.unravel_index(np.argmax(self.im), self.im.shape)
        self.w.wcs.cdelt = foxsi_optics_calib.plate_scale(CCD_PIXEL_PITCH).value * np.ones(2)
        self.w.wcs.crval = [0, 0]
        self.w.wcs.ctype = ["TAN", "TAN"]
        self.plate_scale = self.w.wcs.cdelt * u.arcsec
        self.polar_angle = foxsi_optics_calib.ccd_polar_angles[ind]
        self.offaxis_angle = foxsi_optics_calib.ccd_offaxis_angles[ind]
        pitch, yaw = (-np.sin(self.polar_angle) * self.offaxis_angle,
                      np.cos(self.polar_angle) * self.offaxis_angle)
        self.pitch = pitch
        self.yaw = yaw
        # offset is pitch then yaw
        self.offset = u.Quantity([pitch, yaw])
        if shift:
            self.w.wcs.crval += [self.yaw.to('arcsec').value,
                                 self.pitch.to('arcsec').value]

        x, y = np.meshgrid(*[np.arange(v) for v in self.im.shape]) * u.pixel
        self.xaxis, self.yaxis = self.w.wcs_pix2world(x, y, 1) * u.arcsec

    def _get_xlim(self, x1, x2):
        world = self.w.wcs_world2pix([[x1, 0], [x2, 0]], 1)
        return [world[0][0], world[1][0]] * u.arcsec

    def _get_ylim(self, y1, y2):
        world = self.w.wcs_world2pix([[0, y1], [0, y2]], 1)
        return [world[0][1], world[1][1]] * u.arcsec

    def plot(self, xlim=[-40, 40], ylim=[-40, 40],
             levels=[0.1, 1, 20, 30, 50, 75]):
        norm_im = self.im / self.im.max()
        ax = plt.subplot(projection=self.w)
        c = ax.imshow(norm_im, origin='lower', vmin=1e-4, cmap=plt.cm.viridis,
                      norm=LogNorm())
        ax.grid(color='white', ls='solid', alpha=0.5)
        ax.set_xlabel('X [arcsec]')
        ax.set_ylabel('Y [arcsec]')
        xlim_pix = self._get_xlim(xlim[0], xlim[1]) + self.offset[1]
        ylim_pix = self._get_ylim(ylim[0], ylim[1])  # + self.offset[0]
        ax.set_xlim(xlim_pix.value)
        ax.set_ylim(ylim_pix.value)
        ax.set_title(
            "polar angle {0} offaxis angle {1}".format(self.polar_angle,
                                                       self.offaxis_angle))
        cont = ax.contour(norm_im, levels=np.array(levels) / 100.,
                          colors='white', alpha=0.5)
        cbar = plt.colorbar(c, ax=ax)
        cbar.ax.set_ylabel('normalized DN')
        cbar.add_lines(cont)


def load_hdf(filename):
    """Function to load data saved in an hdf5 file."""
    h = h5py.File(filename, 'r+')
    images = h['X2/ccd_images']

    polar_angles = u.Quantity(h['meta/polar_angle'][...], h['meta/polar_angle'].attrs['units'])
    offaxis_angles = u.Quantity(h['meta/offaxis_angle'][...], h['meta/offaxis_angle'].attrs['units'])
    nimages = images.shape[0]
    print("There are {0} polar angles {1} offaxis_angles".format(
        len(polar_angles), len(offaxis_angles)))
    print("The dimensions of the image cube is {0}".format(images.shape))

    return images

