# -*- coding: utf-8 -*-
"""

"""
import h5py
from astropy.wcs import WCS
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import foxsi_optics_calib
from matplotlib.colors import LogNorm
import scipy.optimize as opt

from astropy.modeling.models import custom_model, Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter


@u.quantity_input(offaxis_angle=u.deg, polar_angle=u.deg)
def find_ccd_image(offaxis_angle, polar_angle):
    """Given an offaxis angle and a polar angle, find the appropriate
        index in the hdf file. If more than one, returns the first.
    """
    offax_condition = foxsi_optics_calib.ccd_offaxis_angles.value == offaxis_angle.value
    polar_condition = foxsi_optics_calib.ccd_polar_angles.value == polar_angle.value
    index = np.where(offax_condition * polar_condition)[0][0]

    if index is None:
        raise ValueError(
            "No index found for {offaxis_angle} and {polar_angle}.".format(
                offaxis_angle=offaxis_angle, polar_angle=polar_angle))
    print("Index found {index}".format(index=index))
    img = foxsi_optics_calib.ccd_images[index, :, :]
    max_pixel = np.unravel_index(np.argmax(img), img.shape)
    # reduce the size of the array centered on the maximum
    sub_img_shape = [250, 250]
    extent = np.array([max_pixel[0] - sub_img_shape[0] / 2., max_pixel[0] + sub_img_shape[0] / 2.,
                       max_pixel[1] - sub_img_shape[1] / 2., max_pixel[1] + sub_img_shape[1] / 2.]).astype('int')
    sub_img = img[extent[0]:extent[1], extent[2]:extent[3]]
    return CCDImage(sub_img, offaxis_angle, polar_angle)


class CCDImage():
    """

    """
    def __init__(self, im, offaxis_angle, polar_angle, shift=False):
        self.im = im
        # define the wcs system
        self.w = WCS(naxis=2)
        self.w.wcs.crpix = np.unravel_index(np.argmax(self.im), self.im.shape)
        self.w.wcs.cdelt = foxsi_optics_calib.plate_scale(foxsi_optics_calib.CCD_PIXEL_PITCH).value * np.ones(2)
        self.w.wcs.crval = [0, 0]
        self.w.wcs.ctype = ["TAN", "TAN"]
        self.plate_scale = self.w.wcs.cdelt * u.arcsec
        self.polar_angle = polar_angle
        self.offaxis_angle = offaxis_angle
        pitch, yaw = (-np.sin(self.polar_angle) * self.offaxis_angle,
                      np.cos(self.polar_angle) * self.offaxis_angle)
        self.pitch = pitch
        self.yaw = yaw
        # offset is pitch then yaw
        self.offset = u.Quantity([pitch, yaw])
        #if shift:
        #    self.w.wcs.crval += [self.yaw.to('arcsec').value,
        #                         self.pitch.to('arcsec').value]

        x, y = np.meshgrid(*[np.arange(v) for v in self.im.shape]) * u.pixel
        self.xaxis, self.yaxis = self.w.wcs_pix2world(x, y, 1) * u.arcsec

    def _get_xlim(self, x1, x2):
        world = self.w.wcs_world2pix([[x1, 0], [x2, 0]], 1)
        return [world[0][0], world[1][0]] * u.arcsec

    def _get_ylim(self, y1, y2):
        world = self.w.wcs_world2pix([[0, y1], [0, y2]], 1)
        return [world[0][1], world[1][1]] * u.arcsec

    def plot_image(self, xlim=[-40, 40], ylim=[-40, 40],
             levels=[0.1, 1, 20, 30, 50, 75], ax=None, title=None,
                   colorbar=False, im=None):
        """Creates a plot of the PSF centered on the center. Levels are in
        percent."""
        if not ax:
            ax = plt.subplot(projection=self.w)
        if im is None:
            im = self.im
        norm_im = im / self.im.max()
        imshow = ax.imshow(norm_im, origin='lower', vmin=1e-4, cmap=plt.cm.viridis,
                      norm=LogNorm())
        ax.grid(color='white', ls='solid', alpha=0.5)
        ax.set_xlabel('X [arcsec]')
        ax.set_ylabel('Y [arcsec]')
        xlim_pix = self._get_xlim(xlim[0], xlim[1]) #+ self.offset[1]
        ylim_pix = self._get_ylim(ylim[0], ylim[1])  # + self.offset[0]
        ax.set_xlim(xlim_pix.value)
        ax.set_ylim(ylim_pix.value)
        if title is None:
            ax.set_title(
                "polar angle {0} offaxis angle {1}".format(self.polar_angle,
                                                           self.offaxis_angle))
        else:
            ax.set_title(title)

        cont = ax.contour(norm_im, levels=np.array(levels) / 100.,
                          colors='white', alpha=0.5)

        cont2 = ax.contour(norm_im, levels=np.array([0.5]),
                          colors='red', alpha=0.5)

        if colorbar:
            cbar = plt.colorbar(imshow, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('normalized DN')
            cbar.add_lines(cont)
            cbar.add_lines(cont2)
        return ax, imshow, cont

    def plot_cut(self, xlim=[-40, 40], ax=None, title=None, direction='x'):
        max_pixel = self.w.wcs.crpix
        if direction is 'x':
            data = self.im[int(max_pixel[0]), :]
            axis = self.xaxis[int(max_pixel[0]), :]
        if direction is 'y':
            data = self.im[:, int(max_pixel[0])]
            axis = self.yaxis[:, int(max_pixel[0])]
        if not ax:
            ax = plt.subplot()
        ax.plot(axis, data / data.max())
        if title is None:
            ax.set_title("{0} direction polar angle {1} offaxis angle {2}".format(direction,
                                                                                  self.polar_angle,
                                                                                  self.offaxis_angle))
        else:
            ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_xlabel('{0} [arcsec]'.format(direction.upper()))
        ax.set_ylabel('normalized DN')
        return ax
        #ax.set_yscale('log')


class CCDFitImage(CCDImage):

    def __init__(self, ccd_image):
        CCDImage.__init__(self, ccd_image.im, ccd_image.offaxis_angle,
                          ccd_image.polar_angle)
        self._fit()
        self.im_fit = self.fit_func(self.xaxis.value, self.yaxis.value)
        self.fwhm = self.calculate_fwhm()

    def _fit(self):
        amplitude = self.im.max()
        g_init = psf(amplitude1=amplitude * 1.0, x_stddev1=1, y_stddev1=1,
                     amplitude2=amplitude * 0.5, x_stddev2=5, y_stddev2=5,
                     amplitude3=amplitude * 0.5, x_stddev3=10, y_stddev3=10,
                     x_mean=0, y_mean=0, theta=0.0)
        fit = LevMarLSQFitter()
        fitted_model = fit(g_init, self.xaxis, self.yaxis, self.im)
        self.fit_func = fitted_model
        self.fwhm = self.calculate_fwhm()

    def calculate_fwhm(self):
        f = self.fit_func
        maximum = f.amplitude1 + f.amplitude2 + \
                  f.amplitude3
        half_maximum = maximum / 2.0
        f.offset -= half_maximum
        f_x = psf_x(f.y_mean, f)
        f_y = psf_y(f.x_mean, f)
        fwhm_x = 2 * opt.brentq(f_x, f.x_mean.value, f.x_mean.value + 5)
        fwhm_y = 2 * opt.brentq(f_y, f.y_mean.value, f.y_mean.value + 5)
        f.offset += half_maximum
        return np.array([fwhm_x, fwhm_y])

    def plot_fit(self, ax=None, title=None):
        self.plot_image(ax=ax, im=self.im_fit, title=title)
        pix = self.w.wcs_world2pix([[-35, -35]], 1)
        plt.text(pix[0][0], pix[0][1], 'fwhm=[{0:.2f}, {1:.2f}] arcsec'.format(self.fwhm[0], self.fwhm[1]), color='white')

    def plot_fit_and_image(self, ax=None, title=None, levels=[0.1, 1, 20, 30, 50, 75]):
        ax, imshow, cont = self.plot_image(ax=ax, title=title, levels=levels)
        cont = ax.contour(self.im_fit / self.im_fit.max(), levels=np.array(levels) / 100.,
                          colors='white', alpha=0.5)
        return ax, cont

    def plot_cut_fit(self, direction='x', ax=None, title=None, xlim=[-40, 40]):
        ax = self.plot_cut(ax=ax, title=title, direction=direction, xlim=xlim)
        x = np.linspace(xlim[0], xlim[1], 100)
        if direction.count('x'):
            f = psf_x(self.fit_func.y_mean, self.fit_func)
        if direction.count('y'):
            f = psf_y(self.fit_func.x_mean, self.fit_func)
        ax.plot(x, [f(this_x) / self.im_fit.max() for this_x in x], color='blue')

@custom_model
def psf(x, y,
        amplitude1=1.0, x_stddev1=1.0, y_stddev1=1,
        amplitude2=0.5, x_stddev2=5.0, y_stddev2=5,
        amplitude3=0.1, x_stddev3=10., y_stddev3=10,
        x_mean=0., y_mean=0,
        theta=0.0, offset=0):
    """The model of the FOXSI PSF"""
    if amplitude1 < 0:
        amplitude1 = 1e12
    if amplitude2 < 0:
        amplitude2 = 1e12
    if amplitude3 < 0:
        amplitude3 = 1e12

    g1 = Gaussian2D(amplitude=amplitude1, x_mean=x_mean, y_mean=y_mean, x_stddev=x_stddev1, y_stddev=y_stddev1, theta=theta)
    g2 = Gaussian2D(amplitude=amplitude2, x_mean=x_mean, y_mean=y_mean, x_stddev=x_stddev2, y_stddev=y_stddev2, theta=theta)
    g3 = Gaussian2D(amplitude=amplitude3, x_mean=x_mean, y_mean=y_mean, x_stddev=x_stddev3, y_stddev=y_stddev3, theta=theta)
    y1 = g1.evaluate(x, y, amplitude=amplitude1, x_mean=x_mean, y_mean=y_mean, x_stddev=x_stddev1, y_stddev=y_stddev1, theta=theta)
    y2 = g2.evaluate(x, y, amplitude=amplitude2, x_mean=x_mean, y_mean=y_mean, x_stddev=x_stddev1, y_stddev=y_stddev1, theta=theta)
    y3 = g3.evaluate(x, y, amplitude=amplitude3, x_mean=x_mean, y_mean=y_mean, x_stddev=x_stddev3, y_stddev=y_stddev3, theta=theta)
    return y1 + y2 + y3 + offset


def psf_x(y_value, psf_function):
    """The PSF as a function of x for a given y."""
    return lambda x: psf_function(x, y_value)


def psf_y(x_value, psf_function):
    """The PSF as a function of x for a given y."""
    return lambda y: psf_function(x_value, y)


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

