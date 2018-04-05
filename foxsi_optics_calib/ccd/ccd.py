# -*- coding: utf-8 -*-
"""
Code to read and analyze data from the CCD camera
"""
import h5py
from astropy.wcs import WCS
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.optimize as opt
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.io import fits as pyfits
from astropy.nddata import CCDData
from copy import deepcopy
import os.path

import foxsi_optics_calib
from foxsi_optics_calib.psf import psf2d, psf_x, psf_y, calculate_best_guess_params, PSF2DModel

CCD_PLATE_SCALE = foxsi_optics_calib.plate_scale(foxsi_optics_calib.FOCAL_LENGTH).value


class AndorCCDImage(CCDData):
    """
    A generic class to handle fits files created by the Andor CCD.
    This is a general class to inspect a file and
    makes no assumptions about the image. This class inherits from
    `~astropy.nddata.CCDData`.

    Parameters
    ----------
    data : `~numpy.ndarray`
        A 2d ndarray containing the image
    meta : dict-like object or None, optional
        Metadata for this object.
    wcs : `~astropy.wcs.WCS`
        WCS-object containing the world coordinate system for the data.
        Calculated automatically assuming the pixel size of the Andor camera
        and the distance to the focal plane provided.
    unit : `astropy.units.Unit`
        The units of the data.
    mask : ~numpy.ndarray` or None, optional
        Mask for the data, given as a boolean Numpy array with a shape
        matching that of the data. The values must be False where the data
        is valid and True when it is not (like Numpy masked arrays).
    filename : str
        The filename of the original data.

    Examples
    --------
    >>> from foxsi_optics_calib.ccd import AndorCCDImage
    >>> import astropy.units as u
    >>> ccd = AndorCCDImage('filename.fits', 2 * u.m)
    """

    def __init__(self, filename, darkfile, distance):

        fits = pyfits.open(filename)
        darks = pyfits.open(darkfile)
        # compress all images into one image by average all the pixels
        if len(fits[0].data.shape) == 3:
            print("Found {0} exposures. Averaging...".format(
                fits[0].data.shape[0]))
            data = np.average(fits[0].data, axis=0) - np.average(darks[0].data, axis=0)
        else:
            data = fits[0].data - darks[0].data

        # create the wcs information
        w = WCS(naxis=2)
        w.wcs.crpix = [0, 0]
        plate_scale = foxsi_optics_calib.plate_scale(distance).to('arcsec')
        w.wcs.cdelt = plate_scale.value * np.ones(2)
        w.wcs.crval = [0, 0]
        w.wcs.ctype = ["TAN", "TAN"]
        CCDData.__init__(self, data, wcs=w, unit='adu',
                         header=deepcopy(fits[0].header))
        # save the name of the filename
        self.filename = os.path.basename(filename)

        x, y = np.meshgrid(*[np.arange(v) for v in self.data.shape]) * u.pixel
        self.xaxis, self.yaxis = self.wcs.wcs_pix2world(x, y, 1) * u.arcsec
        self.xlim = np.floor(
            [np.min(self.xaxis).value, np.max(self.xaxis).value])
        self.ylim = np.floor(
            [np.min(self.yaxis).value, np.max(self.yaxis).value])

    def plot_image(self, levels=[0.1, 1, 20, 30, 50, 75], vmin=1e-4, vmax=1,
                   ax=None, title=None, colorbar=False, norm=False):
        """Create a standard plot of the image.

        Parameters
        ----------
        levels : `~numpy.ndarray`
            Percent of the level contours to display in percent.
        vmin : float
            Minimum data to display
        vmax : float
            Maximum data to display
        ax : `~matplotlib.axes` or None
            If provided the image will be plotted on the given axes. Else the
            current matplotlib axes will be used.
        title : str, optional
            The title to display on the plot
        colorbar : bool
            Whether to display the colorbar
        norm : bool
            Whether to normalize the data to the maximum pixel
        """

        if not ax:
            ax = plt.subplot(projection=self.wcs)
        norm_im = self.data / self.data.max()
        norm = LogNorm()
        if not vmin:
            vmin = 1e-4
        if not vmax:
            vmax = 1

        imshow = ax.imshow(norm_im, origin='lower', vmin=vmin, vmax=vmax,
                           cmap=plt.cm.viridis)
        ax.grid(color='white', ls='solid', alpha=0.5)
        ax.set_xlabel('X [arcsec]')
        ax.set_ylabel('Y [arcsec]')
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.autoscale(False)

        if title is None:
            ax.set_title("{0}".format(self.filename))
        else:
            ax.set_title(title)

        cont = ax.contour(norm_im, levels=np.array(levels) / 100.,
                          colors='white', alpha=0.5)

        if colorbar:
            cbar = plt.colorbar(imshow, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('normalized ADU')
            cbar.add_lines(cont)
        ax.autoscale(False)
        return ax, imshow, cont

    def set_xlim(self, x1, x2):
        """Set the xlim provided in world coordinates when plotting."""
        world = self.wcs.wcs_world2pix([[x1, 0], [x2, 0]], 1)
        self.xlim = [world[0][0], world[1][0]]

    def set_ylim(self, y1, y2):
        """Set the ylim provided in world coordinates when plotting."""
        world = self.wcs.wcs_world2pix([[0, y1], [0, y2]], 1)
        self.ylim = [world[0][1], world[1][1]]

    def read(filename):
        pass

    def write(filename):
        pass


class AndorCCDPsfImage(AndorCCDImage):
    """
        A class to handle fits files created by the Andor CCD that contain
        a point spread function (PSF) image. This means it assumes that there is a maximum
        pixel and that is where the coordinate system should be centered.
        Also provides functions to measure properties of the PSF.
         This class inherits from `~astropy.nddata.CCDData`.

        Parameters
        ----------
        data : `~numpy.ndarray`
            A 2d ndarray containing the image
        meta : dict-like object or None, optional
            Metadata for this object.
        wcs : `~astropy.wcs.WCS`
            WCS-object containing the world coordinate system for the data.
            Calculated automatically assuming the pixel size of the Andor camera
            and the distance to the focal plane provided.
        unit : `astropy.units.Unit`
            The units of the data.
        mask : ~numpy.ndarray` or None, optional
            Mask for the data, given as a boolean Numpy array with a shape
            matching that of the data. The values must be False where the data
            is valid and True when it is not (like Numpy masked arrays).
        filename : str
            The filename of the original data.

        Examples
        --------
        >>> from foxsi_optics_calib.ccd import AndorCCDImage
        >>> import astropy.units as u
        >>> ccd = AndorCCDPsfImage('filename.fits', 2 * u.m)
    """
    def __init__(self, filename, darkfile, distance):
        AndorCCDImage.__init__(self, filename, darkfile, distance)
        maxpix = np.unravel_index(np.argmax(self.data), self.shape)
        self.wcs.wcs.crpix = [maxpix[1], maxpix[0]]
        self.set_xlim(-40, 40)
        self.set_ylim(-40, 40)

    def hpd(self):
        """Provide the half power diameter array."""
        max_pixel_range = 100
        max_pixel = self.wcs.wcs.crpix.astype('int')
        x, y = np.meshgrid(*[np.arange(v) for v in self.data.shape])
        r = np.sqrt((x - max_pixel[0]) ** 2 + (y - max_pixel[1]) ** 2)
        hpd_array = np.zeros_like(np.arange(max_pixel_range).astype('float'))
        for i in np.arange(max_pixel_range):
            hpd_array[i] = np.sum(self.data[r < i])
        hpd_array /= hpd_array.max()
        binning = 2 # Binning used for the ANDOR camera when taking data
        diameter = 2 * binning * np.arange(max_pixel_range) * CCD_PLATE_SCALE
        hpd_value = np.interp(0.5, hpd_array, diameter)
        return hpd_value, hpd_array, diameter

    def plot_cut(self, xlim=[-40, 40], ax=None, title=None, direction='x'):
        max_pixel = self.wcs.crpix
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


class AndorCCDPsfFitImage(AndorCCDPsfImage):

    def __init__(self, filename, darkfile, distance):
        AndorCCDPsfImage.__init__(self, filename, darkfile, distance)

        self._fit()
        self.im_fit = self.fit_func(self.xaxis.value, self.yaxis.value)
        self.fwhm = self.calculate_fwhm()

    def _fit(self):
        amplitude = self.data.max()
        guess_params = calculate_best_guess_params(0 * u.arcmin, 0 * u.arcmin)
        print(guess_params)
        g_init = PSF2DModel(amplitude1=amplitude, x_stddev1=3, y_stddev1=3,
                            amplitude2=amplitude / 10., x_stddev2=5, y_stddev2=5,
                            amplitude3=amplitude / 20., x_stddev3=10, y_stddev3=10,
                            x_mean=0, y_mean=0,
                            theta=0, offset=0.0)
        #print("angle={0}".format(self.polar_angle.to('deg').value))
        fit = LevMarLSQFitter()
        fitted_model = fit(g_init, self.xaxis.to('arcsec').value,
                           self.yaxis.to('arcsec').value, self.data, maxiter=200)
        print(fit.fit_info['message'])
        print("amplitude: {0} {1} {2}".format(fitted_model.amplitude1.value,
                                              fitted_model.amplitude2.value,
                                              fitted_model.amplitude3.value))
        print("center: {0}, {1}".format(fitted_model.x_mean.value,
                                        fitted_model.y_mean.value))
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
        try:
            fwhm_x = 2 * opt.brentq(f_x, f.x_mean.value, f.x_mean.value + 20)
        except ValueError:
            fwhm_x = np.nan
            pass
        try:
            fwhm_y = 2 * opt.brentq(f_y, f.y_mean.value, f.y_mean.value + 20)
        except ValueError:
            fwhm_y = np.nan
            pass
        f.offset += half_maximum
        return np.array([fwhm_x, fwhm_y])


    def plot_fit(self, ax=None, title=None):
        self.plot_image(ax=ax, im=self.im_fit, title=title)
        pix = self.w.wcs_world2pix([[-35, -35]], 1)
        plt.text(pix[0][0], pix[0][1],
                 'fwhm=[{0:.2f}, {1:.2f}] arcsec'.format(self.fwhm[0],
                                                         self.fwhm[1]),
                 color='white')


    def plot_fit_and_image(self, ax=None, title=None,
                           levels=[0.1, 1, 20, 30, 50, 75]):
        ax, imshow, cont = self.plot_image(ax=ax, title=title, levels=levels)
        cont = ax.contour(self.im_fit / self.im_fit.max(),
                          levels=np.array(levels) / 100.,
                          colors='white', alpha=0.5)
        return ax, cont


    def plot_cut_fit(self, direction='x', ax=None, title=None, xlim=[-40, 40]):
        ax = self.plot_cut(ax=ax, title=title, direction=direction, xlim=xlim)
        x = np.linspace(xlim[0], xlim[1], 100)
        if direction.count('x'):
            f = psf_x(self.fit_func.y_mean, self.fit_func)
        if direction.count('y'):
            f = psf_y(self.fit_func.x_mean, self.fit_func)
        y = np.array([f(this_x) for this_x in x])
        # normalize
        y = y / y.max()
        ax.plot(x, y, color='blue')


@u.quantity_input(offaxis_angle=u.deg, polar_angle=u.deg)
def find_ccd_image(offaxis_angle, polar_angle):
    """Given an offaxis angle and a polar angle, find the appropriate
        index in the hdf file. If more than one, returns the first.
    """
    offax_condition = foxsi_optics_calib.ccd_offaxis_angles.value == offaxis_angle.value
    polar_condition = foxsi_optics_calib.ccd_polar_angles.value == polar_angle.value
    index = np.where(offax_condition * polar_condition)
    print(index)
    index = index[0][0]
    if index is None:
        raise ValueError(
            "No index found for {offaxis_angle} and {polar_angle}.".format(
                offaxis_angle=offaxis_angle, polar_angle=polar_angle))
    #print("Index found {index}".format(index=index))
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
    A class for PSF CCD images. It assumes that the image should be centered on
    the brightest pixel.
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
        if shift:
            self.w.wcs.crval += [self.yaw.to('arcsec').value,
                                 self.pitch.to('arcsec').value]

        x, y = np.meshgrid(*[np.arange(v) for v in self.im.shape]) * u.pixel
        self.xaxis, self.yaxis = self.w.wcs_pix2world(x, y, 1) * u.arcsec

    def hpd(self):
        max_pixel_range = 100
        max_pixel = self.w.wcs.crpix.astype('int')
        x, y = np.meshgrid(*[np.arange(v) for v in self.im.shape])
        r = np.sqrt((x - max_pixel[0]) ** 2 + (y - max_pixel[1]) ** 2)
        hpd_array = np.zeros_like(np.arange(max_pixel_range).astype('float'))
        for i in np.arange(max_pixel_range):
            hpd_array[i] = np.sum(self.im[r < i])
        hpd_array /= hpd_array.max()
        print(2 * np.interp(0.5, hpd_array,
                            np.arange(max_pixel_range) * CCD_PLATE_SCALE))
        return hpd_array

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

    def __init__(self, ccd_image, shift=False):
        CCDImage.__init__(self, ccd_image.im, ccd_image.offaxis_angle,
                          ccd_image.polar_angle, shift=shift)
        self._fit()
        self.im_fit = self.fit_func(self.xaxis.value, self.yaxis.value)
        self.fwhm = self.calculate_fwhm()

    def _fit(self):
        amplitude = self.im.max()
        guess_params = calculate_best_guess_params(self.offaxis_angle, self.polar_angle)
        print(guess_params)
        g_init = PSF2DModel(amplitude1=amplitude, x_stddev1=3, y_stddev1=3,
                       amplitude2=amplitude/10., x_stddev2=5, y_stddev2=5,
                       amplitude3=amplitude/20., x_stddev3=10, y_stddev3=10,
                       x_mean=0, y_mean=0, theta=self.polar_angle.to('deg').value, offset=0.0)
        print("angle={0}".format(self.polar_angle.to('deg').value))
        fit = LevMarLSQFitter()
        fitted_model = fit(g_init, self.xaxis.to('arcsec').value,
                           self.yaxis.to('arcsec').value, self.im, maxiter=200)
        print(fit.fit_info['message'])
        print("amplitude: {0} {1} {2}".format(fitted_model.amplitude1.value,
                                              fitted_model.amplitude2.value,
                                              fitted_model.amplitude3.value))
        print("center: {0}, {1}".format(fitted_model.x_mean.value, fitted_model.y_mean.value))
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
        try:
            fwhm_x = 2 * opt.brentq(f_x, f.x_mean.value, f.x_mean.value + 20)
        except ValueError:
            fwhm_x = np.nan
            pass
        try:
            fwhm_y = 2 * opt.brentq(f_y, f.y_mean.value, f.y_mean.value + 20)
        except ValueError:
            fwhm_y = np.nan
            pass
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
        y = np.array([f(this_x) for this_x in x])
        # normalize
        y = y / y.max()
        ax.plot(x, y, color='blue')


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

