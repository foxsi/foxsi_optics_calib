# -*- coding: utf-8 -*-
"""
Define the FOXSI Point Spread Function
"""
from astropy.modeling.models import custom_model, Gaussian2D, Gaussian1D
from astropy.modeling import Fittable2DModel, Parameter
import scipy.optimize as opt
import numpy as np
import astropy.units as u
import foxsi_optics_calib


class PSF2DModel(Fittable2DModel):
    amplitude1 = Parameter(min=0)
    amplitude2 = Parameter(min=0)
    amplitude3 = Parameter(min=0)
    x_stddev1 = Parameter(min=0)
    x_stddev2 = Parameter(min=0)
    x_stddev3 = Parameter(min=0)
    y_stddev1 = Parameter(min=0)
    y_stddev2 = Parameter(min=0)
    y_stddev3 = Parameter(min=0)
    x_mean = Parameter()
    y_mean = Parameter()
    theta = Parameter()
    offset = Parameter(min=0)

    @staticmethod
    def evaluate(x, y, amplitude1, x_stddev1, y_stddev1, amplitude2,
                 x_stddev2, y_stddev2, amplitude3, x_stddev3, y_stddev3,
                 x_mean, y_mean, theta, offset):
        g1 = Gaussian2D(amplitude=amplitude1, x_mean=x_mean, y_mean=y_mean,
                        x_stddev=x_stddev1, y_stddev=y_stddev1, theta=theta)
        g1.amplitude.min = 0
        g2 = Gaussian2D(amplitude=amplitude2, x_mean=x_mean, y_mean=y_mean,
                        x_stddev=x_stddev2, y_stddev=y_stddev2, theta=theta)
        g2.amplitude.min = 0
        g3 = Gaussian2D(amplitude=amplitude3, x_mean=x_mean, y_mean=y_mean,
                        x_stddev=x_stddev3, y_stddev=y_stddev3, theta=theta)
        g3.amplitude.min = 0
        y1 = g1.evaluate(x, y, amplitude=amplitude1, x_mean=x_mean,
                         y_mean=y_mean,
                         x_stddev=x_stddev1, y_stddev=y_stddev1, theta=theta)
        y2 = g2.evaluate(x, y, amplitude=amplitude2, x_mean=x_mean,
                         y_mean=y_mean,
                         x_stddev=x_stddev2, y_stddev=y_stddev2, theta=theta)
        y3 = g3.evaluate(x, y, amplitude=amplitude3, x_mean=x_mean,
                         y_mean=y_mean,
                         x_stddev=x_stddev3, y_stddev=y_stddev3, theta=theta)
        return y1 + y2 + y3 + offset

@custom_model
def psf2d(x, y,
        amplitude1=1.0, x_stddev1=1.0, y_stddev1=1,
        amplitude2=0.5, x_stddev2=5.0, y_stddev2=5,
        amplitude3=0.1, x_stddev3=10., y_stddev3=10,
        x_mean=0., y_mean=0, theta=0.0, offset=0):
    """The model of the FOXSI PSF"""
    # the following is to force amplitudes to be positive only

    g1 = Gaussian2D(amplitude=amplitude1, x_mean=x_mean, y_mean=y_mean,
                    x_stddev=x_stddev1, y_stddev=y_stddev1, theta=theta)
    g1.amplitude.min = 0
    g2 = Gaussian2D(amplitude=amplitude2, x_mean=x_mean, y_mean=y_mean,
                    x_stddev=x_stddev2, y_stddev=y_stddev2, theta=theta)
    g2.amplitude.min = 0
    g3 = Gaussian2D(amplitude=amplitude3, x_mean=x_mean, y_mean=y_mean,
                    x_stddev=x_stddev3, y_stddev=y_stddev3, theta=theta)
    g3.amplitude.min = 0
    y1 = g1.evaluate(x, y, amplitude=amplitude1, x_mean=x_mean, y_mean=y_mean,
                     x_stddev=x_stddev1, y_stddev=y_stddev1, theta=theta)
    y2 = g2.evaluate(x, y, amplitude=amplitude2, x_mean=x_mean, y_mean=y_mean,
                     x_stddev=x_stddev2, y_stddev=y_stddev2, theta=theta)
    y3 = g3.evaluate(x, y, amplitude=amplitude3, x_mean=x_mean, y_mean=y_mean,
                     x_stddev=x_stddev3, y_stddev=y_stddev3, theta=theta)
    result = y1 + y2 + y3 + offset
    if ((amplitude1 < 0) or (amplitude2 < 0) or (amplitude3 < 0) or (offset < 0)):
        result *= 1e99
    return result


@custom_model
def psf1d(x, y,
          amplitude1=1.0, stddev1=1.0,
          amplitude2=0.5, stddev2=5.0,
          amplitude3=0.1, stddev3=10.,
          mean=0., offset=0.):
    if amplitude1 < 0:
        amplitude1 = 1e12
    if amplitude2 < 0:
        amplitude2 = 1e12
    if amplitude3 < 0:
        amplitude3 = 1e12
    g1 = Gaussian1D(amplitude=amplitude1, mean=mean, stddev=stddev1)
    g2 = Gaussian1D(amplitude=amplitude2, mean=mean, stddev=stddev2)
    g3 = Gaussian1D(amplitude=amplitude3, mean=mean, stddev=stddev3)
    y1 = g1.evaluate(x, amplitude=amplitude1, mean=mean, stddev=stddev1)
    y2 = g2.evaluate(x, amplitude=amplitude2, mean=mean, stddev=stddev2)
    y3 = g3.evaluate(x, amplitude=amplitude3, mean=mean, stddev=stddev3)
    return y1 + y2 + y3 + offset


def psf_x(y_value, psf_function):
    """The PSF as a function of x for a given y."""
    return lambda x: psf_function(x, y_value)


def psf_y(x_value, psf_function):
    """The PSF as a function of x for a given y."""
    return lambda y: psf_function(x_value, y)


def calculate_fwhm2d(psf2d_func):
    f = psf2d_func
    maximum = f.amplitude1 + f.amplitude2 + f.amplitude3
    half_maximum = maximum / 2.0
    f.offset -= half_maximum
    f_x = psf_x(f.y_mean, f)
    f_y = psf_y(f.x_mean, f)
    fwhm_x = 2 * opt.brentq(f_x, f.x_mean.value, f.x_mean.value + 5)
    fwhm_y = 2 * opt.brentq(f_y, f.y_mean.value, f.y_mean.value + 5)
    f.offset += half_maximum
    return np.array([fwhm_x, fwhm_y])


def calculate_fwhm1d(psf1d_func):
    f = psf1d_func
    maximum = f.amplitude1 + f.amplitude2 + f.amplitude3
    half_maximum = maximum / 2.0
    f.offset -= half_maximum
    fwhm = 2 * opt.brentq(f, f.x_mean.value, f.x_mean.value + 5)
    f.offset += half_maximum
    return fwhm


def calculate_best_guess_params(pitch, yaw):
    """pitch and yaw in arcmin."""
    path = foxsi_optics_calib.__path__[0]
    params_file = path+'/params.txt'
    params = np.loadtxt(params_file)
    offaxis_angle = np.sqrt(pitch ** 2 + yaw ** 2)
    polar_angle = np.arctan2(pitch, yaw) + np.pi / 2. * u.rad
    poly_params = []
    theta = polar_angle.to('deg').value
    for g in params:
        f = np.poly1d(g)
        poly_params.append(f(offaxis_angle.to('arcmin').value))
    result = {'amplitude1': poly_params[0], 'amplitude2': poly_params[1],
              'amplitude3': poly_params[2],
              'x_stddev1': poly_params[3], 'y_stddev1': poly_params[4],
              'x_stddev2': poly_params[5], 'y_stddev2': poly_params[6],
              'x_stddev3': poly_params[7], 'y_stddev3': poly_params[8],
              'theta': theta
              }
    #print(poly_params[0])
    #result = {'amplitude1': 10000, 'amplitude2': 5000,
    #          'amplitude3': 100,
    #          'x_stddev1': 4, 'y_stddev1': 4,
    #          'x_stddev2': 10, 'y_stddev2': 10,
    #          'x_stddev3': 40, 'y_stddev3': 40
    #          }
    #print("Best guess parameters are.")
    #print("amplitude = {0} {1} {2}".format(result['amplitude1'],
    #                                       result['amplitude2'],
    #                                       result['amplitude3']))
    #print("x stddev = {0} {1} {2}".format(result['x_stddev1'],
    #                                      result['x_stddev2'],
    #                                      result['x_stddev3']))
    #print("y stddev = {0} {1} {2}".format(result['y_stddev1'],
    #                                      result['y_stddev2'],
    #                                      result['y_stddev3']))

    return result
