# -*- coding: utf-8 -*-
"""
Code to read raster data from CdTe or Si pinhole detector
"""
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.modeling.fitting import LevMarLSQFitter

import foxsi_optics_calib
from foxsi_optics_calib.psf import psf1d, calculate_fwhm1d

ENERGY_CALIB_M = 0.1073
ENERGY_CALIB_B = -0.4849
NUMBER_OF_CHANNELS = 256


class RasterSpectrum(object):
    def __init__(self, filename, pinhole_size=0.05*u.mm):
        xy = []
        spectra = []
        self.energy_axis = (ENERGY_CALIB_M * (np.arange(0, NUMBER_OF_CHANNELS)+1) - ENERGY_CALIB_B) * u.keV
        self.filename = filename
        self.position, self.data = self._parse_file(filename)
        self.pinhole_size = pinhole_size
        self.plate_scale = foxsi_optics_calib.plate_scale(1 * u.mm)
        self.error = np.sqrt(self.data.value) * u.count
        # recenter based on max position
        total_counts = np.sum(self.data, axis=1)
        max_position = self.position[np.argmax(total_counts), 1]
        self.position[:, 1] = self.position[:, 1] - max_position
        self._fit()

    def _parse_file(self, filename):
        with open(filename) as f:
            spectra = []
            xy = []
            for i, line in enumerate(f):
                if i == 2:
                    self.exposure_time = float(f.readline()[6:13]) * u.s
                if i > 4 and (i % 2) == 0:
                    spectra.append(
                        [float(this_val) for this_val in line.split('\t')])
                if i > 4 and (i % 2) == 1:
                    xy.append([float(line[4:7]), float(line[16:19])])
        return np.array(xy) * u.mm, np.array(spectra) * u.count

    def plot_spectrum(self, index=None, ax=None):
        if not ax:
            ax = plt.subplot()
        if index is None:
            data = np.sum(self.data, axis=0)
        else:
            data = self.data[index, :]
        ax.plot(self.energy_axis, data)
        ax.set_ylabel('Counts')
        ax.set_xlabel('Energy [{0}]'.format(self.energy_axis.unit))
        return ax

    def _fit(self):
        y = np.sum(self.data, axis=1)
        y = y / (y.max() * self.exposure_time)
        x = self.position[:, 1]
        x_arcsec = x * self.plate_scale
        amplitude = y.max()
        g_init = psf1d(amplitude1=amplitude * 1.0, stddev1=1,
                       amplitude2=amplitude * 0.5, stddev2=5,
                       amplitude3=amplitude * 0.5, stddev3=10,
                       mean=0, offset=0.0)
        fit = LevMarLSQFitter()
        fitted_model = fit(g_init, x, y)
        self.fit_func = fitted_model
        self.fwhm = self.calculate_fwhm1d()

    def plot_scan(self, energy_range=None, ax=None):
        if not ax:
            ax = plt.subplot()
        total_counts = np.sum(self.data, axis=1)
        err_counts = np.sqrt(total_counts.value) * u.count
        norm_rate = total_counts / (self.exposure_time * total_counts.max())
        norm_rate_error = err_counts / (self.exposure_time * total_counts.max())
        x = self.position[:, 1]
        x_arcsec = x * self.plate_scale
        xerr = foxsi_optics_calib.plate_scale(self.pinhole_size)
        ax.errorbar(x_arcsec.value, norm_rate.value,
                    yerr=norm_rate_error.value,
                    xerr=xerr.value)
        ax.set_ylabel('Normalized Rate')
        ax.set_xlabel('Position [arcsec]')
        ax.set_yscale('log')

    def plot_fit(self, ax=None):
        if not ax:
            ax = plt.subplot()
