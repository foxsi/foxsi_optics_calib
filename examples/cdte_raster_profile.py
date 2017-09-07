from foxsi_optics_calib.cdte.cdte import RasterSpectrum
import matplotlib.pyplot as plt

test_file = '/Users/schriste/Data/FOXSI Calibration/FOXSI backup 2011Feb28/Fwhm/ModuleX0/onaxis/modulex0_coarseY_0.05pin_startminus1.0'

raster = RasterSpectrum(test_file)
print(raster.position)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
raster.plot_spectrum(ax=ax)
ax = fig.add_subplot(1, 2, 2)
raster.plot_scan(ax=ax)
plt.show()