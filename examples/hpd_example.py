import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from foxsi_optics_calib.ccd.ccd import find_ccd_image, CCD_PLATE_SCALE

fig = plt.figure(figsize=(13, 4))

foxsi_ccd_image = find_ccd_image(0 * u.arcsec, 0 * u.arcsec)
hpd = foxsi_ccd_image.hpd()

hpd_index = hpd[hpd < 0.5].argmax()
r = 2 * np.arange(len(hpd)) * CCD_PLATE_SCALE
plt.plot(r, hpd)
plt.xlabel('R [arcsec]')
plt.ylabel('Flux Fraction')
plt.axhline(0.5)

#plt.savefig('hpd_example.pdf')
plt.show()
plt.close()