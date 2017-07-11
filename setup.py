from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()

version = '0.1'

install_requires = [
    # List your project dependencies here.
    'matplotlib',
    'numpy',
    'astropy',
    'h5py'
]

setup(name='foxsi_optics_calib',
    version=version,
    description="Code to analyze FOXSI Rocket Optics Calibration Files",
    long_description=README + '\n\n',
    classifiers=[
      # Get strings from
    ],
    keywords='foxsi',
    author='Steven Christe',
    author_email='steven.christe@nasa.gov',
    url='',
    license='MIT',
    #packages=find_packages('ledlifetest'),
    packages=['foxsi_optics_calib'],
    #package_dir = {'': 'ledlifetest'}, include_package_data=True,
    zip_safe=False,
    install_requires=install_requires
)
