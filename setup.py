from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds

setup(name='PyMSTk',
      version='0.5',
      description='Python Model Selection Toolkit',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='',
      package_dir={'pymstk':'src'},
      packages=['pymstk']
     )
