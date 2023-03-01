# -*- coding: utf-8 -*-
import setuptools
from distutils.util import convert_path

with open('README.md', 'r') as fh:
    long_description = fh.read()

main_ns = {}
ver_path = convert_path('scorekit/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setuptools.setup(name = 'scorekit',
      version = main_ns['__version__'],
      author = 'Ruslan Sakovich',
      description = 'Package for solving binary classification problems by logistic regression',
      license = 'BSD',
      install_requires = ['pandas', 'matplotlib', 'numpy', 'seaborn', 'scikit-learn', 'scipy', 'statsmodels', 
                          'python-docx', 'openpyxl', 'sklearn', 'optbinning', 'cloudpickle', 'tqdm', 'xlsxwriter'],
      long_description = long_description,
      packages = setuptools.find_packages())
