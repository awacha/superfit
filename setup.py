#!/usb/bin/env python
import os
import sys

from numpy.lib import get_include
from setuptools import setup, find_packages, Extension

try:
    from PyQt5.uic import compileUi
except ImportError:
    def compileUi(*args):
        pass

rcc_output = os.path.join('src', 'superfit', 'resource', 'icons_rc.py')
rcc_input = os.path.join('src','superfit', 'resource', 'icons', 'icons.qrc')
os.system('pyrcc5 {} -o {}'.format(rcc_input, rcc_output))

def compile_uis(packageroot):
    if compileUi is None:
        return
    for dirpath, dirnames, filenames in os.walk(packageroot):
        for fn in [fn_ for fn_ in filenames if fn_.endswith('.ui')]:
            fname = os.path.join(dirpath, fn)
            pyfilename = os.path.splitext(fname)[0] + '_ui.py'
            with open(pyfilename, 'wt', encoding='utf-8') as pyfile:
                compileUi(fname, pyfile, from_imports=True, import_from='superfit.resource')
            print('Compiled UI file: {} -> {}.'.format(fname, pyfilename))

compile_uis(os.path.join('src','superfit'))



def getresourcefiles():
    print('Generating resource list', flush=True)
    reslist = []
    for directory, subdirs, files in os.walk(os.path.join('src','superfit','resource')):
        reslist.extend([os.path.join(directory, f).split(os.path.sep, 1)[1] for f in files])
    print('Generated resource list:\n  ' + '\n  '.join(x for x in reslist) + '\n', flush=True)
    return reslist


extensions=[]

#update_languagespec()
setup(name='superfit', author='Andras Wacha',
      author_email='awacha@gmail.com', url='http://github.com/awacha/superfit',
      description='Least Squares fitting GUI',
      package_dir = {'':'src'},
      packages=find_packages('src'),
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      install_requires=['numpy>=1.11.1', 'scipy>=0.18.0', 'matplotlib>=1.5.2', 'openpyxl', 'fffs', 'docutils'],
      entry_points={'gui_scripts': ['superfit = superfit.__main__:run',
                                    ],
                    },
      keywords="saxs sans sas small-angle scattering x-ray instrument control",
      license="",
      package_data={'': getresourcefiles()},
      #      include_package_data=True,
      zip_safe=False,
      )
