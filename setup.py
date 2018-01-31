#!/usb/bin/env python
import os
import sys

from Cython.Build import cythonize
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
                compileUi(fname, pyfile, from_imports=True, import_from='cct.resource')
            print('Compiled UI file: {} -> {}.'.format(fname, pyfilename))

compile_uis(os.path.join('src','superfit'))



def getresourcefiles():
    print('Generating resource list', flush=True)
    reslist = []
    for directory, subdirs, files in os.walk(os.path.join('src','superfit','resource')):
        reslist.extend([os.path.join(directory, f).split(os.path.sep, 1)[1] for f in files])
    print('Generated resource list:\n  ' + '\n  '.join(x for x in reslist) + '\n', flush=True)
    return reslist


if sys.platform.lower().startswith('win') and sys.maxsize>2**32:
    krb5_libs=['krb5_64']
else:
    krb5_libs=['krb5']

#extensions = [Extension("cct.qtgui.tools.optimizegeometry.estimateworksize",
#                        [os.path.join("cct","qtgui","tools","optimizegeometry","estimateworksize.pyx")],
#                        include_dirs=[get_include()]),
#              Extension("cct.core.services.accounting.krb5_check_pass",
#                        [os.path.join("cct","core","services","accounting","krb5_check_pass.pyx")],
#                        include_dirs=[get_include()], libraries=krb5_libs)
#              ]

extensions=[]

#update_languagespec()
setup(name='superfit', author='Andras Wacha',
      author_email='awacha@gmail.com', url='http://github.com/awacha/superfit',
      description='Least Squares fitting GUI',
      package_dir = {'':'src'},
      packages=find_packages('src'),
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      #      cmdclass = {'build_ext': build_ext},
      ext_modules=cythonize(extensions),
      install_requires=['numpy>=1.11.1', 'scipy>=0.18.0', 'matplotlib>=1.5.2', 'openpyxl'],
      entry_points={'gui_scripts': ['superfit = superfit.__main__:run',
                                    ],
                    },
      keywords="saxs sans sas small-angle scattering x-ray instrument control",
      license="",
      package_data={'': getresourcefiles()},
      #      include_package_data=True,
      zip_safe=False,
      )
