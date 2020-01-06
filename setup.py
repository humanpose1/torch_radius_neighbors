from setuptools import setup, Extension, find_packages
import glob
import torch
from torch.utils import cpp_extension

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

extra_compile_args = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    extra_compile_args += ['-DVERSION_GE_1_3']

setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(name="torch_nearest_neighbors",
      ext_modules=[cpp_extension.CppExtension(name="torch_radius_search",
                                              sources=["torch_nearest_neighbors.cpp",
                                                       "utils/neighbors.cpp"],
                                              include_dirs=['utils/'])],

      cmdclass={'build_ext':cpp_extension.BuildExtension},
      packages=find_packages(),
      setup_requires=setup_requires,
      tests_require=tests_require)
