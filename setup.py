from setuptools import setup, Extension
import glob
from torch.utils import cpp_extension


setup(name="torch_nearest_neighbors",
      ext_modules=[cpp_extension.CppExtension(name="neighbors",
                                              sources=["torch_nearest_neighbors.cpp",
                                                       "utils/neighbors.cpp",
                                                       "utils/cloud.cpp"],
                                              include_dirs=glob.glob('utils/*.h*'))],
      cmdclass={'build_ext':cpp_extension.BuildExtension})
