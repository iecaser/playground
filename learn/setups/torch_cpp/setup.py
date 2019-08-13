from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(name='test_cpp',
      ext_modules=[CppExtension(name='test_cpp',
                                sources=['src/test.cpp']),
                   ],
      cmdclass={'build_ext': BuildExtension}
      )
