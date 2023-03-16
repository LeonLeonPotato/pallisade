import sys
import os
import numpy
sys.argv.append("build_ext")
sys.argv.append("--inplace")

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(
        module_list=["mcts_rewrite.pyx"],
        annotate=True
    ),
    extra_compile_args=['-arch', 'arm64e'],
    include_dirs=[numpy.get_include()]
)

import numpy as np
import mcts_rewrite
import matplotlib.pyplot as plt

x = np.array(mcts_rewrite.balls)
y = np.array(mcts_rewrite.ballsy)
x = np.trim_zeros(x)
y = np.trim_zeros(y)
plt.plot(x, y, color='red')
plt.show()