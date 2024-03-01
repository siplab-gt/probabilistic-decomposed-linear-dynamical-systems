from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.14'
DESCRIPTION = 'Probabilistic Decomposed Linear Dynamical Systems'
LONG_DESCRIPTION = 'Probabilistic Inference for Decomposed Linear Dynamical Systems (dLDS) model'

# Setting up
setup(
    name="pdLDS",
    version=VERSION,
    author="Yenho Chen",
    author_email="<yenho@gatech.edu>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['tqdm', 'numpy', 'numba', "torch", "scipy"],
    keywords=['python', 'dynamical systems',],
    classifiers=[
        # "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)