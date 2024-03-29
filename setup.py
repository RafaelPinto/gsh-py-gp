from setuptools import find_packages, setup

description = """
    2023 GSH Spring Symposium Python in Geophysics,
    Re-ranking seismic uncertainty analysis surfaces with Python.
"""

setup(
    name="src",
    packages=find_packages(),
    version="0.0.1",
    description=description,
    author="Rafael Pinto",
    license="BSD 3-Clause",
)
