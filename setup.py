from setuptools import setup, find_packages

setup(
    name="plottinglib",
    version="0.1.0",
    description="A library for plotting data",
    author="Timo Hillmann",
    author_email="timo.hillmann@rwth-aachen.de",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
)
