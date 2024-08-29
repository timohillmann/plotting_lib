from setuptools import setup, find_packages


setup(
    name="plotting_lib",
    version="0.1.0",
    description="A library for plotting data",
    author="Timo Hillmann",
    author_email="timo.hillmann@rwth-aachen.de",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"plotting_lib": ["journal_styles/*"]},
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
    ],
)
