"""Setup script for PTZ-Calib Python package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ptzcalib",
    version="0.1.0",
    author="PTZ-Calib Contributors",
    description="Python port of PTZ-Calib for robust PTZ camera calibration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "pyceres>=2.6",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
