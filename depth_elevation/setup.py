from setuptools import setup, find_packages

setup(
    name="depth-elevation",
    version="0.1.0",
    packages=find_packages(include=["depth_elevation*", "examples*"]),
)
