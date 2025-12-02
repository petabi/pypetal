from setuptools import setup
from setuptools_rust import RustExtension

setup(
    name="pypetal",
    version="0.1.0",
    rust_extensions=[RustExtension("pypetal.pypetal")],
    packages=["pypetal"],
    zip_safe=False,
)
