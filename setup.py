import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

VERSION = '0.2' 
DESCRIPTION = 'Stencil Python package for Data Science Utils'

# Setting up
setup(
        name="stencil-data-science", 
        version=VERSION,
        author="Trung Hoang",
        author_email="<trunght1402@email.com>",
        description=DESCRIPTION,
        long_description=README,
        long_description_content_type="text/markdown",
        packages=["stencil"],
        license="MIT",
        install_requires=[
            "numpy >= 1.19.1",
            "pandas >= 1.1.2",
            "scikit-learn >= 0.23.2"
        ], 
        
        keywords=['python'],
        classifiers= [
            "License :: OSI Approved :: MIT License",
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ],
)