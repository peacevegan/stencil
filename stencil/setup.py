from setuptools import setup, find_packages

VERSION = '0.1' 
DESCRIPTION = 'Stencil Python package for Data Science Utils'
LONG_DESCRIPTION = 'Provide some useful functions for data processing'

# Setting up
setup(
        name="stencil", 
        version=VERSION,
        author="Trung Hoang",
        author_email="<trunght1402@email.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "numpy >= 1.19.1",
            "pandas >= 1.1.2",
            "scikit-learn >= 0.23.2"
        ], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)