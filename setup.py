"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from os import path
from itertools import chain
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

extras_require={
    'test': ['coverage', 'pytest'],
    'dev': ['asv'],
}

extras_require['all'] = list(set(chain(*extras_require.values())))

setup(
    name='DLA',
    version='0.1.0',
    description='A simulation of diffusion-limited aggregation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/StanczakDominik/DLA',
    author='Dominik Stańczak',
    author_email='stanczakdominik@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics'
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='diffusion-limited-aggregation simulation',
    packages=find_packages(exclude=['docs', 'tests', 'benchmarks']),
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'tqdm', 'numba', 'matplotlib', 'pandas'],
    extras_require=extras_require,

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },

    project_urls={
        'Bug Reports': 'https://github.com/StanczakDominik/DLA/issues',
        'Say Thanks!': 'https://saythanks.io/to/StanczakDominik',
        'Source': 'https://github.com/StanczakDominik/DLA/',
    },
)
