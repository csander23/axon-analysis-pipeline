from setuptools import setup, find_packages

setup(
    name="axon_analysis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'scikit-image',
        'pyyaml',
        'aicsimageio',
        'nd2reader'
    ],
    entry_points={
        'console_scripts': [
            'run-axon-analysis=axon_analysis.cli:main',
        ],
    },
    author="Charles Sander",
    description="Axon Analysis Pipeline for confocal microscopy images",
    python_requires='>=3.9',
)
