from setuptools import setup, find_packages

setup(
    name="seismic-interpretation",
    version="1.0.0",
    author="Ing. Kelvin Cabrera",
    description="ML-based seismic attribute interpretation: horizon tracking and facies classification",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "flask>=3.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
    ],
)
