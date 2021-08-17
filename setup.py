# setup.py

from setuptools import find_packages, setup

setup(
    name="curriculum_learning",
    version="0.0.0",
    packages=find_packages(exclude=["docs", "scripts", "tests"]),
    install_requires=[
        "gym",
        "gym_fishing",
        "gym_conservation",
        "numpy",
        "pandas",
        "matplotlib",
        "stable_baselines3",
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
        ],
    },
)
