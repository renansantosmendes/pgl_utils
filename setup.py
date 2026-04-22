"""
Setup configuration for Post Graduation Utils
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pgl-utils",
    version="0.1.4",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine Learning, Deep Learning, and GenAI utilities for Post-Graduation Lectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/renansantosmendes/pgl_utils",
    packages=find_packages(exclude=["black"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "networkx>=2.8.0",
        "mlflow",
        "dagshub",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.4.0",
            "mypy>=1.0.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
        ],
        "deep_learning": [
            "torch>=2.7.0",
            "torchvision",
            "torchaudio",
            "tensorflow>=2.16.0",
            "keras>=3.0.0",
        ],
        "all": [
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "torch>=2.7.0",
            "torchvision",
            "torchaudio",
            "tensorflow>=2.16.0",
            "keras>=3.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
