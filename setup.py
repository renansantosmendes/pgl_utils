"""
Setup configuration for Post Graduation Utils
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pgl-utils",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine Learning, Deep Learning, and GenAI utilities for Post-Graduation Lectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/renansantosmendes/pgl_utils",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
        ],
        "dl": [
            "torch>=1.10.0",
            "tensorflow>=2.8.0",
            "keras>=2.8.0",
        ],
        "genai": [
            "openai>=0.27.0",
            "langchain>=0.0.100",
            "huggingface-hub>=0.12.0",
        ],
        "all": [
            "torch>=1.10.0",
            "tensorflow>=2.8.0",
            "keras>=2.8.0",
            "openai>=0.27.0",
            "langchain>=0.0.100",
            "huggingface-hub>=0.12.0",
            "scikit-learn>=1.0.0",
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
