"""
Setup configuration for Post Graduation Utils
"""

from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pgl-utils",
    version="0.2.0",
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
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.4.0",
            "mypy>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
