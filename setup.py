from setuptools import setup, find_packages

setup(
    name="operator-test-framework",
    version="0.1.0",
    description="Deep Learning Operator Testing Framework",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "pytest>=6.0.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)