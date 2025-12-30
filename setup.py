# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="candidate_fit_predictor",
    version="2.0.0",
    author="AI Assistant",
    author_email="example@example.com",
    description="Machine learning system for candidate-job compatibility prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Subhashk75/candidate_fit_predictor.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: HR Professionals",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "nltk>=3.8.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "candidate-fit-predictor=candidate_fit_predictor:main",
        ],
    },
)