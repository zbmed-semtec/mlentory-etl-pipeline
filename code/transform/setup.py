from setuptools import setup, find_packages

setup(
    name="transform",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "watchdog",
        "datasets",
        "huggingface-hub",
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "demjson3",
        "rdflib",
        "black",
        "tqdm",
    ],
    python_requires=">=3.8.10",
    author="Nelson Q",
    author_email="your.email@example.com",
    description="A package for transforming ML model metadata",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
