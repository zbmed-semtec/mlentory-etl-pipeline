from setuptools import setup, find_packages

setup(
    name="mlentory_extract",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "transformers",
        "datasets",
        "torch",
        "huggingface-hub",
        "tqdm",
        "openml", 
    ],
    python_requires=">=3.8.10",
    author="Nelson Q, Suhasini V",
    author_email="your.email@example.com",
    description="Collection of extractors for model information from different ML platforms",
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
