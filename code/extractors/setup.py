from setuptools import setup, find_packages

setup(
    name="mlentory_extract",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytest>=7.0.0",
        "pandas>=2.0.0,<3.0.0",
        "transformers>=4.48.0",
        "datasets~=2.19.0",
        # "torch==2.1.2",
        "huggingface-hub>=0.22.0",
        "tqdm>=4.62.1",
        "openml>=0.14.0",
        "arxiv>=2.1.0",
        "scikit-learn>=1.6.0",
        "vllm>=0.1.0",
        "selenium>=4.11.0"
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
