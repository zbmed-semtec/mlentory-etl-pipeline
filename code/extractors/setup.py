from setuptools import setup, find_packages

setup(
    name="mlentory_extract",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytest>=8.4.1",
        "pandas>=2.3.2,<3.0.0",
        "transformers>=4.56.0",
        "datasets>=2.19.2",
        "huggingface-hub",
        "tqdm>=4.67.1",
        "openml>=0.15.1",
        "arxiv>=2.2.0",
        "scikit-learn>=1.7.1",
        "selenium>=4.35.0",
        "hypha_rpc==0.20.55"
    ],
    extras_require={
        "gpu": [
            "torch>=2.7.1",
            "vllm>=0.10.1.1",
        ],
    },
    python_requires=">=3.8.10",
    author="Nelson Q, Suhasini V ,Dhwani S",
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
