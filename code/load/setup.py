from setuptools import setup, find_packages

setup(
    name="mlentory_loader",
    version="0.1.0",
    description="Loading component for MLENTORY",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Nelson Q",
    author_email="your.email@example.com",
    packages=find_packages(),
    package_dir={"": "src"},
    install_requires=[
        "watchdog",
        "matplotlib",
        "numpy",
        "pandas",
        "rdflib",
        "docker",
        "sparqlwrapper",
        "elasticsearch==7.13.4",
        "elasticsearch-dsl<8.0.0",
        "psycopg2",
        "tqdm"
    ],
    python_requires=">=3.8.10",
) 
