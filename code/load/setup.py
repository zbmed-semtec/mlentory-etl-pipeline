from setuptools import setup, find_packages

setup(
    name="mlentory_loader",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "watchdog",
        "matplotlib",
        "numpy",
        "pandas",
        "rdflib",
        "docker",
        "sparqlwrapper",
        "urllib3<2.0.0",
        "psycopg2-binary",
        "elasticsearch==7.13.4",
        "elasticsearch-dsl<8.0.0",
        "tqdm",
    ],
    python_requires=">=3.8.10",
    author="Nelson Q",
    author_email="your.email@example.com",
    description="Loading component for MLENTORY",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
