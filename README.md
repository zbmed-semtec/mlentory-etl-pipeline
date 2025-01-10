# MLentory Extraction/Transformation/Loader (ETL) Pipeline

MLentory is centered around information on  ML models, how to harmonize that data, and how to make it available and searchable on an FDO (FAIR Digital Object) registry.

## Purpose
To build a system that extracts ML (Machine Learning) model information from different platforms, normalizes that data in a common format, stores it, and shares it in a FDO registry to facilitate IR (Information Retrieval) and comparison/recommendation systems.

This [TDD](https://docs.google.com/document/d/1aczsHqJ5xxc9Gdd9wC_sfutz1yVUgNJ7WttuSl3SsXU/edit?usp=sharing) (Technical Design Document) will help new contributors understand and old ones remember what decisions were made on the system's design, the motivation behind them, and their impact. The document focuses on the design of the ETL pipeline to collect, transform, and store extracted information.


## Run The Project
There are different things you can execute in this project.
- The first one is the whole ETL pipeline, which is the main component of the project. See instructions here: [ETL Pipeline](deployment/README.md)
- The second one is the test component, which is the component that tests the ETL pipeline. See instructions here: [Test Component](tests/README.md)

## Background
This project is part of the NFDI4DataScience initiative, a German Consortium whose vision is to support all steps of the complex and interdisciplinary research data lifecycle, including collecting/creating, processing, analyzing, publishing, archiving, and reusing resources in Data Science and Artificial Intelligence.

<img src="docs/Readme_images/NFDI4DataScience_structure.png"/>
<p style=" text-align: center; font-size: 0.8em; color: #cccccc">Figure 1. General diagram of the NFDI4DataScience main project.</p>

A big effort in this project will be using semantic technologies aligned to the FDO concept, so information from various ML model platforms is stored using a common data structure, easy for semantic purposes but also optimized for IR.

<img src="docs/Readme_images/Metadata for ML models-ZB MED draft action-centric.jpg"/>
<p style="text-align: center; font-size: 0.8em; color: #cccccc">Figure 2. The BETA  version of the Metadata Schema proposed by ZB MED</p>

In the end, the system and the data it collects will help researchers and other interested parties find and pick ML models from different platforms to solve their particular problems.

## Project architecture

The project architecture is the following:

<img src="docs/Readme_images/MLentory Backend TDD Diagrams-Main_component_interaction_Diagram_v3.png"/>
<p style="text-align: center; font-size: 0.8em; color: #cccccc">Figure 3. Diagram of the whole ETL pipeline.</p>

<img src="docs/Readme_images/MLentory Backend TDD Diagrams-General MLentory diagram_v3.png"/>
<p style="text-align: center; font-size: 0.8em; color: #cccccc">Figure 4. Another view of the ETL pipeline.</p>

The ETL pipeline is composed of 4 main components:

### The Platform Extractors

The platform extractors are the components that extract information from different platforms. They are implemented as isolated containers that communicate with the rest of the ETL pipeline through json files that are going to be used in a queue to process the information being gather from each platform.

### The Transformation Pipeline

The transformation pipeline will be in charge of transforming the information extracted from the platform extractors into a common format given by the data schema described in Figure 2.

### The Loading Pipeline

Will be a container in charge of uploading, updating and resolving conflicts in the information provided by the transformation pipeline, this data will be stored in the RDF database.

### The Orchestrator

Will be a container in charge of launching the pipelines in the right order, and of monitoring the status of the pipelines.

## Project Structure

The project structure is the following:

- A [code](/code/) folder where the python packages of the project are located.
- A [deployment](/deployment/) folder where the execution files for the ETL pipeline are located, also the docker-compose.yml to bring up the whole system.
- A [playground](/playground/) folder where we test and get familiar with the different technologies used in the project.
- A [data](/data/) Where you can find data folders that are used in the different sections of the project.
- A [docs](/docs/) folder where you can find resources related to the project like diagrams, functions, and documentation.
- A [tests](/tests/) folder where you can find the tests configuration of the project.

```
.
├── CITATION.cff
├── LICENSE
├── README.md
├── code
│   ├── README.md
│   ├── extractors
│   ├── load
│   └── transform
├── deployment
│   ├── README.md
│   ├── hf_etl
│   ├── scheduler
│   ├── docker-compose.yml
│   └── requirements.txt
├── data
│   ├── configuration
│   ├── datasets
│   ├── elasticsearch_data
│   ├── postgres_data
│   └── virtuoso_data
├── docs
│   ├── Analysis_graphs
│   ├── HF_legal_files
│   └── Readme_images
├── playground
│   ├── DB
│   ├── Dev_tests
│   ├── Dockerfile.gpu
│   ├── Dockerfile.no_gpu
│   ├── HF_API
│   ├── README.md
│   ├── docker-compose.yml
│   └── requirements.txt
└──tests/
    ├── unit/                     
    │   ├── hf/
    │   └── oml/
    │
    ├── fixtures/
    │   ├── data/
    │   └── mocks/
    │
    ├── integration/
    │   ├── api/
    │   ├── database/
    │   │   ├── postgres/
    │   │   ├── elastic/
    │   │   └── virtuoso/
    │   └── pipeline
    │
    ├── config/
    │   ├── docker/
    |   ├── codecove
    |   ├── hf
    |   ├── oml
    │   └── pytest.ini
    │
    ├── scripts/
    │   ├── validate_tests.sh
    │   ├── local_validate_tests.sh
    │   └── wait-for-it.sh
    │
    └── utils/
        ├── conftest.py
        └── check_licenses.py
```



## Acknowledgements

This project is part of NFDI4DataScience consortium services. [NFDI4DataScience](https://www.nfdi4datascience.de/) is funded by the [German Research Foundation (DFG)](https://www.dfg.de/), [project number 460234259](https://gepris.dfg.de/gepris/projekt/460234259).

