# mlentory-load-data

## Purpose
To build a system that extracts ML metadata from different platforms, normalizes that data in a common format, stores it, and shares it in a platform to help researchers.

This repository will focus on the implementation of the ETL pipeline that collects, transforms, and stores the metadata.

## Background
This project is part of the NFDI4DataScience initiative, a German Consortium whose vision is to support all steps of the complex and interdisciplinary research data lifecycle, including collecting/creating, processing, analyzing, publishing, archiving, and reusing resources in Data Science and Artificial Intelligence. Our project, in particular, is centered around the metadata of ML models, how to harmonize that data, and how to make it available and searchable for researchers.


<img src="Assets/Readme_images/NFDI4DataScience_structure.png"/>
<p style="text-align: center; font-size: 0.8em; color: #cccccc">General diagram of the NFDI4DataScience main project.</p>

A big effort in this project will be using semantic technologies to handle and store the metadata, from creating a metadata schema for the ML models to storing this information in RDF format.

<img src="Assets/Readme_images/Metadata for ML models-ZB MED draft action-centric.jpg"/>
<p style="text-align: center; font-size: 0.8em; color: #cccccc">The BETA  version of the Metadata Schema proposed by ZB MED</p>

In the end, the system and the data it collects will help researchers and other interested parties find and pick ML models from different platforms to solve their particular problems.
