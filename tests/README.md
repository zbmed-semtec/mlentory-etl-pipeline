# Tests package

This folder contains all the files needed to test the whole backend of the ETL-pipeline.

<img src="../../docs/Readme_images/MLentory Backend TDD Diagrams-Main_component_interaction_Diagram.jpg"/>
<p style=" text-align: center; font-size: 0.8em; color: #cccccc">MLentory Pipeline</p>

The order on which each component is executed in the normal program behavior is the following:

- test_extract_HF_MetadataParser.py
- test_transform_QueueObserver.py
- test_transform_FileProcessor.py
- test_transform_FieldProcessor.py
- test_transform_GraphCreator.py

So you can follow that order to understand the codebase.

## How to use it

### Using shell script
You can directly run the script *validate_tests.sh* to run all the backend tests.
1. You go to the /Tests directory
2. Then you run
    ```
    sh validate_tests.sh
    ```

Is important to note that if you want to use a shell script in Windows you need third party tools like WSL2.
If you follow the instructions on the [Backend README.md](https://github.com/zbmed-semtec/mlentory-etl-pipeline/tree/main/Backend) you should not have problems running it.

### Using docker-compose
If you want to enter the container that runs all the tests, to test something more specific you can do the following:

1. You need to build the images for the containers in the project, 

```
docker-compose --profile test build
```

2. Bring up the container architecture.

```
docker-compose --profile test up
```

3. If you want to access any of the running containers:

```
docker ps #Check the containers that are running
docker exec -it <test_container_name> /bin/bash
```

4. To run the tests inside the container:

```
pytest
```

To learn more about how to run specific tests check the [pytest documentation](https://docs.pytest.org/en/6.2.x/usage.html). If for example you want to run just the tests for the class *test_transform_FieldProcessor.py* you can do the following:

5. Run:
```
pytest Test/test_transform_FieldProcessor.py
```


