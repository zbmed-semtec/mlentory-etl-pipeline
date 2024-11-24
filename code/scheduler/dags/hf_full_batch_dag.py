from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

# Define the DAG
with DAG(
    dag_id="data_pipeline_dag",
    default_args=default_args,
    description="ETL pipeline using Docker containers",
    schedule_interval=None,  # Trigger manually or adjust as needed
    start_date=days_ago(1),
    catchup=False,
) as dag:

    # Task 1: Extraction
    extract_task = DockerOperator(
        task_id="extract_data",
        image="your_extraction_container_image",  # Replace with your Docker image
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",  # Docker socket
        network_mode="bridge",  # Use your preferred network mode
        command="python extract.py",  # Adjust as per your container's command
    )

    # Task 2: Transformation
    transform_task = DockerOperator(
        task_id="transform_data",
        image="your_transformation_container_image",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        command="python transform.py",
    )

    # Task 3: Loading
    load_task = DockerOperator(
        task_id="load_data",
        image="your_loading_container_image",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        command="python load.py",
    )

    # Set task dependencies
    extract_task
