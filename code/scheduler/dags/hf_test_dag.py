from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.bash_operator import BashOperator
import docker


# Custom Python function to execute a script inside a running container
def execute_script_in_container(container_name, script_path):
    client = docker.from_env()
    container = client.containers.get(container_name)
    exec_cmd = f"python3 {script_path}"
    exit_code, output = container.exec_run(exec_cmd)
    print(output.decode("utf-8"))
    if exit_code != 0:
        raise RuntimeError(f"Script failed with exit code {exit_code}")


# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

# Define DAG
with DAG(
    "Test_dag_container",
    default_args=default_args,
    description="Run a Python script inside different running containers",
    schedule_interval=None,  # Trigger manually
    catchup=False,
) as dag:
    t1 = BashOperator(task_id="print_current_date", bash_command="date")
    # Task: Execute script in a running container
    extract = PythonOperator(
        task_id="hf_gpu",
        python_callable=execute_script_in_container,
        op_kwargs={
            "container_name": "hf_gpu",
            "script_path": "test_extract.py",  # Path inside the container
        },
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=execute_script_in_container,
        op_kwargs={
            "container_name": "transform",
            "script_path": "test_transform.py",  # Path inside the container
        },
    )

    load = PythonOperator(
        task_id="load",
        python_callable=execute_script_in_container,
        op_kwargs={
            "container_name": "load",
            "script_path": "test_load.py",  # Path inside the container
        },
    )

    # Set task dependencies
    t1 >> extract >> transform >> load
