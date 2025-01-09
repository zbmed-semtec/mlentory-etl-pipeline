from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.models import Variable
import docker
from datetime import datetime


def execute_script_in_container(container_name, script_path, **kwargs):
    client = docker.from_env()
    container = client.containers.get(container_name)
    
    # Build command with arguments
    cmd_args = []
    if kwargs.get('from_date'):
        cmd_args.append(f"--from-date {kwargs['from_date']}")
    if kwargs.get('num_models'):
        cmd_args.append(f"--num-models {kwargs['num_models']}")
    if kwargs.get('save_extraction'):
        cmd_args.append("--save-extraction")
    if kwargs.get('save_transformation'):
        cmd_args.append("--save-transformation")
    if kwargs.get('save_load_data'):
        cmd_args.append("--save-load-data")
    if kwargs.get('output_dir'):
        cmd_args.append(f"--output-dir {kwargs['output_dir']}")

    exec_cmd = f"python3 {script_path} {' '.join(cmd_args)}"
    print(f"Executing command: {exec_cmd}")
    
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
    "trigger_full_HF_extraction_GPU",
    default_args=default_args,
    description="Run HF ETL script inside a running container",
    schedule_interval=None,  # Trigger manually
    catchup=False,
) as dag:
    
    t1 = BashOperator(task_id="print_current_date", bash_command="date")
    
    # Task: Execute script in a running container
    run_script = PythonOperator(
        task_id="hf_gpu",
        python_callable=execute_script_in_container,
        op_kwargs={
            "container_name": "hf_gpu",
            "script_path": "run_extract_transform_load.py",
            # Get values from Airflow Variables with defaults
            "from_date": Variable.get("hf_from_date", default_var="2023-01-01"),
            "num_models": int(Variable.get("hf_num_models", default_var="100")),
            "save_extraction": Variable.get("hf_save_extraction", default_var="false").lower() == "true",
            "save_transformation": Variable.get("hf_save_transformation", default_var="false").lower() == "true",
            "save_load_data": Variable.get("hf_save_load_data", default_var="false").lower() == "true",
            "output_dir": Variable.get("hf_output_dir", default_var="/app/output")
        },
    )

    # Set task dependencies
    t1 >> run_script
