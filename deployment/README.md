# MLentory Deployment

This folder contains all the necessary configuration files and scripts to deploy the MLentory system using Docker containers.

## Structure

```
deployment/
├── docker-compose.yml # Main container orchestration file
├── hf_etl/ # HuggingFace ETL service
│ ├── Dockerfile.gpu # Dockerfile for GPU-enabled container
│ ├── Dockerfile.no_gpu # Dockerfile for CPU-only container
│ └── run_extract_transform_load.py
├── scheduler/ # Airflow scheduler configuration
├── dags/ # Airflow DAG definitions
├── logs/ # Airflow logs
├── plugins/ # Airflow plugins
├── scripts/ # Airflow scripts
└── requirements.txt # Python dependencies for scheduler
```

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Toolkit (for GPU support)
- At least 8GB of RAM
- (Optional) NVIDIA GPU with CUDA support


## Quick Start

1. Create the required Docker network:

```bash
docker network create mlentory_network
```

2. Choose your deployment profile:

Make sure to be in the deployment folder when running the following commands.

For GPU-enabled deployment:

```bash
docker-compose up -d --profile gpu
```

For CPU-only deployment:

```bash
docker-compose --profile no_gpu up -d
```

## Running ETL Jobs

The ETL process can be triggered through Airflow or manually using the provided Python script:

```bash
docker exec hf_gpu python3 /app/hf_etl/run_extract_transform_load.py [options]
```

Available options:
- `--save-extraction`: Save extraction results
- `--save-transformation`: Save transformation results
- `--save-load-data`: Save load data
- `--from-date YYYY-MM-DD`: Download models from specific date
- `--num-models N`: Number of models to process
- `--output-dir DIR`: Output directory for results

## Services

The system consists of several containerized services:

- **Airflow Components**:
  - Scheduler (Port 8794)
  - Webserver (Port 8080)
  - PostgreSQL Database (Port 5442)

- **ETL Service** (either GPU or no-GPU):
  - HuggingFace model extraction
  - Data transformation
  - Data loading

- **Storage Services**:
  - PostgreSQL (Port 5432)
  - Virtuoso RDF Store (Ports 1111, 8890)
  - Elasticsearch (Ports 9200, 9300)

## Accessing Services

- Airflow UI: http://localhost:8080 (default credentials: admin/admin)
- Virtuoso SPARQL endpoint: http://localhost:8890/sparql
- Elasticsearch: http://localhost:9200
- PostgreSQL: localhost:5432

## Troubleshooting

1. If services fail to start, check:
   - Docker daemon is running
   - Required ports are available
   - Sufficient system resources
   - Network `mlentory_network` exists

2. For GPU-enabled deployment:
   - Verify NVIDIA drivers are installed
   - Check NVIDIA Container Toolkit is properly configured
   - Run `nvidia-smi` to confirm GPU access