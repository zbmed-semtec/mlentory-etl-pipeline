# MLentory Deployment

This folder contains all the necessary configuration files and scripts to deploy the MLentory system using Docker containers.

## Structure

```
deployment/
├── docker-compose.yml                    # Main container orchestration file
├── hf_etl/                               # HuggingFace ETL service
│ ├── Dockerfile.gpu
│ ├── Dockerfile.no_gpu
│ └── run_extract_transform_load.py
├── scheduler/                            # Airflow scheduler configuration
│ ├── dags/
│ ├── logs/
│ ├── plugins/
│ ├── scripts/
│ └── requirements.txt
├── start_mlentory_etl.sh                     # Script to set up and start the environment
├── setup_pgadmin.sh                      # Script to set up pgAdmin permissions
├── db_connect.py                         # Database connection utility
└── requirements.txt
```

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Toolkit (for GPU support)
- At least 8GB of RAM
- (Optional) NVIDIA GPU with CUDA support

If you want further information on how to configure your machine to run the MLentory system please refer to the [Installing prerequisites](README.md#installing-prerequisites) section.

## Quick Start

1. Make sure to include a `.env` file with your **Hugging Face Token** inside the `deployment` folder:

```bash
HF_TOKEN=your_hugging_face_token_here
```

2. Use the automated setup script (recommended):

```bash
sudo ./start_mlentory_etl.sh
```

This script will:
- Detect if GPU is available and select the appropriate profile
- Create required directories
- Set proper permissions for data directories
- Create the Docker network if it doesn't exist
- Start all containers with the appropriate profile

You can override the profile selection:

```bash
sudo ./start_mlentory_etl.sh --profile no_gpu
```

3. Alternatively, choose your deployment profile manually:

Make sure to be in the deployment folder when running the following commands.

For GPU-enabled deployment:

```bash
docker-compose --profile gpu up -d
```

For docker compose version 2.0 or higher run:

```bash
docker compose --profile gpu up -d
docker-compose -d --profile up  gpu
```

For CPU-only deployment:

```bash
docker-compose --profile no_gpu up -d
```

For docker compose version 2.0 or higher run:

```bash
docker compose --profile no_gpu up -d
```

## Database Management

The MLentory system includes tools for database management and visualization:

### pgAdmin for PostgreSQL Visualization

A pgAdmin container is included to provide a web-based interface for managing PostgreSQL databases:

- Access URL: http://localhost:5050
- Default credentials:
  - Email: admin@admin.com
  - Password: admin

To set up server connections in pgAdmin:

1. Log in to pgAdmin
2. Right-click on "Servers" in the left panel and select "Create" > "Server..."
3. In the "General" tab, give your server a name (e.g., "Main Database" or "Airflow Database")
4. In the "Connection" tab, enter:
   - For main database:
     - Host: postgres
     - Port: 5432
     - Username: user
     - Password: password
     - Database: history_DB
   - For Airflow database:
     - Host: airflow_postgres
     - Port: 5442
     - Username: airflow
     - Password: airflow
     - Database: airflow

### Database Connection Utility

The `db_connect.py` script provides connection strings for various database clients:

```bash
python db_connect.py
```

This will display connection information for:
- PostgreSQL command line (psql)
- JDBC connection URL
- SQLAlchemy connection URI
- Python connection code

You can specify a particular format:

```bash
python db_connect.py --format psql
```

## Running ETL Jobs

The ETL process can be triggered through Airflow or manually using the provided Python script:

```bash
docker exec hf_gpu python3 /app/hf_etl/run_extract_transform_load.py
```
[options]
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

- **Management Services**:
  - pgAdmin (Port 5050)

## Accessing Services

- Airflow UI: http://localhost:8080 (default credentials: admin/admin)
- pgAdmin: http://localhost:5050 (default credentials: admin@admin.com/admin)
- Virtuoso SPARQL endpoint: http://localhost:8890/sparql
- Elasticsearch: http://localhost:9200
- PostgreSQL: localhost:5432

## Installing prerequisites

If you are in machine with a Unix based operating system you just need to install the Docker and Docker Compose services.

If you are in Windows we recommend installing the Windows subsystem for Linux (WSL 2) and install Ubuntu 20.04. The idea is to have a Linux machine inside Windows so that everything can run smoothly. Particularly when working with machine learning libraries using the Windows service for Docker can become troublesome.

### Setting up Docker on Linux

For Linux distribution like Ubuntu, Debian, CentOS, etc, we do the following:

1. Update your existing list of packages:

```console
sudo apt update
```

2. Install a few prerequisite packages which let apt use packages over HTTPS:

```console
sudo apt install apt-transport-https ca-certificates curl software-properties-common
```

3. Add the GPG key for the official Docker repository:

```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

4. Add the Docker repository to APT sources:

```
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
```

5. Update the package database with the Docker packages:

```
sudo apt update
```

7. Install Docker:

```
sudo apt install docker-ce
```

8. Verify the installation:

```
sudo docker run hello-world
```

### Manage Docker as Non-root User

If you don't want to write sudo before every command, do the following:

1. Create the docker group if it does not exist:

```
sudo groupadd docker
```

2. Add your user to the docker group:

```
sudo usermod -aG docker ${USER}
```

3. Log out and log back in for changes to take effect.

4. Verify you can run Docker commands without sudo:

```
docker run hello-world
```

### Install Docker compose

1. Run this command to download the latest version of Docker Compose:

```
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
```

2. Apply executable permissions to the binary:

```
sudo chmod +x /usr/local/bin/docker-compose
```

3. Verify the installation:

```
docker-compose --version
```

### Setup NVIDIA GPUs

- It is not necessary to have a gpu to run the Backend, but it will make the pipeline run faster.

- You can follow the guide at https://docs.nvidia.com/cuda/wsl-user-guide/index.html if you want to setup the NVDIA GPUs in your WSL.

- But in general you have to guarantee that you have the GPU drivers, the NVIDIA container toolkit, and you have CUDA toolkit install.

- If you are using Windows with WSL you have to install the GPU drivers in Windows, otherwise just install the drivers in your host OS.
  - In Windows you can check the NVIDIA GPU drivers at: https://www.nvidia.com/Download/index.aspx
  - In Ubuntu you can check how to download the drivers at: https://ubuntu.com/server/docs/nvidia-drivers-installation
  - Remember to restart your system after installation.

If you don't have CUDA drivers installed to use your GPU for ML development you can follow the instructions here:
https://developer.nvidia.com/cuda-downloads

### Update the default Docker DNS server

If you are using the WSL or a Linux distribution as your OS you need to configure the following in order for the private container network to resolve outside hostnames and interact correctly with the internet.

1. Install dnsmasq and resolvconf.

```
sudo apt update
sudo apt install dnsmasq resolvconf
```

2. Find your docker IP (in this case, 172.17.0.1):

```
root@host:~# ifconfig | grep -A2 docker0
docker0   Link encap:Ethernet  HWaddr 02:42:bb:b4:4a:50
          inet addr:172.17.0.1  Bcast:0.0.0.0  Mask:255.255.0.0
```

3. Edit /etc/dnsmasq.conf and add these lines:

```
sudo nano /etc/dnsmasq.conf
```

```
interface=docker0
bind-interfaces
listen-address=172.17.0.1
```

5. Create/edit /etc/resolvconf/resolv.conf.d/tail (you can use vim or nano) and add this line, you have to change the line there with the IP of your default network interface eth0:

```
nameserver 8.8.8.8
```

6. Re-read the configuration files and regenerate /etc/resolv.conf.

```
sudo resolvconf -u
```

7. Restart your OS. If you are using WSL run the following in your windows terminal:

```
wsl.exe --shutdown
```

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

3. For database connection issues:
   - Use the `db_connect.py` utility to verify connection parameters
   - Check if the database containers are running: `docker ps | grep postgres`
