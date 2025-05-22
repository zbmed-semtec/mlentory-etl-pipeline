#!/bin/bash
# start_mlentory_etl.sh
# Script to set up and start the MLentory environment

set -e  # Exit on error

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run with sudo: sudo ./start_mlentory_etl.sh"
  exit 1
fi

# Load environment variables from .env file if it exists
ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
  echo "Loading environment variables from .env file"
  # Export all variables from .env file
  export $(grep -v '^#' "$ENV_FILE" | xargs)
else
  echo "Warning: .env file not found at $ENV_FILE"
  echo "Create a .env file with your environment variables (e.g., HF_TOKEN=your_token)"
fi


echo "HF_TOKEN is set and will be passed to containers"

# Determine if GPU is available
if command -v nvidia-smi &> /dev/null; then
  PROFILE="gpu"
  echo "GPU detected, using GPU profile"
else
  PROFILE="no_gpu"
  echo "No GPU detected, using CPU-only profile"
fi

# Allow user to override profile
if [ "$1" = "--profile" ] && [ -n "$2" ]; then
  PROFILE="$2"
  echo "Profile override: Using $PROFILE profile"
fi

# Create required directories
echo "Creating required directories..."
mkdir -p ../data/pgadmin_data
mkdir -p ../data/postgres_data
mkdir -p ../data/virtuoso_data
mkdir -p ../data/elasticsearch_data

# Set permissions for pgAdmin
echo "Setting permissions for pgAdmin..."
chown -R 5050:5050 ../data/pgadmin_data
chmod -R 750 ../data/pgadmin_data

# Set permissions for PostgreSQL
echo "Setting permissions for PostgreSQL..."
chown -R 999:999 ../data/postgres_data  # 999 is the postgres user in the container
chmod -R 700 ../data/postgres_data

# Create mlentory network if it doesn't exist
if ! docker network inspect mlentory_network &> /dev/null; then
  echo "Creating mlentory_network..."
  docker network create mlentory_network
fi

# Stop any running containers
echo "Stopping any running containers..."
if command -v docker compose &> /dev/null; then
    docker compose --profile "$PROFILE" down
else
    docker-compose --profile "$PROFILE" down
fi
# docker compose --profile $PROFILE down

# Build containers
echo "Building containers..."
if command -v docker compose &> /dev/null; then
    docker compose --profile "$PROFILE" build
else
    docker-compose --profile "$PROFILE" build
fi

# Start containers with environment variables explicitly passed
echo "Starting containers with profile: $PROFILE"
if command -v docker compose &> /dev/null; then
    docker compose --profile "$PROFILE" --env-file="$ENV_FILE" up -d
else
    docker-compose --profile "$PROFILE" --env-file="$ENV_FILE" up -d
fi
# docker compose --profile $PROFILE --env-file="$ENV_FILE" up -d

echo "MLentory environment started successfully!"
echo ""
echo "Access points:"
echo "- pgAdmin: http://localhost:5050 (admin@admin.com / admin)"
echo "- Airflow: http://localhost:8080 (admin / admin)"
echo "- Virtuoso: http://localhost:8890/conductor"
echo "- Elasticsearch: http://localhost:9200"
echo ""
echo "To view logs: docker-compose --profile $PROFILE logs -f"
echo "To stop: docker-compose --profile $PROFILE down"