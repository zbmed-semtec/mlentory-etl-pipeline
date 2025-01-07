#!/bin/bash

# Get the absolute path of the script directory, regardless of where it's called from
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null )"

cd "$SCRIPT_DIR/../config/docker"
docker-compose --profile=ci_test up -d

"$SCRIPT_DIR/wait-for-it.sh" postgres:5432

docker exec ci_test pytest ./unit

EXIT_CODE=$?

docker-compose --profile=ci_test down

exit $EXIT_CODE