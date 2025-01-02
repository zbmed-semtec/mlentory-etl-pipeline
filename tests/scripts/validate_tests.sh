#!/bin/bash
cd "$(dirname "$0")/../config/docker"
docker-compose --profile=ci_test up -d

../scripts/wait-for-it.sh mysql:3306

docker exec ci_test pytest ./tests/

exit $? 