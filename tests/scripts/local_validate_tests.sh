#!/bin/bash
docker-compose --profile=ci_test build
docker-compose --profile=ci_test up -d

./wait-for-it.sh mysql:3306

docker exec ci_test pytest ./tests/

exit $?

docker-compose --profile=ci_test down