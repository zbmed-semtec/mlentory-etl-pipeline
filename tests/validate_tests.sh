#!/bin/bash
docker-compose --profile=ci_test up -d

./wait-for-it.sh mysql:3306

docker exec ci_test pytest ./tests/

if [ $? -eq 0 ]  
then
  echo "Tests successful!"
  # docker-compose --profile ci_test down    
  exit 0
else
  echo "Tests failed!"
  # docker-compose --profile ci_test down    
  exit 1
fi

                            
