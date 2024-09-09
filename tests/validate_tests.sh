#!/bin/bash
test_results=$(docker-compose --profile ci_test up --abort-on-container-exit)
echo "RUNNING THE TESTS $test_results"

last_three_lines=$(echo "$test_results" | tail -n 3)

if echo "$last_three_lines" | grep -i -q "error" > /dev/null || echo "$last_three_lines" | grep -i -q "errors" > /dev/null || echo "$last_three_lines" | grep -i -q "failed" > /dev/null; then
  echo "Tests failed!"
  exit 1
else
  echo "Tests successful!"
fi

docker-compose --profile ci_test down                                
