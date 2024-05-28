#!/bin/bash
test_results=$(docker-compose --profile ci_test up)
echo "RUNNING THE TESTS $test_results"

last_three_lines=$(echo "$test_results" | tail -n 3)

if echo "$last_three_lines" | grep -q "errors" > /dev/null || echo "$last_three_lines" | grep -q "failed" > /dev/null; then
  echo "Tests failed!"
  exit 1
else
  echo "Tests successful!"
fi

docker-compose --profile ci_test down                                
