FROM python:3.11

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install torch -f https://download.pytorch.org/whl/cu113

RUN mkdir Transform_Queue
RUN mkdir Load_Queue

WORKDIR /app_test
COPY ./wait-for-it.sh ./tests/wait-for-it.sh
RUN chmod +x ./tests/wait-for-it.sh
ENTRYPOINT ["/bin/bash","-c", "\
    ./tests/wait-for-it.sh mysql:3306 -- && \
    echo 'HELLOOOOOOOO' && \
    pytest \
"]
# ENTRYPOINT []
# RUN mkdir tests
# COPY ./transform ./Transform
# COPY ./Extractors ./Extractors
# COPY . ./tests
# RUN chmod +x /tests/validate_tests.sh
