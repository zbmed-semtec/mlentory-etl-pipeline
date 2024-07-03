FROM python:3.11

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install torch -f https://download.pytorch.org/whl/cu113

RUN mkdir Transform_Queue

WORKDIR /app_test
# RUN mkdir tests
# COPY ./transform ./Transform
# COPY ./Extractors ./Extractors
# COPY . ./tests
# RUN chmod +x /tests/validate_tests.sh
