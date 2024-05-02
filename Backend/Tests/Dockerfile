FROM python:3.11

COPY Tests/requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir Transform_Queue

WORKDIR /app
RUN mkdir Tests
COPY ./Transform ./Transform
COPY ./Extractors ./Extractors
COPY ./Tests ./Tests

