FROM python:3.11

COPY transform/requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir transform_queue

WORKDIR /app
COPY transform .

