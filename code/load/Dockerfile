FROM python:3.11

COPY load/requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir load_queue

WORKDIR /app
COPY load .

