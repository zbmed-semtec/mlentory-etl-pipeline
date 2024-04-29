FROM python:3.11

COPY Transform/requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir Transform_Queue

WORKDIR /app
COPY Transform .

