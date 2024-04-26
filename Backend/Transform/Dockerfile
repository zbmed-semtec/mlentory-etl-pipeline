FROM python:3.9

COPY Transform/requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir Transform_Queue

WORKDIR /app
COPY Transform .


# RUN jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root

