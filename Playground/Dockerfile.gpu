FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y \
    wget\
    curl \
    git \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN nvcc --version

RUN pip3 install torch -f https://download.pytorch.org/whl/cu113

COPY requirements.txt .
RUN pip3 install -r requirements.txt


WORKDIR /app
COPY . .

# RUN jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root

