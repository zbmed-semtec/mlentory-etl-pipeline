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

# RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh \
#     && sh ~/miniconda.sh -b -p /opt/conda \
#     && rm ~/miniconda.sh

# ENV PATH /opt/conda/bin:$PATH
# RUN conda update -n base -c defaults conda
# RUN conda install cudatoolkit

RUN nvcc --version


# RUN conda activate pytorch
RUN pip3 install torch -f https://download.pytorch.org/whl/cu113

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# RUN jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root

WORKDIR /app
COPY . .