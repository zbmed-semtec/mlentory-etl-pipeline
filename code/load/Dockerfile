FROM python:3.11

COPY load/requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir load_queue

ENV DOCKER_VERSION=18.09.4
RUN curl -sfL -o docker.tgz "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" && \
  tar -xzf docker.tgz docker/docker --strip=1 --directory /usr/local/bin && \
  rm docker.tgz
  
WORKDIR /app
COPY load .

