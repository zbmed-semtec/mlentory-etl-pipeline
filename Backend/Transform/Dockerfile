FROM python:3.9

COPY Extractors/HF_Extractor/requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir To_transform

WORKDIR /app
COPY Extractors/HF_Extractor .


# RUN jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root

