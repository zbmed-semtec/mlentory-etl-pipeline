FROM python:3.10

# Create a working directory
WORKDIR /app

# Install dependencies from requirements.txt
COPY requirements.txt ./
RUN pip cache purge
RUN pip install -r requirements.txt

CMD ["tail", "-f", "/dev/null"]