FROM tensorflow/tensorflow:2.4.1-gpu

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
