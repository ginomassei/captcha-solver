# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster

COPY ./API /API
WORKDIR /API

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

CMD [ "python3", "-m" , "flask", "run", "--host", "0.0.0.0"]