FROM ubuntu:20.04

RUN apt-get update -y && apt-get install -y python3-pip

COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app/
WORKDIR /app

EXPOSE 8000

CMD ["/usr/bin/python3", "fastapi_main.py"]
