FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch
COPY . /app/

RUN python setup.py install

ENTRYPOINT ["spliformer"]
