# syntax=docker/dockerfile:1
# disaster/inference_backend

FROM python:3.10-slim

WORKDIR /code

ARG PORT=3100

ENV PORT=$PORT

COPY requirements.txt .


# Install requirements
RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

EXPOSE $PORT

CMD ["python3", "main.py"]