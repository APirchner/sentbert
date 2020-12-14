FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .