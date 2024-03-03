# https://github.com/Softcatala/whisper-ctranslate2/issues/75
FROM ubuntu:20.04

# Alternatively, use a base image with CUDA and cuDNN support
# FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y python3-pip

# Set the working directory
WORKDIR /app

# Copy the app code and requirements filed
COPY . /app

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install whisper-ctranslate2
RUN pip install -U whisper-ctranslate2

# Set the entry point
ENTRYPOINT ["whisper-ctranslate2"]