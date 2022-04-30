# nainstaluj Ubuntu 20.04 LTS
FROM ubuntu:20.04

# nastav jazyk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# nastav apt-get
ARG DEBIAN_FRONTEND=noninteractive

###########################################
# Dependencies
###########################################
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libopencv-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

