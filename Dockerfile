FROM google/cloud-sdk:slim

COPY . /workspace/
COPY nltk_data /usr/nltk_data
COPY geckodriver /usr/bin/

WORKDIR /workspace

RUN echo deb http://deb.debian.org/debian stable main contrib non-free >> /etc/apt/sources.list && \
    apt-get update && \
    apt install -y python3 python3-pip firefoxdriver && \
    apt clean

RUN pip3 install --upgrade --no-cache-dir .
