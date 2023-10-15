FROM nvcr.io/nvidia/pytorch:23.08-py3 as BASE

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

CMD [ "/bin/bash" ]

RUN pip install accelerate appdirs loralib bitsandbytes black datasets fire transformers>=4.31.0 sentencepiece py7zr scipy optimum peft wandb

COPY . /Prometheus/
WORKDIR /Prometheus
