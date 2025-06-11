FROM python:3.11-slim

ENV PYTHONENCODING=UTF-8

USER root

RUN set -eux && \
    apt-get update -y && \
    apt-get install -q -y --no-install-recommends --allow-remove-essential \
        ca-certificates gnupg2 bash build-essential python3-pip procps bc

RUN rm -rf /var/lib/{apt,cache,log}

EXPOSE 5000

COPY . /app

WORKDIR /app

RUN pip3 install -r /app/requirements.txt

ENTRYPOINT ["/usr/local/bin/python3"]
CMD ["./launch_backend.py"]

