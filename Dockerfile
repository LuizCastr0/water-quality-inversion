FROM 10.200.99.202:15080/zero2x002/competition-base:ubuntu22.04-py310.19
WORKDIR /app

ARG PIP_CACHE_DIR

COPY requirements.txt .
RUN pip install --cache-dir /tmp/pip-cache -r requirements.txt \
    -i https://repo.huaweicloud.com/repository/pypi/simple

COPY src/ ./src/
COPY models/ ./models/
COPY run.sh .
RUN chmod +x run.sh
