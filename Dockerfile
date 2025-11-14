FROM python:3.12-slim
USER root

RUN mkdir -p /app
COPY . /app/
WORKDIR /app/

RUN pip3 install -r requirements.txt

# Correct ENV syntax
ENV AWS_DEFAULT_REGION="us-east-1"
ENV BUCKET_NAME="mynetworksecurity"
ENV PREDICTION_BUCKET_NAME="my-network-datasource"
ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW_CORE_DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW_CORE_ENABLE_XCOM_PICKLING=True

# Install OS packages needed by Airflow
RUN apt update -y && apt install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && apt clean

# Copy entrypoint script
RUN chmod +x start.sh

ENTRYPOINT ["/bin/sh"]
CMD ["start.sh"]
