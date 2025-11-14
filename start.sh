#!/bin/sh

echo "Initializing Airflow DB..."
airflow db migrate

echo "Creating Airflow admin user..."
airflow users create \
    --username admin \
    --password admin \
    --firstname Artur \
    --lastname Dragunov \
    --role Admin \
    --email dragunovartur61@gmail.com || true

echo "Starting Scheduler..."
airflow scheduler &

echo "Starting Webserver..."
airflow webserver
