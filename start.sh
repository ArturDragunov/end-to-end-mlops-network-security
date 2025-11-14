#!/bin/sh

echo "Migrating Airflow database..."
airflow db migrate

echo "Creating default admin user if needed..."
# If users CLI still works in your version, use it; else skip.
# Example fallback: use REST API or admin UI to create user manually
airflow users create \
    --username admin \
    --password admin \
    --firstname Artur \
    --lastname Dragunov \
    --role Admin \
    --email dragunovartur61@gmail.com

echo "Starting Airflow in standalone mode (includes web UI, scheduler, and default user)..."
exec airflow standalone
