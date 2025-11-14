#!/bin/sh
echo "Migrating Airflow database..."
airflow db migrate

echo "Creating admin user..."
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin

echo "Starting Airflow webserver and scheduler..."
airflow webserver --port 8080 &
exec airflow scheduler