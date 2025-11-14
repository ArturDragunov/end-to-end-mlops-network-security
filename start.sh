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

echo "Starting Airflow api-server and scheduler..."
airflow api-server --port 8080 &
exec airflow scheduler