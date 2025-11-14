#!/bin/sh

echo "Migrating Airflow database..."
airflow db migrate

echo "Starting Airflow in standalone mode (includes web UI, scheduler, and default user)..."
exec airflow standalone
