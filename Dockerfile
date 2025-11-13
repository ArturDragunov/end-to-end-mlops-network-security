FROM python:3.10-slim-buster
USER root
RUN mkdir /app
COPY . /app/
WORKDIR /app/
RUN pip3 install -r requirements.txt
ENV AWS_DEFAULT_REGION = "us-east-1"
ENV BUCKET_NAME="mynetworksecurity"
ENV PREDICTION_BUCKET_NAME="my-network-datasource"
ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW_CORE_DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW_CORE_ENABLE_XCOM_PICKLING=True
# by default, airflow has sqlite db for metadata
RUN airflow db init
# -f first name, -l last name -p password -r root user name
RUN airflow users create -e dragunovartur61@gmail.com -f artur -l dragunov -p admin -r Admin -u admin
RUN chmod 777 start.sh
RUN apt update -y
ENTRYPOINT [ "/bin/sh" ]
# Launches the Airflow scheduler in the background and
# Starts the Airflow webserver in the foreground

# start.sh is a wrapper to start both processes (scheduler and webserver)
# by default Docker can run only 1 command
CMD ["start.sh"]