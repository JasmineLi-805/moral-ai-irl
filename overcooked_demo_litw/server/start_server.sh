#!/bin/bash
#EXAMPLE: docker run
# -v /var/log/data_litws/gunicorn/:/var/log/gunicorn/
# -p 8080:8080 --detach litws:v1

gunicorn --workers=1 --bind=$HOST:8080 --worker-class=eventlet --access-logfile=$LOG_ACCESS --log-file=$LOG_FILE app:app