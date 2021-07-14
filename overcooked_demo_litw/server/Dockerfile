FROM python:3.7-stretch

ARG BUILD_ENV
ARG OVERCOOKED_BRANCH
ARG HARL_BRANCH
ARG GRAPHICS

WORKDIR /app

# Install non-chai dependencies
RUN python3 -m pip install -U pip
COPY ./requirements.txt ./requirements.txt
#RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements.txt

# Clone chai code
RUN git clone https://github.com/nigini/overcooked_ai.git --branch $OVERCOOKED_BRANCH --single-branch /overcooked_ai
RUN git clone https://github.com/nigini/human_aware_rl.git --branch $HARL_BRANCH --single-branch /human_aware_rl

# Dummy data_dir so things don't break
RUN echo "import os; DATA_DIR=os.path.abspath('.')" >> /human_aware_rl/human_aware_rl/data_dir.py

# Install chai dependencies
RUN pip install -e /overcooked_ai
RUN pip install -e /human_aware_rl

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev

# Copy over remaining files
COPY ./start_server.sh ./start_server.sh
COPY ./static ./static
COPY ./*.py ./
#COPY ./graphics/$GRAPHICS ./static/js/graphics.js
COPY ./config.json ./config.json

# Set environment variables that will be used by app.py
ENV HOST 0.0.0.0
ENV CONF_PATH config.json
ENV LOG_ACCESS /var/log/gunicorn/access.log
ENV LOG_FILE /var/log/gunicorn/output.log
RUN mkdir /var/log/gunicorn/

EXPOSE 8080
CMD ./start_server.sh

# Do the thing
#EXPOSE 5000
#CMD ["python", "-u", "app.py"]