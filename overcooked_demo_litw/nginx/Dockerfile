FROM nginx:1.21-alpine

ARG NGINX_CONF
ARG NGINX_PORT
ARG SERVER_NAME
ARG SERVER_SSL_CERTIFICATE_KEY
ARG SERVER_SSL_CERTIFICATE_CHAIN

RUN rm /etc/nginx/conf.d/default.conf
#RUN rm /etc/nginx/nginx.conf
COPY ./$NGINX_CONF /etc/nginx/conf.d

ENV NGINX_PORT $NGINX_PORT
ENV SERVER_NAME $SERVER_NAME
ENV SERVER_SSL_CERTIFICATE_KEY $SERVER_SSL_CERTIFICATE_KEY
ENV SERVER_SSL_CERTIFICATE_CHAIN $SERVER_SSL_CERTIFICATE_CHAIN
