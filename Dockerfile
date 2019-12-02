FROM ubuntu
RUN apt update && apt install -y python3 && apt install python3-pip -y && pip3 install flask
WORKDIR /app 
COPY server.py .
EXPOSE 5000
CMD ["python", "server.py"]