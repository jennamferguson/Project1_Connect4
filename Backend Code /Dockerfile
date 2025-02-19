FROM python:3.8
ADD requirements.txt /
RUN pip install -r /requirements.txt
RUN mkdir /connect4_backend/
RUN chmod -R 777 /connect4
ADD cnn.py /
ENV PYTHONUNBUFFERED=1
CMD [ "python", "./cnn.py" ]