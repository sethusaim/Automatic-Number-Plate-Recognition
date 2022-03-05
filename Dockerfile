FROM python:3.6.9-buster

COPY . /app

WORKDIR /app

ARG AWS_ACCESS_KEY_ID

ARG AWS_SECRET_ACCESS_KEY

ARG AWS_DEFAULT_REGION

ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID

ENV AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY

ENV AWS_DEFAULT_REGION $AWS_DEFAULT_REGION

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD [ "python","main.py" ]