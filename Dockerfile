FROM python:3.9-slim

ENV PYTHONUNBUFFERED=True \
    PORT=9090

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y

RUN git clone -b inference https://github.com/yhsmiley/yolov7.git && cd yolov7 && python3 -m pip install --no-cache-dir .