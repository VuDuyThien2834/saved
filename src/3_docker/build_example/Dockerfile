# FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
FROM thienbd4/ai-ocr_chip:1.4.1
LABEL authors="FIS DIP_AI - FIS - FPT COPR"

ENV APP_HOME /ai_ocr
WORKDIR $APP_HOME
COPY . $APP_HOME

# RUN apt update && apt-get install -y wget && apt-get install zip -y && apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get install -y tesseract-ocr
#RUN apt-get install libzbar0 -y  && apt-get install zbar-tools -y
#---------------- Prepare the envirennment
## INSTALL WITH PIP
#RUN ai-ocr/bin/pip install pyzbar
#RUN ai-ocr/bin/pip install readmrz

#RUN unzip -qq recapture.zip
#RUN rm -rf recapture.zip
# RUN unzip root.zip -d /root
# RUN rm -rf root.zip
#RUN rm -rf ./debug/preprocessing/idcard/temp
#RUN mkdir ./debug/preprocessing/idcard/temp
ENV PYTHONPATH="$PYTHONPATH:./engine/"
ENV PYTHONPATH="$PYTHONPATH:./src/"
ENV PYTHONIOENCODING=utf-8
ENV WORKER_COUNT 1
EXPOSE 8686
CMD ai-ocr/bin/gunicorn app_server:app --bind=0.0.0.0:8686 --timeout 500 --workers=${WORKER_COUNT} --threads=5 --worker-connections=1000 --log-level=debug


