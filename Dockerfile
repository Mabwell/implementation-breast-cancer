FROM python:3.10-slim
WORKDIR /src
COPY breast-cancer-ml/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir gunicorn
COPY BreastCancerProject/ /src/app/
WORKDIR /src/app
RUN python train_model.py
ENV PORT=10000
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0: app:app"]
