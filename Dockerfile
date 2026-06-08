FROM python:3.10-slim

WORKDIR /src

# Install runtime deps only
COPY breast-cancer-ml/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy Flask app and pre-trained models (you must commit breast-cancer-ml/models/)
COPY BreastCancerProject/ /src/app/
COPY breast-cancer-ml/models/ /src/app/model/

WORKDIR /src/app
ENV PORT=5000

ARG CACHEBUST=1
RUN echo "cachebust=${CACHEBUST}"

# Serve the Flask app (app.py must expose "app")
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} app:app"]