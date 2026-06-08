FROM python:3.10-slim

WORKDIR /src

# Install OS packages for build if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (expects breast-cancer-ml/requirements.txt)
COPY breast-cancer-ml/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copy training project and train model at build time (creates models/)
COPY breast-cancer-ml/ /src/bcml/
RUN python /src/bcml/src/train_model.py

# Copy Flask app and baked model files into image
COPY BreastCancerProject/ /src/app/
# copy models produced above into app/model (if train saved to ../models)
RUN mkdir -p /src/app/model && \
    cp -r /src/bcml/models/* /src/app/model/ || true

WORKDIR /src/app

ENV PORT=5000
# Use gunicorn; let runtime PORT env be used by the platform
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} app:app"]