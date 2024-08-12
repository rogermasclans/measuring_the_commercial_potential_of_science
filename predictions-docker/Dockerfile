FROM us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest
COPY app /app
WORKDIR /app
USER root
RUN mkdir -p /cache
RUN chmod 755 /cache
ENV TRANSFORMERS_CACHE=/cache
ENV HF_HOME=/cache
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "app.py"]