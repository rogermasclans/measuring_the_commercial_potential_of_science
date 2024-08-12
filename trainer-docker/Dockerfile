FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest
COPY trainer /trainer
WORKDIR /trainer
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "trainer.py"]