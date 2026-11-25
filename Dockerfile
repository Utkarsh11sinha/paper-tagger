# CPU-only image that runs the Streamlit demo on any machine with Docker.
#   docker build -t paper-tagger .
#   docker run -p 8501:8501 paper-tagger
# then open http://localhost:8501
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
