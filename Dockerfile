FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY chatbot.py .
COPY data_txt ./data_txt

ENV OLLAMA_URL="http://host.docker.internal:11434"
ENV PYTHONUNBUFFERED=1

CMD ["python", "chatbot.py"]