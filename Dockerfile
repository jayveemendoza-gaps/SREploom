FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc zlib1g-dev

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
