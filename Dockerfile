FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libffi-dev \
    libssl-dev \
    pkg-config \
    # deps for pillow (zlib/jpeg)
    zlib1g-dev \
    libjpeg62-turbo-dev \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Newer pip helps with PEP517/build deps (esp. cryptography on ARM)
RUN pip install --no-cache-dir -U pip setuptools wheel \
  && pip install --no-cache-dir poetry==1.8.3

# Copy only the poetry files first to leverage Docker layer caching
COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

COPY . /app

EXPOSE 8501

#ENV WEAVIATE_URL=http://localhost:8080

WORKDIR /app

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]