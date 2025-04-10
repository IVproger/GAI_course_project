FROM python:3.11-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
 && poetry install --no-root

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "deploy/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
