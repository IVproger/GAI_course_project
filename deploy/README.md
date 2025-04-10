# 🚀 Deployment Guide

This guide covers both **local development** and **Docker-based deployment** of your Streamlit + FastAPI app.

---

## 🧪 Local Development (no Docker)

Use this setup if you're working locally and want fast reloads during development.

### ✅ 1. Install dependencies

Make sure you have [Poetry](https://python-poetry.org/) installed, then run:

```bash
poetry install
```

### ✅ 2. Enable local backend mode

Open `deploy/app.py` and make sure the following lines are set (around lines 14–15):

```python
USE_BACKEND = True
DOCKER = False
```

### ✅ 3. Run the frontend and backend (in two separate terminal windows)

#### Frontend (Streamlit)

```bash
poetry run python -m streamlit run deploy/app.py
```

#### Backend (FastAPI)

```bash
poetry run python -m uvicorn deploy.api:app --reload
```

### 🔗 Access the app

- Frontend: [http://localhost:8501](http://localhost:8501)
- Backend API: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Docker Deployment

Use this method if you want to run the app in isolated containers (great for testing or production).

### ✅ 1. Enable Docker mode

Open `deploy/app.py` and set:

```python
USE_BACKEND = True
DOCKER = True
```

> This ensures your frontend knows to talk to `http://api:8000` inside Docker.

### ✅ 2. Build and run everything with Docker Compose

In the **project root**, run:

```bash
docker compose up --build
```

### 🔗 Access the app

- Frontend: [http://localhost:8501](http://localhost:8501)
- Backend API: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ✅ Notes & Tips
- You can switch between mock mode and backend mode by toggling `USE_BACKEND` in `app.py`.
- Need to reset everything? Run:
  ```bash
  docker compose down -v
  ```