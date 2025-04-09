# Deployment folder

## Local deploy

1) Install dependencies with `poetry`
2) Navigate to `deploy/app.py` and make sure you see this (line 14, 15):
```py
USE_BACKEND = True
DOCKER = False
```
3) in separate windowns run this:
for frontend
```bash
poetry run python -m streamlit run deploy/app.py
```
for backend
```bash
poetry run python -m uvicorn deploy.api:app --reload
```

## Deploy with Docker
1) Navigate to `deploy/app.py` and make sure you see this (line 14, 15):
```py
USE_BACKEND = True
DOCKER = True
```
2) in project root execute this command:
```bash
docker compose up --build
```