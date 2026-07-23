# Deployment Guide - Seismic Interpretation

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5013

CMD ["python", "app.py"]
```

### Build and Run

```bash
docker build -t seismic-interpretation .
docker run -p 5013:5013 seismic-interpretation
```

## Docker Compose

```yaml
version: '3.8'
services:
  seismic-interpretation:
    build: .
    ports:
      - "5013:5013"
    environment:
      - FLASK_ENV=production
    volumes:
      - model-data:/app/outputs

volumes:
  model-data:
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment mode | development |
| PORT | Server port | 5013 |

## Production Considerations

- Use gunicorn for production serving:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5013 app:app
  ```
- Set `debug=False` in `app.py` (already set)
- Configure reverse proxy (nginx) for SSL termination
- Set up health check monitoring on `/api/health`
- Use a process manager (systemd, supervisor) for auto-restart

## Training Pipeline

1. `python train.py` generates synthetic seismic data
2. StandardScaler fitted on features
3. Models trained and evaluated
4. Artifacts saved to `outputs/models/`:
   - `horizon_tracker.pkl` - Porosity predictor
   - `facies_classifier.pkl` - Lithology classifier
   - `scaler.pkl` - Feature scaler
   - `metadata.json` - Training summary

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Runs on push to main
- Installs dependencies
- Runs training pipeline
- Executes API tests

---

*Elaborado por Ing. Kelvin Cabrera*
