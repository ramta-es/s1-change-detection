.PHONY: install test demo serve lint docker-build docker-up

install:
	pip install -e ".[dev,postgis]"

test:
	pytest -q

lint:
	ruff check app tests

# 2-minute demo: synthetic data, no credentials, outputs in runs/demo
demo:
	python -m app.cli run examples/request_synthetic.json --run-id demo
	@echo "open runs/demo/preview.png"

serve:
	uvicorn app.main:app --reload --port 8000

docker-build:
	docker build -t s1-change-detection .

docker-up:
	docker compose up --build
