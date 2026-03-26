PYTHON ?= python
DOCKER_COMPOSE ?= docker compose

.PHONY: up down api api-dev workers loadtest-api smoke smoke-kafka smoke-flink smoke-flink-runtime integration integration-kafka integration-flink integration-flink-runtime

up:
	$(DOCKER_COMPOSE) up -d --build kafka clickhouse flink-jobmanager flink-taskmanager

down:
	$(DOCKER_COMPOSE) down

api:
	$(PYTHON) main.py start-api

api-dev:
	$(PYTHON) main.py start-api --dev

workers:
	$(PYTHON) main.py start-workers

loadtest-api:
	$(PYTHON) scripts/loadtest_api_baseline.py

smoke: up smoke-kafka smoke-flink smoke-flink-runtime

smoke-kafka:
	$(PYTHON) scripts/integration_smoke.py

smoke-flink:
	$(PYTHON) scripts/flink_smoke.py

smoke-flink-runtime:
	$(PYTHON) scripts/flink_runtime_smoke.py

integration: up integration-kafka integration-flink integration-flink-runtime

integration-kafka:
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_integration_pipeline.py

integration-flink:
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_integration_flink.py

integration-flink-runtime:
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_integration_flink_runtime.py
