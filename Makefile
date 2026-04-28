PYTHON ?= python
DOCKER_COMPOSE ?= docker compose

.PHONY: up down api api-dev workers loadtest-api smoke smoke-kafka smoke-redis-stack-memory smoke-flink smoke-flink-runtime smoke-flink-analytics smoke-flink-analytics-runtime integration integration-kafka integration-flink integration-flink-runtime integration-flink-analytics integration-flink-analytics-runtime

up:
	$(DOCKER_COMPOSE) up -d --build redis-stack kafka clickhouse flink-jobmanager flink-taskmanager

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

smoke: up smoke-kafka smoke-flink smoke-flink-runtime smoke-flink-analytics smoke-flink-analytics-runtime

smoke-kafka:
	$(PYTHON) scripts/integration_smoke.py

smoke-redis-stack-memory:
	$(DOCKER_COMPOSE) up -d redis-stack
	$(PYTHON) scripts/redis_stack_memory_smoke.py

smoke-flink:
	$(PYTHON) scripts/flink_smoke.py

smoke-flink-runtime:
	$(PYTHON) scripts/flink_runtime_smoke.py

smoke-flink-analytics:
	$(PYTHON) scripts/flink_analytics_smoke.py

smoke-flink-analytics-runtime:
	$(PYTHON) scripts/flink_analytics_runtime_smoke.py

integration: up integration-kafka integration-flink integration-flink-runtime integration-flink-analytics integration-flink-analytics-runtime

integration-kafka:
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_integration_pipeline.py

integration-flink:
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_integration_flink.py

integration-flink-runtime:
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_integration_flink_runtime.py

integration-flink-analytics:
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_integration_flink_analytics.py

integration-flink-analytics-runtime:
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_integration_flink_analytics_runtime.py
