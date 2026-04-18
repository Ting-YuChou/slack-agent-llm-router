# Slack Agent LLM Router

A production-grade multi-model deployment system for enterprise AI applications with Slack integration, real-time analytics, and intelligent model routing.

## Project Overview

This platform provides intelligent routing between multiple LLM providers (OpenAI, Anthropic, vLLM) with comprehensive monitoring, cost optimization, and enterprise-ready deployment capabilities.

## Features

- Multi-provider LLM routing with cost optimization
- Real-time data pipeline using Kafka and ClickHouse
- Comprehensive monitoring with Prometheus and Grafana
- Slack bot integration with conversation management
- Context compression for long queries
- Response caching and performance optimization
- Docker-based deployment with CI/CD pipeline
- Stream processing with Apache Flink
- Interactive dashboard with Streamlit

## Architecture

The system consists of several core components:

- Model Router: Intelligent query classification and model selection
- Inference Engine: Multi-provider model integration with caching
- Data Pipeline: Real-time data processing and analytics storage
- Monitoring Service: System metrics and alerting
- Slack Integration: Enterprise chat bot with rich responses
- Web Dashboard: Real-time performance and usage analytics

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- 8GB RAM minimum (16GB recommended)
- CUDA-compatible GPU (optional, for local models)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/slack-agent-llm-router.git
cd slack-agent-llm-router
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### Clean Local Env

For a clean virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest tests -q
```

For a clean conda environment:

```bash
conda env create -f environment.yml
conda activate slack-llm-router
python -m pytest tests -q
```

Pytest is configured in `pytest.ini` to disable the built-in capture plugin because the current macOS + conda base environment can segfault while importing `readline`.

### Configuration

Set the required environment variables before starting the API:

```bash
LLM_ROUTER_API_KEYS=dev-api-key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
SLACK_BOT_TOKEN=xoxb-your-slack-token
```

Notes:

- API key auth is enabled in `config/config.yaml` by default.
- `LLM_ROUTER_API_KEYS` accepts a comma-separated list of valid keys.
- The API reads runtime config from `config/config.yaml` unless `LLM_ROUTER_CONFIG` is set.
- For `docker compose` local development, the app containers use `config/config.compose.yaml`.
- Set at least one cloud provider key before sending requests. The compose stack is intentionally cloud-only and does not start a local vLLM model server.

### Running with Docker

Bring up the local full stack:

```bash
docker compose up -d --build
```

This starts:

- Redis for shared state and caching
- Kafka for event streaming
- ClickHouse for analytics storage
- API service on `http://localhost:8080`
- Background workers for the Kafka -> ClickHouse pipeline
- Flink JobManager / TaskManager for stream-processing experiments

Useful follow-up commands:

```bash
docker compose logs -f api workers
docker compose ps
```

### Manual Startup

If you do not want to run the API and workers inside containers:

```bash
# Start infra dependencies
docker compose up -d redis kafka clickhouse

# Start the API in development mode
python main.py start-api --dev --config config/config.yaml

# Start background workers when pipeline/slack/monitoring are enabled
python main.py start-workers --config config/config.yaml
```

For a single-process local run that starts both API and enabled background services:

```bash
python main.py start --config config/config.yaml
```

### Verification

Check that services are running:

```bash
# Liveness
curl http://localhost:8080/live

# Readiness
curl http://localhost:8080/ready

# Test query
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{"query": "Hello, how are you?", "user_id": "test_user"}'
```

`/health` is kept as a backward-compatible alias for `/ready`.

## Project Structure

``` 
slack-agent-llm-router/
├── main.py                           # Application entry point
├── docker-compose.yml                # Local full-stack compose (infra + API + workers)
├── config/
│   ├── config.yaml                   # Host-run configuration
│   └── config.compose.yaml           # Compose runtime configuration
├── docker/
│   ├── Dockerfile                    # API/worker runtime image
│   ├── requirements-runtime.txt      # Runtime-only Python dependencies
│   └── flink/
│       └── Dockerfile                # Flink image build
├── flink/
│   └── analytics_job.py              # Stream processing logic
├── slack/
│   ├── bot.py                        # Slack bot entrypoint
│   └── bot_real.py                   # Slack bot implementation
├── streamlit_ui/
│   └── app.py                        # Optional Streamlit UI source
├── src/
│   ├── models/                       # Provider and engine implementations
│   └── utils/                        # Logging, schema, metrics helpers
└── tests/
    ├── test_main.py                  # API tests
    ├── test_inference.py             # Inference tests
    └── test_router.py                # Routing tests
```

## Configuration

### Model Configuration

Configure available models in `config/config.yaml`:

```yaml
models:
  gpt-5:
    provider: openai
    cost_per_token: 0.00001
    max_tokens: 128000
    capabilities: [text, analysis, coding, general]
  
  claude-sonnet-4-6:
    provider: anthropic
    cost_per_token: 0.000003  
    max_tokens: 1000000
    capabilities: [text, analysis, reasoning]
  
  mistral-7b:
    provider: vllm
    cost_per_token: 0.0
    max_tokens: 8192
    capabilities: [text, coding]
```

### Routing Rules

Define query routing logic:

```yaml
routing_rules:
  enterprise: claude-sonnet-4-6
  heavy_analysis: claude-sonnet-4-6
  code_generation: [gpt-5, mistral-7b]

scoring:
  default_model_bonus: 4
  gpt5_default_bonus: 3
  sonnet_enterprise_bonus: 12
  sonnet_difficult_task_bonus: 10
  local_simple_task_bonus: 12
  simple_task_token_threshold: 1200
```

### Slack Configuration

Configure the Slack bot integration:

```yaml
slack:
  enabled: true
  bot_token_env: SLACK_BOT_TOKEN
  app_token_env: SLACK_APP_TOKEN
  channels:
    - general
  state_backend: redis
  state_key_prefix: slack_state
  redis:
    host: localhost
    port: 6379
    db: 2
    password_env: REDIS_PASSWORD
```

`state_backend: redis` is the recommended production default for Slack because it survives process restarts and supports multi-instance state sharing. Use `file` only for single-node local runs.

For compose-based local development, `config/config.compose.yaml` already points Slack state, the inference cache, and policy-cache Redis to the internal `redis` service name.

## Development

### Running Tests

Execute the test suite:

```bash
# Run all tests
python -m pytest tests -q

# Run specific tests
python -m pytest tests/test_pipeline.py -q
python -m pytest tests/test_inference.py -q

# Generate coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

`pytest.ini` disables pytest's built-in capture plugin because the current macOS + conda base combination can segfault while importing `readline`.

### Code Quality

Maintain code quality with formatting and linting:

```bash
# Format code
black src/ tests/

# Lint code  
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Models

To add a new model provider:

1. Update the model configuration in `config/config.yaml`
2. Implement the provider interface in `src/models/`
3. Add routing rules for the new model
4. Update tests to include the new model

## Monitoring

### Metrics Dashboard

The application exposes these monitoring-oriented API endpoints:

- `GET /dashboard`
- `GET /dashboard/logs`
- `GET /metrics`

When API key auth is enabled, these endpoints require a valid API key. The checked-in root compose file does not start Grafana or Prometheus.

### Key Metrics

Important metrics tracked include:

- Request rates and response times by model
- Cost tracking and optimization opportunities
- Error rates and system health indicators
- Cache performance and hit rates
- User behavior and usage patterns

### Alerting

Alerts are configured for:

- High error rates above 5%
- Response times exceeding 10 seconds
- System resource usage above 90%
- Model availability issues
- Unusual cost patterns

## Deployment

### Docker Production

This repository does not check in a production compose file. To generate deployment assets:

```bash
python main.py deploy --output .
```

### CI/CD Pipeline

The project includes GitHub Actions workflows for:

- Automated testing on pull requests
- Code quality checks and security scans
- Docker image building and publishing
- Deployment to staging and production environments

The repository now includes a test workflow at [`.github/workflows/tests.yml`](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/.github/workflows/tests.yml).

How to use it:

1. Push a branch to GitHub or open a pull request.
2. Open the `Actions` tab in GitHub.
3. Select the `Tests` workflow to see logs and results.
4. To run it manually, open `Actions` -> `Tests` -> `Run workflow`.

What it does:

- Sets up Python 3.12
- Installs the lightweight CI dependencies from `requirements-ci.txt`
- Runs `python -m pytest tests -q`
- Verifies the source tree compiles with `python -m compileall src tests slack`

### Scaling

For high-volume deployments:

- Use multiple application instances behind a load balancer
- Scale Kafka partitions for increased throughput
- Configure ClickHouse clustering for analytics
- Implement Redis clustering for cache scaling

## Security

### Authentication

The platform supports:

- API key authentication for all application endpoints except `/live`, `/ready`, and `/health`
- Request validation via Pydantic models
- Structured application logging for diagnostics and audit trails

### Data Privacy

Security measures include:

- Optional query logging with configurable retention
- Encryption in transit for all communications
- Audit logging for compliance requirements
- Secure credential management

## Troubleshooting

### Common Issues

Connection errors:
- Verify infra services are running with `docker compose ps`
- Check network connectivity between containers
- Validate environment variables and configuration

Performance issues:
- Inspect `/dashboard` or `/metrics` with a valid API key
- Check cache hit rates and optimize if needed
- Review query routing efficiency

Authentication failures:
- Verify API keys are correctly configured
- Check that `LLM_ROUTER_API_KEYS` is set in the API process environment
- Check Slack bot token permissions
- Validate user tier assignments

### Logs

Application logs are available in several locations:

- Application logs: `logs/llm_router.log`
- Container logs: `docker compose logs [service_name]`

## Contributing

### Development Setup

Fork the repository and set up your development environment:

```bash
git clone https://github.com/your-username/slack-agent-llm-router.git
cd slack-agent-llm-router
git checkout -b feature/your-feature-name
pip install -r requirements-dev.txt
```

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include comprehensive docstrings
- Maintain test coverage above 90%
- Update documentation for new features

### Pull Request Process

1. Create a feature branch from main
2. Implement your changes with tests
3. Ensure all tests pass and code is formatted
4. Submit a pull request with a clear description
5. Address any review feedback


## Support

For questions, issues, or contributions:

- GitHub Issues: Report bugs and request features
- Documentation: Refer to component-specific README files
- Community: Join discussions in GitHub Discussions

For enterprise support and custom implementations, contact the development team.
