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
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Copy the environment template and configure your settings:

```bash
cp .env.example .env
```

Required environment variables:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key  
SLACK_BOT_TOKEN=xoxb-your-slack-token
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
CLICKHOUSE_HOST=localhost
REDIS_HOST=localhost
```

Update the main configuration file at `config/config.yaml` with your model preferences and routing rules.

### Running with Docker

Start all services using Docker Compose:

```bash
docker-compose up -d
```

This starts:
- Kafka for message streaming
- ClickHouse for analytics storage
- Redis for response caching
- Prometheus for metrics collection
- Grafana for visualization

### Manual Startup

For development, you can run components individually:

```bash
# Start external services
docker-compose up -d kafka clickhouse redis prometheus grafana

# Start the main application
python main.py
```

### Verification

Check that services are running:

```bash
# Health check
curl http://localhost:8080/health

# Test query
curl -X POST http://localhost:8080/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, how are you?", "user_id": "test_user"}'
```

Access the web interfaces:
- Streamlit Dashboard: http://localhost:8501
- Grafana Monitoring: http://localhost:3000 (admin/admin)
- Prometheus Metrics: http://localhost:9090

## Project Structure

```
slack-agent-llm-router/
├── main.py                           # Application entry point
├── config/
│   └── config.yaml                   # Main configuration file
├── data/
│   ├── queries/
│   │   └── logs.csv                  # Sample query data
│   └── processed/
│       └── routed/                   # Processing outputs
├── docker/
│   ├── docker-compose.yml            # Service orchestration
│   ├── Dockerfile                    # Application container
│   └── requirements.txt              # Container dependencies
├── flink/
│   ├── job.py                        # Stream processing logic
│   └── README.md                     # Flink deployment guide
├── kafka/
│   ├── producer.log                  # Producer logs
│   ├── consumer.log                  # Consumer logs
│   └── topics.json                   # Topic configuration
├── clickhouse/
│   ├── schema.sql                    # Database schema
│   └── data/                         # Data storage
├── monitoring/
│   ├── prometheus.yml                # Metrics configuration
│   └── grafana/
│       └── dashboard.json            # Dashboard definition
├── slack/
│   ├── bot.py                        # Slack bot endpoints
│   └── credentials/
│       └── token.txt                 # Bot authentication
├── streamlit_ui/
│   ├── app.py                        # Web dashboard
│   └── config.toml                   # UI settings
├── src/
│   ├── models/
│   │   └── mistral_engine.py         # Model abstraction layer
│   └── utils/
│       ├── logger.py                 # Logging utilities
│       ├── schema.py                 # Data validation models
│       └── metrics.py                # Prometheus metrics
└── tests/
    ├── test_pipeline.py              # Pipeline tests
    ├── test_inference.py             # Inference tests
    └── conftest.py                   # Test configuration
```

## Configuration

### Model Configuration

Configure available models in `config/config.yaml`:

```yaml
models:
  gpt-4-turbo:
    provider: openai
    cost_per_token: 0.00001
    max_tokens: 128000
    capabilities: [text, analysis, coding]
  
  claude-3.5-sonnet:
    provider: anthropic
    cost_per_token: 0.000003  
    max_tokens: 200000
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
  general: claude-3.5-sonnet
  code_generation: mistral-7b
  analysis: gpt-4-turbo
  creative_writing: claude-3.5-sonnet
```

### Slack Configuration

Configure the Slack bot integration:

```yaml
slack:
  enabled: true
  bot_token: "${SLACK_BOT_TOKEN}"
  rate_limits:
    free: 10
    premium: 100
    enterprise: 1000
```

## Development

### Running Tests

Execute the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_pipeline.py -v
pytest tests/test_inference.py -v

# Generate coverage report
pytest tests/ --cov=src --cov-report=html
```

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

The platform provides comprehensive monitoring through:

- Streamlit UI at http://localhost:8501 for real-time metrics
- Grafana dashboards at http://localhost:3000 for detailed analytics
- Prometheus metrics at http://localhost:9090 for raw data

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

Build and deploy the production image:

```bash
docker build -t slack-agent-llm-router:latest .
docker-compose -f docker-compose.prod.yml up -d
```

### CI/CD Pipeline

The project includes GitHub Actions workflows for:

- Automated testing on pull requests
- Code quality checks and security scans
- Docker image building and publishing
- Deployment to staging and production environments

### Scaling

For high-volume deployments:

- Use multiple application instances behind a load balancer
- Scale Kafka partitions for increased throughput
- Configure ClickHouse clustering for analytics
- Implement Redis clustering for cache scaling

## Security

### Authentication

The platform supports:

- API key authentication for all endpoints
- Role-based access control for different user tiers
- Rate limiting to prevent abuse
- Request validation and sanitization

### Data Privacy

Security measures include:

- Optional query logging with configurable retention
- Encryption in transit for all communications
- Audit logging for compliance requirements
- Secure credential management

## Troubleshooting

### Common Issues

Connection errors:
- Verify all services are running with `docker-compose ps`
- Check network connectivity between containers
- Validate environment variables and configuration

Performance issues:
- Monitor resource usage in Grafana dashboards
- Check cache hit rates and optimize if needed
- Review query routing efficiency

Authentication failures:
- Verify API keys are correctly configured
- Check Slack bot token permissions
- Validate user tier assignments

### Logs

Application logs are available in several locations:

- Application logs: `logs/llm_router.log`
- Kafka logs: `kafka/producer.log` and `kafka/consumer.log`
- Container logs: `docker-compose logs [service_name]`

## Contributing

### Development Setup

Fork the repository and set up your development environment:

```bash
git clone https://github.com/your-username/slack-agent-llm-router.git
cd slack-agent-llm-router
git checkout -b feature/your-feature-name
pip install -r requirements.txt
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
