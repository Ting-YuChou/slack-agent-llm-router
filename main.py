#!/usr/bin/env python3
"""
LLM Router & Execution Platform - Main Entry Point
Production-grade multi-model deployment system for Slack AI

This is the unified entry point that orchestrates all system components:
- Model Router Service
- Inference Engine
- Kafka Pipeline
- Monitoring Stack
- Slack Bot Integration
"""

import asyncio
import logging
import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import click
import uvicorn
import yaml
from fastapi import FastAPI
from prometheus_client import start_http_server

# Import all our components
from src.llm_router_part1_router import ModelRouter
from src.llm_router_part2_inference import InferenceEngine
from src.llm_router_part3_pipeline import KafkaIngestionPipeline
from src.llm_router_part4_monitor import MonitoringService
from src.utils.logger import setup_logging
from src.utils.metrics import SYSTEM_METRICS
from slack.bot import SlackBot


class LLMRouterPlatform:
    """Main orchestrator for the LLM Router Platform"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.services: Dict[str, any] = {}
        self.processes: List[mp.Process] = []
        self.shutdown_event = asyncio.Event()
        
        # Setup logging
        setup_logging(
            log_level=self.config.get("logging", {}).get("level", "INFO"),
            log_file=self.config.get("logging", {}).get("file", "logs/llm_router.log")
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration: {e}")
            sys.exit(1)
    
    async def _initialize_services(self):
        """Initialize all platform services"""
        self.logger.info("Initializing LLM Router Platform services...")
        
        # 1. Initialize Model Router
        self.services["router"] = ModelRouter(
            config=self.config.get("router", {})
        )
        await self.services["router"].initialize()
        
        # 2. Initialize Inference Engine
        self.services["inference"] = InferenceEngine(
            config=self.config.get("inference", {}),
            router=self.services["router"]
        )
        await self.services["inference"].initialize()
        
        # 3. Initialize Kafka Pipeline
        self.services["pipeline"] = KafkaIngestionPipeline(
            config=self.config.get("kafka", {})
        )
        await self.services["pipeline"].initialize()
        
        # 4. Initialize Monitoring Service
        self.services["monitoring"] = MonitoringService(
            config=self.config.get("monitoring", {})
        )
        await self.services["monitoring"].initialize()
        
        # 5. Initialize Slack Bot (if enabled)
        if self.config.get("slack", {}).get("enabled", False):
            self.services["slack_bot"] = SlackBot(
                config=self.config.get("slack", {}),
                inference_engine=self.services["inference"]
            )
            await self.services["slack_bot"].initialize()
        
        self.logger.info("All services initialized successfully")
    
    async def _start_services(self):
        """Start all platform services"""
        self.logger.info("Starting platform services...")
        
        # Start Prometheus metrics server
        metrics_port = self.config.get("monitoring", {}).get("prometheus_port", 8000)
        start_http_server(metrics_port)
        self.logger.info(f"Prometheus metrics server started on port {metrics_port}")
        
        # Start services concurrently
        tasks = []
        
        # Start Kafka pipeline
        tasks.append(
            asyncio.create_task(
                self.services["pipeline"].start(),
                name="kafka_pipeline"
            )
        )
        
        # Start monitoring service
        tasks.append(
            asyncio.create_task(
                self.services["monitoring"].start(),
                name="monitoring_service"
            )
        )
        
        # Start Slack bot if enabled
        if "slack_bot" in self.services:
            tasks.append(
                asyncio.create_task(
                    self.services["slack_bot"].start(),
                    name="slack_bot"
                )
            )
        
        # Start FastAPI server for API endpoints
        api_config = self.config.get("api", {})
        api_task = asyncio.create_task(
            self._start_api_server(api_config),
            name="api_server"
        )
        tasks.append(api_task)
        
        self.logger.info("All services started successfully")
        
        # Wait for shutdown signal or service failure
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Service error: {e}")
            await self._shutdown_services()
    
    async def _start_api_server(self, api_config: Dict):
        """Start the FastAPI server"""
        app = self._create_fastapi_app()
        
        config = uvicorn.Config(
            app,
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8080),
            loop="asyncio",
            log_level=api_config.get("log_level", "info")
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application with all endpoints"""
        app = FastAPI(
            title="LLM Router & Execution Platform",
            description="Production-grade multi-model deployment system",
            version="1.0.0"
        )
        
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            service_status = {}
            for name, service in self.services.items():
                service_status[name] = getattr(service, 'is_healthy', lambda: True)()
            
            return {
                "status": "healthy" if all(service_status.values()) else "unhealthy",
                "services": service_status,
                "timestamp": time.time()
            }
        
        # Model routing endpoint
        @app.post("/route")
        async def route_query(query_data: dict):
            try:
                result = await self.services["inference"].process_query(query_data)
                SYSTEM_METRICS.queries_processed.inc()
                return result
            except Exception as e:
                SYSTEM_METRICS.errors.inc()
                self.logger.error(f"Query processing error: {e}")
                return {"error": str(e)}, 500
        
        # Metrics endpoint
        @app.get("/metrics")
        async def get_metrics():
            return await self.services["monitoring"].get_system_metrics()
        
        return app
    
    async def _shutdown_services(self):
        """Gracefully shutdown all services"""
        self.logger.info("Shutting down platform services...")
        
        for name, service in self.services.items():
            try:
                if hasattr(service, 'shutdown'):
                    await service.shutdown()
                self.logger.info(f"Service {name} shutdown successfully")
            except Exception as e:
                self.logger.error(f"Error shutting down service {name}: {e}")
        
        self.logger.info("Platform shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self._shutdown_services())
        sys.exit(0)
    
    async def run(self):
        """Run the complete platform"""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Initialize and start services
            await self._initialize_services()
            await self._start_services()
            
        except Exception as e:
            self.logger.error(f"Platform startup error: {e}")
            await self._shutdown_services()
            sys.exit(1)


@click.group()
def cli():
    """LLM Router Platform CLI"""
    pass


@cli.command()
@click.option("--config", default="config/config.yaml", help="Configuration file path")
@click.option("--dev", is_flag=True, help="Run in development mode")
def start(config: str, dev: bool):
    """Start the LLM Router Platform"""
    if dev:
        # Development mode with auto-reload
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8080,
            reload=True,
            log_level="debug"
        )
    else:
        # Production mode
        platform = LLMRouterPlatform(config_path=config)
        asyncio.run(platform.run())


@cli.command()
def setup():
    """Setup initial project structure and dependencies"""
    from src.llm_router_part0_setup import setup_project_environment
    setup_project_environment()


@cli.command()
@click.option("--service", help="Specific service to check (router, inference, pipeline, monitoring)")
def health(service: Optional[str]):
    """Check system health"""
    # Implementation for health checks
    click.echo(f"Health check for service: {service or 'all'}")


@cli.command()
@click.option("--output", default="docker-compose.yml", help="Output file for Docker Compose")
def deploy(output: str):
    """Generate deployment configurations"""
    from src.llm_router_part5_deploy import generate_deployment_configs
    generate_deployment_configs(output)


if __name__ == "__main__":
    cli()
