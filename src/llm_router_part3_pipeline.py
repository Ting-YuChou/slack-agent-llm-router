"""
LLM Router Part 3: Kafka Pipeline - Stream Processing and Data Storage
Handles real-time data ingestion, processing, and storage to ClickHouse
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import uuid

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
import clickhouse_connect
from clickhouse_connect.driver import Client

from src.utils.logger import setup_logging
from src.utils.metrics import PIPELINE_METRICS
from src.utils.schema import QueryRequest, InferenceResponse, SystemMetric, UserTier

logger = logging.getLogger(__name__)


@dataclass
class QueryLogEntry:
    """Structure for query log entries"""
    query_id: str
    timestamp: datetime
    user_id: str
    user_tier: str
    query_text: str
    query_type: str
    selected_model: str
    token_count_input: int
    token_count_output: int
    latency_ms: int
    cost_usd: float
    status: str
    error_message: str = ""
    context_compressed: bool = False
    compression_ratio: float = 0.0
    cached_response: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        return {
            'query_id': self.query_id,
            'timestamp': self.timestamp,
            'user_id': self.user_id,
            'user_tier': self.user_tier.value if isinstance(self.user_tier, UserTier) else    str(self.user_tier),
            'query_text': self.query_text[:10000],
            'query_type': self.query_type.value if hasattr(self.query_type, 'value') else str(self.query_type),
            'selected_model': self.selected_model,
            'token_count_input': self.token_count_input,
            'token_count_output': self.token_count_output,
            'latency_ms': self.latency_ms,
            'cost_usd': self.cost_usd,
            'status': self.status,
            'error_message': self.error_message[:1000] if self.error_message else "",
            'context_compressed': self.context_compressed,
            'compression_ratio': self.compression_ratio,
            'cached_response': self.cached_response
    }


@dataclass
class MetricEntry:
    """Structure for system metrics"""
    timestamp: datetime
    service: str
    metric_name: str
    metric_value: float
    labels: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        return {
            'timestamp': self.timestamp,
            'service': self.service,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'labels': labels  # ClickHouse Map type
        }


class KafkaProducerManager:
    """Manages Kafka message production"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bootstrap_servers = config.get('bootstrap_servers', ['localhost:9092'])
        self.producer = None
        self.topics = config.get('topics', {})
        
        # Producer configuration
        self.producer_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'value_serializer': self._serialize_message,
            'key_serializer': lambda x: x.encode('utf-8') if x else None,
            'acks': config.get('producer', {}).get('acks', 'all'),
            'retries': config.get('producer', {}).get('retries', 3),
            'batch_size': config.get('producer', {}).get('batch_size', 16384),
            'linger_ms': config.get('producer', {}).get('linger_ms', 5),
            'compression_type': config.get('producer', {}).get('compression_type', 'gzip')
        }
        
    async def initialize(self):
        """Initialize Kafka producer"""
        try:
            self.producer = AIOKafkaProducer(**self.producer_config)
            await self.producer.start()
            logger.info("Kafka producer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    async def produce_query_log(self, query_request: QueryRequest, inference_response: InferenceResponse, routing_decision: Any):
        """Produce query log message"""
        try:
            query_log = QueryLogEntry(
                query_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                user_id=query_request.user_id,
                user_tier=query_request.user_tier,
                query_text=query_request.query,
                query_type=routing_decision.query_type.value if hasattr(routing_decision.query_type, 'value') else str(routing_decision.query_type),
                selected_model=inference_response.model_name,
                token_count_input=inference_response.token_count_input,
                token_count_output=inference_response.token_count_output,
                latency_ms=inference_response.latency_ms,
                cost_usd=inference_response.cost_usd,
                status='success' if not hasattr(inference_response, 'error') else 'error',
                error_message=getattr(inference_response, 'error', ''),
                context_compressed=getattr(inference_response, 'compressed_context', False),
                compression_ratio=0.3 if getattr(inference_response, 'compressed_context', False) else 0.0,
                cached_response=inference_response.cached
            )
            
            await self.producer.send(
                topic=self.topics.get('queries', 'llm-queries'),
                key=query_log.user_id,
                value=query_log.to_dict()
            )
            
            PIPELINE_METRICS.messages_produced.labels(topic='queries').inc()
            
        except Exception as e:
            logger.error(f"Failed to produce query log: {e}")
            PIPELINE_METRICS.producer_errors.inc()
    
    async def produce_response_log(self, response_data: Dict[str, Any]):
        """Produce response log message"""
        try:
            await self.producer.send(
                topic=self.topics.get('responses', 'llm-responses'),
                key=response_data.get('query_id'),
                value=response_data
            )
            
            PIPELINE_METRICS.messages_produced.labels(topic='responses').inc()
            
        except Exception as e:
            logger.error(f"Failed to produce response log: {e}")
            PIPELINE_METRICS.producer_errors.inc()
    
    async def produce_metric(self, service: str, metric_name: str, metric_value: float, labels: Dict[str, str] = None):
        """Produce system metric"""
        try:
            metric_entry = MetricEntry(
                timestamp=datetime.now(timezone.utc),
                service=service,
                metric_name=metric_name,
                metric_value=metric_value,
                labels=labels or {}
            )
            
            await self.producer.send(
                topic=self.topics.get('metrics', 'llm-metrics'),
                key=f"{service}:{metric_name}",
                value=metric_entry.to_dict()
            )
            
            PIPELINE_METRICS.messages_produced.labels(topic='metrics').inc()
            
        except Exception as e:
            logger.error(f"Failed to produce metric: {e}")
            PIPELINE_METRICS.producer_errors.inc()
    
    async def produce_error(self, error_data: Dict[str, Any]):
        """Produce error message"""
        try:
            error_data['timestamp'] = datetime.now(timezone.utc)
            
            await self.producer.send(
                topic=self.topics.get('errors', 'llm-errors'),
                key=error_data.get('error_id', str(uuid.uuid4())),
                value=error_data
            )
            
            PIPELINE_METRICS.messages_produced.labels(topic='errors').inc()
            
        except Exception as e:
            logger.error(f"Failed to produce error: {e}")
            PIPELINE_METRICS.producer_errors.inc()
    
    def _serialize_message(self, message: Any) -> bytes:
        """Serialize message to JSON bytes"""
        if isinstance(message, dict):
            # Handle datetime serialization
            def default_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            return json.dumps(message, default=default_serializer).encode('utf-8')
        else:
            return json.dumps(message).encode('utf-8')
    
    async def shutdown(self):
        """Shutdown producer"""
        if self.producer:
            await self.producer.stop()
            logger.info("Kafka producer shutdown complete")


class ClickHouseManager:
    """Manages ClickHouse database operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client: Optional[Client] = None
        self.database = config.get('database', 'llm_router')
        self.tables = config.get('tables', {})
        
        # Connection settings
        self.connection_params = {
            'host': config.get('host', 'localhost'),
            'port': config.get('port', 8123),
            'username': config.get('username', 'default'),
            'password': config.get('password', ''),
            'database': self.database
        }
        
        # Batch insertion settings
        self.batch_size = 1000
        self.batch_timeout = 30  # seconds
        self.pending_batches = {
            'query_logs': [],
            'metrics': [],
            'model_performance': []
        }
        self.last_batch_time = time.time()
        
    async def initialize(self):
        """Initialize ClickHouse connection and create tables"""
        try:
            self.client = clickhouse_connect.get_client(**self.connection_params)
            
            # Test connection
            result = self.client.query('SELECT 1')
            if result.result_rows:
                logger.info("ClickHouse connection established successfully")
            
            # Create database if not exists
            await self._create_database()
            
            # Create tables
            await self._create_tables()
            
            logger.info("ClickHouse manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ClickHouse: {e}")
            raise
    
    async def _create_database(self):
        """Create database if not exists"""
        try:
            self.client.command(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            logger.info(f"Database {self.database} created/verified")
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise
    
    async def _create_tables(self):
        """Create all required tables"""
        # Query logs table
        query_logs_sql = """
        CREATE TABLE IF NOT EXISTS {database}.query_logs (
            query_id String,
            timestamp DateTime64(3),
            user_id String,
            user_tier Enum8('free' = 1, 'premium' = 2, 'enterprise' = 3),
            query_text String,
            query_type String,
            selected_model String,
            token_count_input UInt32,
            token_count_output UInt32,
            latency_ms UInt32,
            cost_usd Float64,
            status Enum8('success' = 1, 'error' = 2, 'timeout' = 3),
            error_message String,
            context_compressed Boolean,
            compression_ratio Float32,
            cached_response Boolean
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, user_id, query_id)
        TTL timestamp + INTERVAL 90 DAY
        """.format(database=self.database)
        
        # System metrics table
        metrics_sql = """
        CREATE TABLE IF NOT EXISTS {database}.system_metrics (
            timestamp DateTime64(3),
            service String,
            metric_name String,
            metric_value Float64,
            labels Map(String, String)
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (service, metric_name, timestamp)
        TTL timestamp + INTERVAL 30 DAY
        """.format(database=self.database)
        
        # Model performance table
        performance_sql = """
        CREATE TABLE IF NOT EXISTS {database}.model_performance (
            timestamp DateTime64(3),
            model_name String,
            requests_count UInt32,
            success_rate Float32,
            avg_latency_ms Float32,
            avg_tokens_per_second Float32,
            error_count UInt32,
            total_cost_usd Float64,
            gpu_utilization Float32,
            memory_usage_gb Float32
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (model_name, timestamp)
        TTL timestamp + INTERVAL 60 DAY
        """.format(database=self.database)
        
        # User analytics table
        user_analytics_sql = """
        CREATE TABLE IF NOT EXISTS {database}.user_analytics (
            user_id String,
            date Date,
            total_queries UInt32,
            total_tokens UInt32,
            total_cost_usd Float64,
            avg_query_length Float32,
            most_used_model String,
            query_types Array(String),
            satisfaction_score Float32
        ) ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (user_id, date)
        TTL date + INTERVAL 180 DAY
        """.format(database=self.database)
        
        # Execute table creation
        tables = [
            ("query_logs", query_logs_sql),
            ("system_metrics", metrics_sql),
            ("model_performance", performance_sql),
            ("user_analytics", user_analytics_sql)
        ]
        
        for table_name, sql in tables:
            try:
                self.client.command(sql)
                logger.info(f"Table {table_name} created/verified")
            except Exception as e:
                logger.error(f"Failed to create table {table_name}: {e}")
    
    async def insert_query_log(self, query_log: QueryLogEntry):
        """Insert single query log entry"""
        try:
            data = [query_log.to_dict()]
            self.client.insert('query_logs', data)
            logger.debug(f"Inserted query log: {query_log.query_id}")
        except Exception as e:
            logger.error(f"Failed to insert query log: {e}")
    
    async def batch_insert_query_logs(self, query_logs: List[QueryLogEntry]):
        """Batch insert query log entries"""
        if not query_logs:
            return
        
        try:
            data = [log.to_dict() for log in query_logs]
            self.client.insert('query_logs', data)
            logger.info(f"Batch inserted {len(query_logs)} query logs")
            PIPELINE_METRICS.records_inserted.labels(table='query_logs').inc(len(query_logs))
        except Exception as e:
            logger.error(f"Failed to batch insert query logs: {e}")
            PIPELINE_METRICS.insertion_errors.inc()
    
    async def insert_metric(self, metric: MetricEntry):
        """Insert single metric entry"""
        try:
            data = [metric.to_dict()]
            self.client.insert('system_metrics', data)
            logger.debug(f"Inserted metric: {metric.service}.{metric.metric_name}")
        except Exception as e:
            logger.error(f"Failed to insert metric: {e}")
    
    async def batch_insert_metrics(self, metrics: List[MetricEntry]):
        """Batch insert metric entries"""
        if not metrics:
            return
        
        try:
            data = [metric.to_dict() for metric in metrics]
            self.client.insert('system_metrics', data)
            logger.info(f"Batch inserted {len(metrics)} metrics")
            PIPELINE_METRICS.records_inserted.labels(table='system_metrics').inc(len(metrics))
        except Exception as e:
            logger.error(f"Failed to batch insert metrics: {e}")
            PIPELINE_METRICS.insertion_errors.inc()
    
    async def get_query_analytics(self, user_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get query analytics for dashboard"""
        try:
            base_query = f"""
            SELECT 
                count() as total_queries,
                sum(token_count_input + token_count_output) as total_tokens,
                sum(cost_usd) as total_cost,
                avg(latency_ms) as avg_latency,
                countIf(status = 'success') * 100.0 / count() as success_rate,
                selected_model,
                query_type
            FROM {self.database}.query_logs 
            WHERE timestamp >= now() - INTERVAL {hours} HOUR
            """
            
            if user_id:
                base_query += f" AND user_id = '{user_id}'"
            
            # Overall stats
            overall_query = base_query + " GROUP BY selected_model, query_type"
            result = self.client.query(overall_query)
            
            # Process results
            analytics = {
                'total_queries': 0,
                'total_tokens': 0,
                'total_cost': 0.0,
                'avg_latency': 0.0,
                'success_rate': 0.0,
                'model_breakdown': {},
                'query_type_breakdown': {}
            }
            
            for row in result.result_rows:
                analytics['total_queries'] += row[0]
                analytics['total_tokens'] += row[1]
                analytics['total_cost'] += row[2]
                
                model = row[5]
                query_type = row[6]
                
                if model not in analytics['model_breakdown']:
                    analytics['model_breakdown'][model] = {'queries': 0, 'cost': 0.0}
                analytics['model_breakdown'][model]['queries'] += row[0]
                analytics['model_breakdown'][model]['cost'] += row[2]
                
                if query_type not in analytics['query_type_breakdown']:
                    analytics['query_type_breakdown'][query_type] = 0
                analytics['query_type_breakdown'][query_type] += row[0]
            
            # Calculate averages
            if analytics['total_queries'] > 0:
                analytics['avg_latency'] = sum(row[3] for row in result.result_rows) / len(result.result_rows)
                analytics['success_rate'] = sum(row[4] for row in result.result_rows) / len(result.result_rows)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get query analytics: {e}")
            return {}
    
    async def get_model_performance(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get model performance metrics"""
        try:
            query = f"""
            SELECT 
                selected_model,
                count() as requests,
                countIf(status = 'success') * 100.0 / count() as success_rate,
                avg(latency_ms) as avg_latency,
                sum(token_count_input + token_count_output) / sum(latency_ms) * 1000 as tokens_per_second,
                countIf(status != 'success') as errors,
                sum(cost_usd) as total_cost
            FROM {self.database}.query_logs 
            WHERE timestamp >= now() - INTERVAL {hours} HOUR
            GROUP BY selected_model
            ORDER BY requests DESC
            """
            
            result = self.client.query(query)
            
            performance_data = []
            for row in result.result_rows:
                performance_data.append({
                    'model_name': row[0],
                    'requests': row[1],
                    'success_rate': row[2],
                    'avg_latency_ms': row[3],
                    'tokens_per_second': row[4],
                    'error_count': row[5],
                    'total_cost': row[6]
                })
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            return []
    
    def shutdown(self):
        """Shutdown ClickHouse connection"""
        if self.client:
            self.client.close()
            logger.info("ClickHouse connection closed")


class KafkaConsumerManager:
    """Manages Kafka message consumption"""
    
    def __init__(self, config: Dict[str, Any], clickhouse_manager: ClickHouseManager):
        self.config = config
        self.clickhouse = clickhouse_manager
        self.bootstrap_servers = config.get('bootstrap_servers', ['localhost:9092'])
        self.topics = config.get('topics', {})
        self.consumers = {}
        
        # Consumer configuration
        self.consumer_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'group_id': config.get('consumer', {}).get('group_id', 'llm-router-consumer'),
            'auto_offset_reset': config.get('consumer', {}).get('auto_offset_reset', 'latest'),
            'enable_auto_commit': config.get('consumer', {}).get('enable_auto_commit', True),
            'max_poll_records': config.get('consumer', {}).get('max_poll_records', 500),
            'value_deserializer': self._deserialize_message
        }
        
        # Batch processing
        self.batch_processors = {
            'queries': [],
            'metrics': []
        }
        self.batch_size = 100
        self.last_batch_time = time.time()
        
    async def initialize(self):
        """Initialize Kafka consumers"""
        try:
            # Create consumer for each topic
            for topic_key, topic_name in self.topics.items():
                if topic_key in ['queries', 'responses', 'metrics', 'errors']:
                    consumer = AIOKafkaConsumer(
                        topic_name,
                        **self.consumer_config
                    )
                    await consumer.start()
                    self.consumers[topic_key] = consumer
                    logger.info(f"Consumer for topic {topic_name} initialized")
            
            logger.info(f"Kafka consumers initialized for {len(self.consumers)} topics")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumers: {e}")
            raise
    
    async def start_consuming(self):
        """Start consuming messages from all topics"""
        tasks = []
        
        # Start consumer tasks
        for topic_key, consumer in self.consumers.items():
            if topic_key == 'queries':
                tasks.append(asyncio.create_task(self._consume_queries(consumer)))
            elif topic_key == 'responses':
                tasks.append(asyncio.create_task(self._consume_responses(consumer)))
            elif topic_key == 'metrics':
                tasks.append(asyncio.create_task(self._consume_metrics(consumer)))
            elif topic_key == 'errors':
                tasks.append(asyncio.create_task(self._consume_errors(consumer)))
        
        # Start batch processor
        tasks.append(asyncio.create_task(self._batch_processor()))
        
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _consume_queries(self, consumer: AIOKafkaConsumer):
        """Consume query log messages"""
        try:
            async for message in consumer:
                try:
                    data = message.value
                    query_log = QueryLogEntry(
                        query_id=data['query_id'],
                        timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                        user_id=data['user_id'],
                        user_tier=data['user_tier'],
                        query_text=data['query_text'],
                        query_type=data['query_type'],
                        selected_model=data['selected_model'],
                        token_count_input=data['token_count_input'],
                        token_count_output=data['token_count_output'],
                        latency_ms=data['latency_ms'],
                        cost_usd=data['cost_usd'],
                        status=data['status'],
                        error_message=data.get('error_message', ''),
                        context_compressed=data.get('context_compressed', False),
                        compression_ratio=data.get('compression_ratio', 0.0),
                        cached_response=data.get('cached_response', False)
                    )
                    
                    # Add to batch
                    self.batch_processors['queries'].append(query_log)
                    
                    PIPELINE_METRICS.messages_consumed.labels(topic='queries').inc()
                    
                except Exception as e:
                    logger.error(f"Failed to process query message: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()
                    
        except Exception as e:
            logger.error(f"Query consumer error: {e}")
    
    async def _consume_responses(self, consumer: AIOKafkaConsumer):
        """Consume response log messages"""
        try:
            async for message in consumer:
                try:
                    data = message.value
                    # Process response data (for future analytics)
                    logger.debug(f"Processed response: {data.get('query_id')}")
                    
                    PIPELINE_METRICS.messages_consumed.labels(topic='responses').inc()
                    
                except Exception as e:
                    logger.error(f"Failed to process response message: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()
                    
        except Exception as e:
            logger.error(f"Response consumer error: {e}")
    
    async def _consume_metrics(self, consumer: AIOKafkaConsumer):
        """Consume system metrics messages"""
        try:
            async for message in consumer:
                try:
                    data = message.value
                    metric_entry = MetricEntry(
                        timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                        service=data['service'],
                        metric_name=data['metric_name'],
                        metric_value=data['metric_value'],
                        labels=data.get('labels', {})
                    )
                    
                    # Add to batch
                    self.batch_processors['metrics'].append(metric_entry)
                    
                    PIPELINE_METRICS.messages_consumed.labels(topic='metrics').inc()
                    
                except Exception as e:
                    logger.error(f"Failed to process metric message: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()
                    
        except Exception as e:
            logger.error(f"Metrics consumer error: {e}")
    
    async def _consume_errors(self, consumer: AIOKafkaConsumer):
        """Consume error messages"""
        try:
            async for message in consumer:
                try:
                    data = message.value
                    logger.error(f"System error logged: {data}")
                    
                    PIPELINE_METRICS.messages_consumed.labels(topic='errors').inc()
                    
                except Exception as e:
                    logger.error(f"Failed to process error message: {e}")
                    PIPELINE_METRICS.consumer_errors.inc()
                    
        except Exception as e:
            logger.error(f"Error consumer error: {e}")
    
    async def _batch_processor(self):
        """Process batches periodically"""
        while True:
            try:
                current_time = time.time()
                
                # Process query logs batch
                if (len(self.batch_processors['queries']) >= self.batch_size or 
                    current_time - self.last_batch_time > 30):
                    
                    if self.batch_processors['queries']:
                        await self.clickhouse.batch_insert_query_logs(self.batch_processors['queries'])
                        self.batch_processors['queries'].clear()
                
                # Process metrics batch
                if (len(self.batch_processors['metrics']) >= self.batch_size or 
                    current_time - self.last_batch_time > 30):
                    
                    if self.batch_processors['metrics']:
                        await self.clickhouse.batch_insert_metrics(self.batch_processors['metrics'])
                        self.batch_processors['metrics'].clear()
                
                # Update last batch time
                if current_time - self.last_batch_time > 30:
                    self.last_batch_time = current_time
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(10)
    
    def _deserialize_message(self, message: bytes) -> Dict[str, Any]:
        """Deserialize message from JSON bytes"""
        return json.loads(message.decode('utf-8'))
    
    async def shutdown(self):
        """Shutdown all consumers"""
        for consumer in self.consumers.values():
            await consumer.stop()
        logger.info("Kafka consumers shutdown complete")


class KafkaIngestionPipeline:
    """Main Kafka ingestion pipeline coordinator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clickhouse_manager = ClickHouseManager(config.get('clickhouse', {}))
        self.producer_manager = KafkaProducerManager(config)
        self.consumer_manager = None  # Will be initialized after ClickHouse
        self.running = False
        
    async def initialize(self):
        """Initialize the complete pipeline"""
        logger.info("Initializing Kafka ingestion pipeline...")
        
        # Initialize ClickHouse first
        await self.clickhouse_manager.initialize()
        
        # Initialize producer
        await self.producer_manager.initialize()
        
        # Initialize consumer with ClickHouse manager
        self.consumer_manager = KafkaConsumerManager(self.config, self.clickhouse_manager)
        await self.consumer_manager.initialize()
        
        logger.info("Kafka ingestion pipeline initialized successfully")
    
    async def start(self):
        """Start the pipeline"""
        logger.info("Starting Kafka ingestion pipeline...")
        self.running = True
        
        # Start consuming messages
        await self.consumer_manager.start_consuming()
    
    async def log_query(self, query_request: QueryRequest, inference_response: InferenceResponse, routing_decision: Any):
        """Log query through the pipeline"""
        await self.producer_manager.produce_query_log(query_request, inference_response, routing_decision)
    
    async def log_metric(self, service: str, metric_name: str, metric_value: float, labels: Dict[str, str] = None):
        """Log metric through the pipeline"""
        await self.producer_manager.produce_metric(service, metric_name, metric_value, labels)
    
    async def log_error(self, error_data: Dict[str, Any]):
        """Log error through the pipeline"""
        await self.producer_manager.produce_error(error_data)
    
    async def get_analytics(self, user_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get analytics from ClickHouse"""
        return await self.clickhouse_manager.get_query_analytics(user_id, hours)
    
    async def get_model_performance(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get model performance from ClickHouse"""
        return await self.clickhouse_manager.get_model_performance(hours)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get pipeline health status"""
        return {
            "pipeline_running": self.running,
            "producer_healthy": self.producer_manager.producer is not None,
            "consumer_healthy": bool(self.consumer_manager and self.consumer_manager.consumers),
            "clickhouse_healthy": self.clickhouse_manager.client is not None,
            "total_consumers": len(self.consumer_manager.consumers) if self.consumer_manager else 0
        }
    
    async def shutdown(self):
        """Shutdown the pipeline"""
        logger.info("Shutting down Kafka ingestion pipeline...")
        self.running = False
        
        if self.consumer_manager:
            await self.consumer_manager.shutdown()
        
        await self.producer_manager.shutdown()
        self.clickhouse_manager.shutdown()
        
        logger.info("Kafka ingestion pipeline shutdown complete")
