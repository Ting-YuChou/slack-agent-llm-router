"""
Apache Flink Stream Processing Job for LLM Router Platform
High-priority query processing and real-time analytics
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.functions import MapFunction, FilterFunction, KeyedProcessFunction
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common.typeinfo import Types
from pyflink.common.time import Time

logger = logging.getLogger(__name__)


class QueryPriorityClassifier(MapFunction):
    """Classify queries by priority for routing"""
    
    def __init__(self):
        self.high_priority_keywords = [
            'urgent', 'critical', 'emergency', 'asap', 'immediately',
            'production', 'outage', 'down', 'error', 'bug'
        ]
        self.enterprise_users = set()  # Load from config
        
    def map(self, value: str) -> Dict[str, Any]:
        """Map incoming query to priority classification"""
        try:
            data = json.loads(value)
            
            # Extract query information
            query_text = data.get('query_text', '').lower()
            user_tier = data.get('user_tier', 'free')
            user_id = data.get('user_id', '')
            
            # Determine priority
            priority = self._calculate_priority(query_text, user_tier, user_id)
            
            # Add priority and routing info
            data['priority'] = priority
            data['processing_timestamp'] = datetime.now().isoformat()
            data['route_to_fast_lane'] = priority in ['high', 'critical']
            
            return data
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in message: {value}")
            return {'error': 'invalid_json', 'original': value}
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {'error': str(e), 'original': value}
    
    def _calculate_priority(self, query_text: str, user_tier: str, user_id: str) -> str:
        """Calculate query priority based on content and user"""
        # High priority for enterprise users
        if user_tier == 'enterprise':
            return 'high'
        
        # Critical priority for urgent keywords
        if any(keyword in query_text for keyword in self.high_priority_keywords):
            return 'critical'
        
        # Medium priority for premium users
        if user_tier == 'premium':
            return 'medium'
        
        # Default to low priority
        return 'low'


class HighPriorityFilter(FilterFunction):
    """Filter for high-priority queries"""
    
    def filter(self, value: Dict[str, Any]) -> bool:
        """Filter to only high-priority queries"""
        return value.get('priority') in ['high', 'critical']


class QueryMetricsAggregator(KeyedProcessFunction):
    """Aggregate query metrics in real-time windows"""
    
    def __init__(self, window_size_seconds: int = 60):
        self.window_size = window_size_seconds
        
    def open(self, runtime_context):
        """Initialize state"""
        # State to store query counts
        self.query_count_state = runtime_context.get_state(
            ValueStateDescriptor("query_count", Types.LONG())
        )
        
        # State to store latency sum
        self.latency_sum_state = runtime_context.get_state(
            ValueStateDescriptor("latency_sum", Types.LONG())
        )
        
        # State to store error count
        self.error_count_state = runtime_context.get_state(
            ValueStateDescriptor("error_count", Types.LONG())
        )
        
        # State to store window start time
        self.window_start_state = runtime_context.get_state(
            ValueStateDescriptor("window_start", Types.LONG())
        )
    
    def process_element(self, value: Dict[str, Any], ctx: 'KeyedProcessFunction.Context'):
        """Process each query element"""
        current_time = ctx.timestamp()
        
        # Get current state values
        query_count = self.query_count_state.value() or 0
        latency_sum = self.latency_sum_state.value() or 0
        error_count = self.error_count_state.value() or 0
        window_start = self.window_start_state.value()
        
        # Initialize window if needed
        if window_start is None:
            window_start = current_time
            self.window_start_state.update(window_start)
            # Register timer for window end
            ctx.timer_service().register_processing_time_timer(
                window_start + (self.window_size * 1000)
            )
        
        # Update metrics
        query_count += 1
        latency_sum += value.get('latency_ms', 0)
        
        if value.get('status') == 'error':
            error_count += 1
        
        # Update state
        self.query_count_state.update(query_count)
        self.latency_sum_state.update(latency_sum)
        self.error_count_state.update(error_count)
    
    def on_timer(self, timestamp: int, ctx: 'KeyedProcessFunction.OnTimerContext'):
        """Handle window completion"""
        # Get final metrics for this window
        query_count = self.query_count_state.value() or 0
        latency_sum = self.latency_sum_state.value() or 0
        error_count = self.error_count_state.value() or 0
        window_start = self.window_start_state.value()
        
        # Calculate aggregated metrics
        avg_latency = latency_sum / query_count if query_count > 0 else 0
        error_rate = error_count / query_count if query_count > 0 else 0
        
        # Emit aggregated result
        result = {
            'window_start': window_start,
            'window_end': timestamp,
            'model_name': ctx.get_current_key(),
            'query_count': query_count,
            'avg_latency_ms': avg_latency,
            'error_rate': error_rate,
            'queries_per_second': query_count / self.window_size,
            'timestamp': datetime.now().isoformat()
        }
        
        yield result
        
        # Reset state for next window
        self.query_count_state.clear()
        self.latency_sum_state.clear()
        self.error_count_state.clear()
        self.window_start_state.update(timestamp)
        
        # Register next timer
        ctx.timer_service().register_processing_time_timer(
            timestamp + (self.window_size * 1000)
        )


class AnomalyDetector(KeyedProcessFunction):
    """Detect anomalies in query patterns"""
    
    def __init__(self, threshold_multiplier: float = 2.0):
        self.threshold_multiplier = threshold_multiplier
        
    def open(self, runtime_context):
        """Initialize state for anomaly detection"""
        # Rolling average of latencies
        self.latency_history_state = runtime_context.get_list_state(
            ValueStateDescriptor("latency_history", Types.LONG())
        )
        
        # Rolling average of query counts
        self.query_count_history_state = runtime_context.get_list_state(
            ValueStateDescriptor("query_count_history", Types.LONG())
        )
    
    def process_element(self, value: Dict[str, Any], ctx: 'KeyedProcessFunction.Context'):
        """Detect anomalies in query metrics"""
        current_latency = value.get('avg_latency_ms', 0)
        current_query_count = value.get('query_count', 0)
        
        # Get historical data
        latency_history = list(self.latency_history_state.get())
        query_count_history = list(self.query_count_history_state.get())
        
        # Check for latency anomaly
        if len(latency_history) >= 5:  # Need some history for comparison
            avg_historical_latency = sum(latency_history) / len(latency_history)
            if current_latency > avg_historical_latency * self.threshold_multiplier:
                yield self._create_anomaly_alert(
                    'high_latency',
                    value,
                    f"Latency {current_latency}ms is {current_latency/avg_historical_latency:.1f}x higher than average"
                )
        
        # Check for query volume anomaly
        if len(query_count_history) >= 5:
            avg_historical_count = sum(query_count_history) / len(query_count_history)
            if current_query_count > avg_historical_count * self.threshold_multiplier:
                yield self._create_anomaly_alert(
                    'high_volume',
                    value,
                    f"Query volume {current_query_count} is {current_query_count/avg_historical_count:.1f}x higher than average"
                )
        
        # Update history (keep last 10 values)
        latency_history.append(current_latency)
        query_count_history.append(current_query_count)
        
        if len(latency_history) > 10:
            latency_history.pop(0)
        if len(query_count_history) > 10:
            query_count_history.pop(0)
        
        # Update state
        self.latency_history_state.update(latency_history)
        self.query_count_history_state.update(query_count_history)
    
    def _create_anomaly_alert(self, anomaly_type: str, metrics: Dict[str, Any], description: str) -> Dict[str, Any]:
        """Create anomaly alert"""
        return {
            'alert_type': 'anomaly_detected',
            'anomaly_type': anomaly_type,
            'model_name': metrics.get('model_name'),
            'description': description,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'severity': 'warning'
        }


def create_flink_job(config: Dict[str, Any]):
    """Create and configure the Flink streaming job"""
    
    # Create execution environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(config.get('parallelism', 4))
    
    # Configure checkpointing
    env.enable_checkpointing(config.get('checkpoint_interval_ms', 60000))
    
    # Kafka source configuration
    kafka_props = {
        'bootstrap.servers': ','.join(config.get('kafka', {}).get('bootstrap_servers', ['localhost:9092'])),
        'group.id': config.get('kafka', {}).get('consumer_group', 'flink-llm-processor'),
        'auto.offset.reset': 'latest'
    }
    
    # Create Kafka consumer for query logs
    query_consumer = FlinkKafkaConsumer(
        topics=[config.get('kafka', {}).get('topics', {}).get('queries', 'llm-queries')],
        deserialization_schema=SimpleStringSchema(),
        properties=kafka_props
    )
    
    # Create Kafka producer for processed results
    result_producer = FlinkKafkaProducer(
        topic=config.get('kafka', {}).get('topics', {}).get('processed', 'llm-processed'),
        serialization_schema=SimpleStringSchema(),
        producer_config=kafka_props
    )
    
    # Create Kafka producer for alerts
    alert_producer = FlinkKafkaProducer(
        topic=config.get('kafka', {}).get('topics', {}).get('alerts', 'llm-alerts'),
        serialization_schema=SimpleStringSchema(),
        producer_config=kafka_props
    )
    
    # Main processing pipeline
    query_stream = env.add_source(query_consumer, "Kafka Query Source")
    
    # Step 1: Classify query priority
    classified_stream = query_stream.map(
        QueryPriorityClassifier(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    )
    
    # Step 2: Split stream for different processing paths
    high_priority_stream = classified_stream.filter(HighPriorityFilter())
    all_queries_stream = classified_stream
    
    # Step 3: Real-time metrics aggregation (windowed by model)
    metrics_stream = (all_queries_stream
                     .key_by(lambda x: x.get('selected_model', 'unknown'))
                     .process(QueryMetricsAggregator(window_size_seconds=60))
                     .map(lambda x: json.dumps(x), output_type=Types.STRING()))
    
    # Step 4: Anomaly detection
    anomaly_stream = (metrics_stream
                     .map(lambda x: json.loads(x), output_type=Types.PICKLED_BYTE_ARRAY())
                     .key_by(lambda x: x.get('model_name', 'unknown'))
                     .process(AnomalyDetector())
                     .map(lambda x: json.dumps(x), output_type=Types.STRING()))
    
    # Step 5: High-priority query fast lane
    fast_lane_stream = (high_priority_stream
                       .map(lambda x: json.dumps({
                           **x,
                           'fast_lane_processed': True,
                           'processing_timestamp': datetime.now().isoformat()
                       }), output_type=Types.STRING()))
    
    # Output streams
    metrics_stream.add_sink(result_producer, "Metrics Sink")
    anomaly_stream.add_sink(alert_producer, "Anomaly Alert Sink")
    fast_lane_stream.add_sink(result_producer, "Fast Lane Sink")
    
    return env


def main():
    """Main entry point for Flink job"""
    # Load configuration
    config = {
        'parallelism': 4,
        'checkpoint_interval_ms': 60000,
        'kafka': {
            'bootstrap_servers': ['localhost:9092'],
            'consumer_group': 'flink-llm-processor',
            'topics': {
                'queries': 'llm-queries',
                'processed': 'llm-processed',
                'alerts': 'llm-alerts'
            }
        }
    }
    
    # Create and execute job
    env = create_flink_job(config)
    
    # Execute the job
    env.execute("LLM Router Stream Processing Job")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
