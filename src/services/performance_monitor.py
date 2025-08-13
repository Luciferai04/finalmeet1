from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import threading
from typing import Dict, Any
import psutil
import redis
import json

# Translation metrics as specified in PDF
translation_requests = Counter(
    "translation_requests_total", "Total translation requests", ["language"]
)
translation_latency = Histogram(
    "translation_latency_seconds", "Translation request latency"
)
translation_quality = Gauge(
    "translation_quality_score", "Translation quality score", ["language"]
)

# RL agent metrics as specified in PDF
rl_reward = Gauge("rl_agent_reward", "RL agent reward", ["agent_type"])
rl_episode_length = Histogram("rl_episode_length", "RL episode length", ["agent_type"])

# System metrics as specified in PDF
active_sessions = Gauge(
    "active_sessions_count", "Number of active translation sessions"
)
websocket_connections = Gauge(
    "websocket_connections_count", "Active WebSocket connections"
)

# Additional system metrics
system_cpu_usage = Gauge("system_cpu_usage_percent", "System CPU usage percentage")
system_memory_usage = Gauge("system_memory_usage_bytes", "System memory usage in bytes")
system_disk_usage = Gauge("system_disk_usage_bytes", "System disk usage in bytes")


class PerformanceMonitor:
    """Performance monitoring system with Prometheus metrics"""

    def __init__(
        self, redis_url: str = "redis://localhost:6379", metrics_port: int = 8000
    ):
        self.redis_url = redis_url
        self.metrics_port = metrics_port
        self.running = False
        self.monitor_thread = None

        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
        except BaseException:
            self.redis_client = None
            print("Warning: Redis not available for performance monitoring")

    def start_monitoring(self):
        """Start the performance monitoring system"""
        self.running = True

        # Start Prometheus metrics server
        try:
            start_http_server(self.metrics_port)
            print(
                f"Prometheus metrics server started on port {
                    self.metrics_port}"
            )
        except Exception as e:
            print(f"Failed to start metrics server: {e}")

        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()
        print("Performance monitoring started")

    def stop_monitoring(self):
        """Stop the performance monitoring system"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop with fallback handling"""
        while self.running:
            try:
                self._update_system_metrics()
                self._update_session_metrics()
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                print(f"Monitoring error: {e}. Switching to basic monitoring mode.")
                self._basic_update_metrics()
                time.sleep(5)  # Retry after delay in case of error

    def _basic_update_metrics(self):
        """Basic metrics update with minimal data collection"""
        print("Performing basic metrics update")
        # Basic metrics logging, can be expanded further.

    def _update_system_metrics(self):
        """Update system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            system_memory_usage.set(memory.used)

            # Disk usage
            disk = psutil.disk_usage("/")
            system_disk_usage.set(disk.used)

        except Exception as e:
            print(f"Error updating system metrics: {e}")

    def _update_session_metrics(self):
        """Update session-related metrics from Redis"""
        if not self.redis_client:
            return

        try:
            # Count active sessions
            session_keys = self.redis_client.keys("session:*")
            active_sessions.set(len(session_keys))

            # Count WebSocket connections (mock for now)
            # In real implementation, this would track actual connections
            websocket_connections.set(len(session_keys) * 0.8)  # Estimate

        except Exception as e:
            print(f"Error updating session metrics: {e}")

    def record_translation_request(
        self, language: str, latency: float, quality_score: float = None
    ):
        """Record a translation request with metrics"""
        translation_requests.labels(language=language).inc()
        translation_latency.observe(latency)

        if quality_score is not None:
            translation_quality.labels(language=language).set(quality_score)

    def record_rl_training(self, agent_type: str, reward: float, episode_length: int):
        """Record RL training metrics"""
        rl_reward.labels(agent_type=agent_type).set(reward)
        rl_episode_length.labels(agent_type=agent_type).observe(episode_length)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            session_count = 0
            if self.redis_client:
                try:
                    session_keys = self.redis_client.keys("session:*")
                    session_count = len(session_keys)
                except BaseException:
                    pass

            return {
                "timestamp": time.time(),
                "system": {
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_usage_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                },
                "application": {
                    "active_sessions": session_count,
                    "redis_connected": self.redis_client is not None,
                },
            }
        except Exception as e:
            return {"error": str(e), "timestamp": time.time()}


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def start_performance_monitoring():
    """Start the global performance monitoring"""
    performance_monitor.start_monitoring()


def stop_performance_monitoring():
    """Stop the global performance monitoring"""
    performance_monitor.stop_monitoring()


def get_metrics_summary():
    """Get current metrics summary"""
    return performance_monitor.get_performance_summary()


if __name__ == "__main__":
    # Standalone monitoring service
    monitor = PerformanceMonitor()
    monitor.start_monitoring()

    try:
        print("Performance monitoring running. Press Ctrl+C to stop.")
        while True:
            summary = monitor.get_performance_summary()
            print(
                f"System Summary: CPU: {summary['system']['cpu_usage_percent']:.1f}%, "
                f"Memory: {summary['system']['memory_usage_percent']:.1f}%"
            )
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        monitor.stop_monitoring()
