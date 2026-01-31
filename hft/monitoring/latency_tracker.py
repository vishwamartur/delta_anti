"""
Ultra-precise latency tracking for HFT systems
Measures every component of the execution pipeline
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try numpy for faster calculations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class LatencyMetrics:
    """Latency statistics in microseconds"""
    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0
    count: int = 0


class LatencyTracker:
    """
    High-resolution latency tracking
    Measures: network, parsing, strategy, execution
    All times in nanoseconds internally
    """
    
    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        
        # Latency buckets (all in nanoseconds)
        self.network_latency: deque = deque(maxlen=window_size)
        self.parse_latency: deque = deque(maxlen=window_size)
        self.strategy_latency: deque = deque(maxlen=window_size)
        self.execution_latency: deque = deque(maxlen=window_size)
        self.total_latency: deque = deque(maxlen=window_size)
        self.orderbook_latency: deque = deque(maxlen=window_size)
        
        # Timestamp tracking for current pipeline
        self.current_timestamps: Dict[str, int] = {}
        
        # Summary stats
        self.total_measurements = 0
        self.measurements_since_report = 0
    
    def mark(self, stage: str):
        """Mark timestamp for a pipeline stage"""
        self.current_timestamps[stage] = time.perf_counter_ns()
    
    def record_network(self, exchange_timestamp_ns: int, arrival_timestamp_ns: int):
        """Record network latency from exchange to system"""
        latency = arrival_timestamp_ns - exchange_timestamp_ns
        if latency > 0:  # Sanity check
            self.network_latency.append(latency)
    
    def record_orderbook_update(self, update_time_ns: int):
        """Record order book update time"""
        self.orderbook_latency.append(update_time_ns)
    
    def record_pipeline(self):
        """Record full pipeline latency from marked stages"""
        timestamps = self.current_timestamps
        
        if 'market_data_arrival' not in timestamps:
            self.current_timestamps.clear()
            return
        
        arrival = timestamps['market_data_arrival']
        
        # Calculate each stage
        if 'parse_complete' in timestamps:
            parse_lat = timestamps['parse_complete'] - arrival
            self.parse_latency.append(parse_lat)
        
        if 'strategy_complete' in timestamps:
            if 'parse_complete' in timestamps:
                strategy_lat = timestamps['strategy_complete'] - timestamps['parse_complete']
            else:
                strategy_lat = timestamps['strategy_complete'] - arrival
            self.strategy_latency.append(strategy_lat)
        
        if 'order_sent' in timestamps:
            if 'strategy_complete' in timestamps:
                exec_lat = timestamps['order_sent'] - timestamps['strategy_complete']
            else:
                exec_lat = timestamps['order_sent'] - arrival
            self.execution_latency.append(exec_lat)
            
            # Total latency
            total = timestamps['order_sent'] - arrival
            self.total_latency.append(total)
        
        self.total_measurements += 1
        self.measurements_since_report += 1
        
        # Clear for next measurement
        self.current_timestamps.clear()
    
    def _calculate_stats(self, data: deque) -> LatencyMetrics:
        """Calculate statistics for a latency bucket"""
        if not data:
            return LatencyMetrics()
        
        # Convert to list and microseconds
        data_list = list(data)
        data_us = [x / 1000 for x in data_list]  # ns to μs
        
        if HAS_NUMPY:
            arr = np.array(data_us)
            return LatencyMetrics(
                mean=float(np.mean(arr)),
                median=float(np.median(arr)),
                p95=float(np.percentile(arr, 95)),
                p99=float(np.percentile(arr, 99)),
                min=float(np.min(arr)),
                max=float(np.max(arr)),
                std=float(np.std(arr)),
                count=len(arr)
            )
        else:
            # Pure Python fallback
            sorted_data = sorted(data_us)
            n = len(sorted_data)
            return LatencyMetrics(
                mean=sum(data_us) / n,
                median=sorted_data[n // 2],
                p95=sorted_data[int(n * 0.95)],
                p99=sorted_data[int(n * 0.99)],
                min=min(data_us),
                max=max(data_us),
                std=0.0,  # Skip std in pure Python
                count=n
            )
    
    def get_metrics(self, latency_type: str = 'total') -> LatencyMetrics:
        """Get latency statistics for a specific type"""
        data_map = {
            'network': self.network_latency,
            'parse': self.parse_latency,
            'strategy': self.strategy_latency,
            'execution': self.execution_latency,
            'total': self.total_latency,
            'orderbook': self.orderbook_latency
        }
        
        data = data_map.get(latency_type, self.total_latency)
        return self._calculate_stats(data)
    
    def get_all_metrics(self) -> Dict[str, LatencyMetrics]:
        """Get metrics for all latency types"""
        return {
            'network': self.get_metrics('network'),
            'parse': self.get_metrics('parse'),
            'strategy': self.get_metrics('strategy'),
            'execution': self.get_metrics('execution'),
            'total': self.get_metrics('total'),
            'orderbook': self.get_metrics('orderbook')
        }
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        total = self.get_metrics('total')
        return {
            'total_measurements': self.total_measurements,
            'mean_latency_us': total.mean,
            'p95_latency_us': total.p95,
            'p99_latency_us': total.p99,
            'max_latency_us': total.max
        }
    
    def print_report(self):
        """Print latency report to console"""
        print("\n" + "=" * 70)
        print("HFT LATENCY REPORT (microseconds)")
        print("=" * 70)
        
        for lat_type in ['network', 'parse', 'orderbook', 'strategy', 'execution', 'total']:
            metrics = self.get_metrics(lat_type)
            if metrics.count > 0:
                print(f"\n{lat_type.upper()} ({metrics.count} samples):")
                print(f"  Mean: {metrics.mean:.2f} us | Median: {metrics.median:.2f} us")
                print(f"  P95: {metrics.p95:.2f} us | P99: {metrics.p99:.2f} us")
                print(f"  Range: {metrics.min:.2f} - {metrics.max:.2f} us")
        
        print("=" * 70)
        self.measurements_since_report = 0
    
    def check_alerts(self, threshold_us: float = 1000) -> List[str]:
        """Check for latency alerts (threshold in microseconds)"""
        alerts = []
        
        for lat_type in ['total', 'network', 'strategy']:
            metrics = self.get_metrics(lat_type)
            if metrics.p99 > threshold_us:
                alerts.append(
                    f"{lat_type.upper()} P99 latency {metrics.p99:.0f}us > {threshold_us}us threshold"
                )
        
        return alerts


# Singleton instance
latency_tracker = LatencyTracker()
