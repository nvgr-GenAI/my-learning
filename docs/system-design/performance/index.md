# Performance & Monitoring ⚡

Optimize system performance, measure key metrics, and ensure your systems operate efficiently at scale.

## 🎯 Learning Objectives

- **Performance Optimization**: Identify and resolve bottlenecks
- **Monitoring**: Track system health and performance metrics
- **Profiling**: Analyze application performance characteristics
- **Capacity Planning**: Plan for future growth and load
- **Observability**: Gain insights into system behavior

## 📚 Topics Overview

### ⚡ Performance Patterns

| Pattern | Purpose | Use Cases |
|---------|---------|-----------|
| **Caching** | Reduce latency | Frequently accessed data |
| **Connection Pooling** | Reuse connections | Database/API calls |
| **Lazy Loading** | Defer expensive operations | Large datasets |
| **Batch Processing** | Group operations | Database writes |
| **Compression** | Reduce data size | Network transfers |

### 📊 Key Metrics

| Metric Type | Examples | Measurement |
|-------------|----------|-------------|
| **Latency** | Response time, P95/P99 | Milliseconds |
| **Throughput** | RPS, TPS | Requests/second |
| **Utilization** | CPU, Memory, Disk | Percentage |
| **Errors** | Error rate, 5xx responses | Percentage |

## 🗂️ Section Navigation

### Performance Optimization
- [Performance Fundamentals](fundamentals.md)
- [Caching Strategies](caching-strategies.md)
- [Database Optimization](database-optimization.md)
- [Network Optimization](network-optimization.md)
- [Frontend Performance](frontend-performance.md)

### Monitoring & Metrics
- [Monitoring Strategies](monitoring-strategies.md)
- [Key Performance Indicators](key-metrics.md)
- [Alerting Best Practices](alerting.md)
- [Log Analysis](log-analysis.md)
- [Dashboards & Visualization](dashboards.md)

### Profiling & Analysis
- [Application Profiling](profiling.md)
- [Performance Testing](performance-testing.md)
- [Load Testing](load-testing.md)
- [Bottleneck Analysis](bottleneck-analysis.md)
- [Memory Management](memory-management.md)

### Capacity Planning
- [Capacity Planning](capacity-planning.md)
- [Traffic Forecasting](traffic-forecasting.md)
- [Resource Scaling](resource-scaling.md)
- [Cost Optimization](cost-optimization.md)

## ⚡ Performance Optimization Techniques

### Caching Implementation

```python
import redis
import json
import time
from functools import wraps

class CacheManager:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def cache_result(self, expiration=3600):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Try to get from cache
                cached_result = self.redis.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.redis.setex(
                    cache_key, 
                    expiration, 
                    json.dumps(result, default=str)
                )
                return result
            return wrapper
        return decorator

# Usage
cache = CacheManager(redis.Redis())

@cache.cache_result(expiration=1800)
def expensive_operation(param1, param2):
    time.sleep(2)  # Simulate expensive operation
    return f"Result for {param1}, {param2}"
```

### Connection Pool Management

```python
import psycopg2
from psycopg2 import pool
import threading

class DatabaseConnectionPool:
    def __init__(self, database_url, min_conn=1, max_conn=20):
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            min_conn, max_conn, database_url
        )
        self.lock = threading.Lock()
    
    def get_connection(self):
        with self.lock:
            return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        with self.lock:
            self.connection_pool.putconn(conn)
    
    def execute_query(self, query, params=None):
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return cursor.rowcount
        finally:
            self.return_connection(conn)
```

### Batch Processing

```python
import asyncio
from typing import List, Callable, Any

class BatchProcessor:
    def __init__(self, batch_size=100, flush_interval=5.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffer = []
        self.last_flush = time.time()
    
    async def add_item(self, item: Any, process_func: Callable[[List], None]):
        self.buffer.append(item)
        
        # Check if we should flush
        should_flush = (
            len(self.buffer) >= self.batch_size or
            time.time() - self.last_flush > self.flush_interval
        )
        
        if should_flush:
            await self.flush(process_func)
    
    async def flush(self, process_func: Callable[[List], None]):
        if not self.buffer:
            return
        
        batch = self.buffer.copy()
        self.buffer.clear()
        self.last_flush = time.time()
        
        # Process batch asynchronously
        await asyncio.create_task(self._process_batch(batch, process_func))
    
    async def _process_batch(self, batch: List, process_func: Callable):
        try:
            await process_func(batch)
        except Exception as e:
            print(f"Batch processing error: {e}")
            # Implement retry logic or dead letter queue
```

## 📊 Monitoring Implementation

### Metrics Collection

```python
import time
import threading
from collections import defaultdict, deque
from typing import Dict, List

class MetricsCollector:
    def __init__(self):
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()
    
    def increment(self, metric_name: str, value: int = 1):
        with self.lock:
            self.counters[metric_name] += value
    
    def gauge(self, metric_name: str, value: float):
        with self.lock:
            self.gauges[metric_name] = value
    
    def timer(self, metric_name: str):
        return TimerContext(self, metric_name)
    
    def record_timer(self, metric_name: str, duration: float):
        with self.lock:
            self.timers[metric_name].append(duration)
            self.histograms[metric_name].append(duration)
    
    def get_metrics(self) -> Dict:
        with self.lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'timers': {
                    name: {
                        'count': len(times),
                        'avg': sum(times) / len(times) if times else 0,
                        'p95': self._percentile(times, 0.95),
                        'p99': self._percentile(times, 0.99)
                    }
                    for name, times in self.timers.items()
                }
            }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        if not values:
            return 0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]

class TimerContext:
    def __init__(self, collector: MetricsCollector, metric_name: str):
        self.collector = collector
        self.metric_name = metric_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.collector.record_timer(self.metric_name, duration)

# Usage
metrics = MetricsCollector()

def api_handler():
    metrics.increment('api.requests')
    
    with metrics.timer('api.response_time'):
        # Your API logic here
        time.sleep(0.1)  # Simulate work
    
    if some_error_condition:
        metrics.increment('api.errors')
```

### Performance Profiler

```python
import cProfile
import pstats
import io
from functools import wraps

class PerformanceProfiler:
    def __init__(self, sort_by='cumulative', top_n=20):
        self.sort_by = sort_by
        self.top_n = top_n
    
    def profile_function(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                pr.disable()
                
                # Generate report
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s)
                ps.sort_stats(self.sort_by)
                ps.print_stats(self.top_n)
                
                print(f"\nProfile for {func.__name__}:")
                print(s.getvalue())
        
        return wrapper
    
    def profile_code_block(self):
        return ProfileContext(self)

class ProfileContext:
    def __init__(self, profiler):
        self.profiler = profiler
        self.pr = cProfile.Profile()
    
    def __enter__(self):
        self.pr.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pr.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s)
        ps.sort_stats(self.profiler.sort_by)
        ps.print_stats(self.profiler.top_n)
        
        print("\nProfile Results:")
        print(s.getvalue())

# Usage
profiler = PerformanceProfiler()

@profiler.profile_function
def expensive_function():
    # Your code here
    pass

# Or use context manager
with profiler.profile_code_block():
    # Code to profile
    pass
```

## 🧪 Load Testing

### Simple Load Test Framework

```python
import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class LoadTestResult:
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    p95_response_time: float
    requests_per_second: float

class LoadTester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.response_times = []
    
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str) -> Dict:
        start_time = time.time()
        try:
            async with session.get(f"{self.base_url}{endpoint}") as response:
                await response.read()
                duration = time.time() - start_time
                self.response_times.append(duration)
                return {
                    'success': True,
                    'status_code': response.status,
                    'duration': duration
                }
        except Exception as e:
            duration = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }
    
    async def run_load_test(self, endpoint: str, concurrent_users: int, 
                           requests_per_user: int) -> LoadTestResult:
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for user in range(concurrent_users):
                for request in range(requests_per_user):
                    task = asyncio.create_task(
                        self.make_request(session, endpoint)
                    )
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        test_duration = end_time - start_time
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            sorted_times = sorted(self.response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p95_response_time = sorted_times[p95_index] if sorted_times else 0
        else:
            avg_response_time = 0
            p95_response_time = 0
        
        return LoadTestResult(
            total_requests=len(results),
            successful_requests=successful,
            failed_requests=failed,
            average_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            requests_per_second=len(results) / test_duration
        )

# Usage
async def run_test():
    tester = LoadTester("http://localhost:8000")
    result = await tester.run_load_test("/api/users", 10, 100)
    
    print(f"Total Requests: {result.total_requests}")
    print(f"Success Rate: {result.successful_requests/result.total_requests*100:.2f}%")
    print(f"Average Response Time: {result.average_response_time:.3f}s")
    print(f"P95 Response Time: {result.p95_response_time:.3f}s")
    print(f"Requests/Second: {result.requests_per_second:.2f}")
```

## 📈 Capacity Planning

### Resource Utilization Calculator

```python
import math
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ResourceRequirements:
    cpu_cores: float
    memory_gb: float
    storage_gb: float
    network_mbps: float

class CapacityPlanner:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_required_capacity(self, 
                                  current_load: Dict,
                                  growth_rate: float,
                                  time_horizon_months: int,
                                  target_utilization: float = 0.7) -> ResourceRequirements:
        """
        Calculate required capacity based on current metrics and growth projections
        """
        
        # Project future load
        future_multiplier = (1 + growth_rate) ** (time_horizon_months / 12)
        projected_load = {
            key: value * future_multiplier 
            for key, value in current_load.items()
        }
        
        # Calculate required resources with safety margin
        safety_factor = 1 / target_utilization
        
        required_cpu = projected_load.get('avg_cpu_usage', 0) * safety_factor
        required_memory = projected_load.get('avg_memory_usage_gb', 0) * safety_factor
        required_storage = projected_load.get('storage_gb', 0) * safety_factor
        required_network = projected_load.get('network_mbps', 0) * safety_factor
        
        return ResourceRequirements(
            cpu_cores=math.ceil(required_cpu),
            memory_gb=math.ceil(required_memory),
            storage_gb=math.ceil(required_storage),
            network_mbps=math.ceil(required_network)
        )
    
    def estimate_scaling_needs(self, 
                              current_rps: float,
                              target_rps: float,
                              current_instances: int) -> int:
        """
        Estimate number of instances needed for target RPS
        """
        scaling_factor = target_rps / current_rps
        required_instances = math.ceil(current_instances * scaling_factor)
        
        # Add buffer for redundancy
        return max(required_instances + 1, 2)  # Minimum 2 instances
```

## 🎯 Best Practices

### Performance Optimization

- **Measure First**: Profile before optimizing
- **Cache Strategically**: Cache expensive operations and frequently accessed data
- **Optimize Queries**: Use proper indexing and query optimization
- **Minimize Network Calls**: Batch operations and use connection pooling
- **Use Appropriate Data Structures**: Choose the right tool for the job

### Monitoring

- **Monitor What Matters**: Focus on user-impacting metrics
- **Set Meaningful Alerts**: Alert on trends, not just thresholds
- **Use SLIs/SLOs**: Define and track service level objectives
- **Implement Health Checks**: Monitor system health continuously
- **Correlate Metrics**: Use distributed tracing to connect the dots

### Capacity Planning

- **Plan for Growth**: Monitor trends and project future needs
- **Consider Peak Loads**: Plan for traffic spikes and seasonal variations
- **Budget for Redundancy**: Include failover capacity in calculations
- **Regular Reviews**: Update capacity plans based on actual usage
- **Cost Optimization**: Balance performance needs with cost constraints

## 🔗 Related Topics

- [Scalability Patterns](../scalability/index.md)
- [Reliability & Security](../reliability-security/index.md)
- [Data & Storage](../data-storage/index.md)
- [System Design Interviews](../interviews/index.md)

## 📖 Further Reading

- [High Performance Browser Networking](https://hpbn.co/)
- [Site Reliability Engineering](https://sre.google/books/)
- [Systems Performance](http://www.brendangregg.com/sysperfbook.html)
- [The Art of Capacity Planning](https://www.oreilly.com/library/view/the-art-of/9780596518578/)
