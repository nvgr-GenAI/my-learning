# System Design Principles & Trade-offs ⚖️

Understanding fundamental principles and trade-offs is essential for making informed system design decisions.

## 🎯 Core Design Principles

### 1. Scalability
**Definition**: The ability to handle increased load gracefully.

**Types of Scalability**:

#### Horizontal Scaling (Scale Out)
- **What**: Add more machines to handle load
- **Pros**: Unlimited scaling potential, fault tolerance
- **Cons**: Data consistency complexity, network overhead
- **Best for**: Stateless services, web servers

```python
class HorizontalScaling:
    def __init__(self):
        self.servers = []
        self.load_balancer = LoadBalancer()
    
    def add_server(self, server):
        """Add a new server to the pool"""
        self.servers.append(server)
        self.load_balancer.register_server(server)
        return f"Added server. Total: {len(self.servers)}"
    
    def handle_request(self, request):
        """Distribute request among available servers"""
        server = self.load_balancer.get_server()
        return server.process(request)
```

#### Vertical Scaling (Scale Up)
- **What**: Add more power (CPU, RAM) to existing machines
- **Pros**: Simple, no application changes needed
- **Cons**: Hardware limits, single point of failure
- **Best for**: Databases, legacy applications

```python
class VerticalScaling:
    def __init__(self, cpu_cores=4, memory_gb=16, disk_gb=500):
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_gb = disk_gb
    
    def upgrade_hardware(self, new_cores=None, new_memory=None, new_disk=None):
        """Upgrade hardware specifications"""
        if new_cores:
            self.cpu_cores = new_cores
        if new_memory:
            self.memory_gb = new_memory
        if new_disk:
            self.disk_gb = new_disk
        
        return {
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'disk_gb': self.disk_gb
        }
```

### 2. Reliability
**Definition**: The probability that a system performs correctly over time.

**Key Concepts**:
- **MTBF** (Mean Time Between Failures): Average time between failures
- **MTTR** (Mean Time To Recovery): Average time to recover from failure
- **Availability** = MTBF / (MTBF + MTTR)

```python
import time
import random
from typing import List, Callable

class ReliabilityPatterns:
    
    @staticmethod
    def retry_with_exponential_backoff(
        func: Callable, 
        max_retries: int = 3, 
        base_delay: float = 1.0
    ):
        """Retry failed operations with exponential backoff"""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                delay = base_delay * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed, retrying in {delay}s")
                time.sleep(delay)
    
    @staticmethod
    def circuit_breaker(failure_threshold: int = 5, timeout: int = 60):
        """Prevent cascading failures with circuit breaker pattern"""
        def decorator(func):
            func.failure_count = 0
            func.last_failure_time = 0
            func.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
            
            def wrapper(*args, **kwargs):
                current_time = time.time()
                
                # Check if circuit should move from OPEN to HALF_OPEN
                if func.state == 'OPEN':
                    if current_time - func.last_failure_time >= timeout:
                        func.state = 'HALF_OPEN'
                    else:
                        raise Exception("Circuit breaker is OPEN")
                
                try:
                    result = func(*args, **kwargs)
                    # Success - reset circuit breaker
                    func.failure_count = 0
                    func.state = 'CLOSED'
                    return result
                
                except Exception as e:
                    func.failure_count += 1
                    func.last_failure_time = current_time
                    
                    if func.failure_count >= failure_threshold:
                        func.state = 'OPEN'
                    
                    raise e
            
            return wrapper
        return decorator
```

### 3. Consistency
**Definition**: All nodes see the same data at the same time.

**Consistency Models**:

#### Strong Consistency
```python
import threading

class StrongConsistency:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()
        self.replicas = []
    
    def write(self, key, value):
        """Write with strong consistency - all replicas must acknowledge"""
        with self.lock:
            # Write to all replicas synchronously
            failed_replicas = []
            for replica in self.replicas:
                try:
                    replica.write(key, value)
                except Exception:
                    failed_replicas.append(replica)
            
            if failed_replicas:
                # Rollback if any replica failed
                for replica in self.replicas:
                    if replica not in failed_replicas:
                        replica.rollback(key)
                raise Exception("Write failed - consistency violated")
            
            self.data[key] = value
            return True
    
    def read(self, key):
        """Read with strong consistency"""
        with self.lock:
            return self.data.get(key)
```

#### Eventual Consistency
```python
import asyncio
from collections import deque

class EventualConsistency:
    def __init__(self):
        self.data = {}
        self.replicas = []
        self.pending_updates = deque()
    
    def write(self, key, value):
        """Write with eventual consistency - immediate local write"""
        # Write locally immediately
        self.data[key] = value
        
        # Queue for async replication
        self.pending_updates.append(('write', key, value, time.time()))
        
        # Start async replication (fire and forget)
        asyncio.create_task(self._replicate_async())
        return True
    
    async def _replicate_async(self):
        """Asynchronously replicate to other nodes"""
        while self.pending_updates:
            operation, key, value, timestamp = self.pending_updates.popleft()
            
            for replica in self.replicas:
                try:
                    await replica.async_write(key, value, timestamp)
                except Exception:
                    # Add back to queue for retry
                    self.pending_updates.append((operation, key, value, timestamp))
                    break
```

### 4. Availability
**Definition**: The system remains operational over time.

**Availability Patterns**:

```python
class HighAvailabilityPatterns:
    
    def __init__(self):
        self.primary_db = DatabaseConnection("primary")
        self.secondary_db = DatabaseConnection("secondary")
        self.health_check_interval = 30
    
    def active_passive_failover(self):
        """Primary-secondary failover pattern"""
        if not self.primary_db.is_healthy():
            print("Primary database failed, switching to secondary")
            self.primary_db, self.secondary_db = self.secondary_db, self.primary_db
            return self.primary_db
        
        return self.primary_db
    
    def active_active_load_balancing(self, request):
        """Both nodes actively serve traffic"""
        # Health check both databases
        healthy_dbs = []
        if self.primary_db.is_healthy():
            healthy_dbs.append(self.primary_db)
        if self.secondary_db.is_healthy():
            healthy_dbs.append(self.secondary_db)
        
        if not healthy_dbs:
            raise Exception("No healthy databases available")
        
        # Route based on load or other criteria
        selected_db = healthy_dbs[hash(request.user_id) % len(healthy_dbs)]
        return selected_db.execute(request.query)
```

## ⚖️ Fundamental Trade-offs

### 1. Consistency vs Performance
```python
class ConsistencyPerformanceTradeoff:
    
    def fast_read_eventual_consistency(self, key):
        """Fast read, may return stale data"""
        # Read from local cache or nearest replica
        return self.local_cache.get(key) or self.nearest_replica.read(key)
    
    def slow_read_strong_consistency(self, key):
        """Slower read, guaranteed latest data"""
        # Read from primary or majority of replicas
        values = []
        for replica in self.replicas:
            values.append(replica.read(key))
        
        # Return most recent value (simplified)
        return max(values, key=lambda x: x.timestamp if x else 0)
```

### 2. Latency vs Throughput
```python
class LatencyThroughputTradeoff:
    
    def low_latency_processing(self, request):
        """Process immediately for low latency"""
        # Process request immediately
        return self.process_sync(request)
    
    def high_throughput_batching(self, requests):
        """Batch requests for higher throughput"""
        # Collect requests in batches
        batches = self.create_batches(requests, batch_size=100)
        
        results = []
        for batch in batches:
            # Process entire batch together
            batch_results = self.process_batch(batch)
            results.extend(batch_results)
        
        return results
```

### 3. Space vs Time Complexity
```python
class SpaceTimeTradeoff:
    
    def __init__(self):
        self.cache = {}  # Space for time trade-off
    
    def fibonacci_with_memoization(self, n):
        """Use extra space to reduce time complexity"""
        if n in self.cache:
            return self.cache[n]
        
        if n <= 1:
            return n
        
        result = self.fibonacci_with_memoization(n-1) + self.fibonacci_with_memoization(n-2)
        self.cache[n] = result
        return result
    
    def fibonacci_space_optimized(self, n):
        """Minimize space usage, accept higher time complexity"""
        if n <= 1:
            return n
        
        prev2, prev1 = 0, 1
        for i in range(2, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
```

## 🎯 Design Decision Framework

### 1. Requirements Analysis
```python
class RequirementsAnalysis:
    
    def analyze_functional_requirements(self, requirements):
        """Analyze what the system should do"""
        return {
            'core_features': self.extract_core_features(requirements),
            'user_flows': self.identify_user_flows(requirements),
            'data_models': self.design_data_models(requirements),
            'apis': self.define_apis(requirements)
        }
    
    def analyze_non_functional_requirements(self, requirements):
        """Analyze how the system should perform"""
        return {
            'scalability': {
                'users': requirements.get('expected_users', 0),
                'qps': requirements.get('queries_per_second', 0),
                'data_size': requirements.get('data_size_gb', 0)
            },
            'availability': {
                'uptime_sla': requirements.get('uptime_percent', 99.9),
                'disaster_recovery': requirements.get('dr_required', False)
            },
            'performance': {
                'latency_ms': requirements.get('max_latency_ms', 100),
                'throughput': requirements.get('min_throughput', 1000)
            },
            'security': {
                'authentication': requirements.get('auth_required', True),
                'encryption': requirements.get('encryption_required', True)
            }
        }
```

### 2. Technology Selection
```python
class TechnologySelection:
    
    def select_database(self, requirements):
        """Select appropriate database based on requirements"""
        if requirements['consistency'] == 'strong' and requirements['data_structure'] == 'relational':
            return {
                'type': 'SQL',
                'recommended': ['PostgreSQL', 'MySQL'],
                'reason': 'ACID properties and relational data model'
            }
        elif requirements['scalability'] == 'high' and requirements['data_structure'] == 'document':
            return {
                'type': 'NoSQL',
                'recommended': ['MongoDB', 'DynamoDB'],
                'reason': 'Horizontal scalability and flexible schema'
            }
        elif requirements['query_type'] == 'key_value' and requirements['performance'] == 'high':
            return {
                'type': 'Key-Value',
                'recommended': ['Redis', 'DynamoDB'],
                'reason': 'Fast key-value operations'
            }
    
    def select_caching_strategy(self, access_patterns):
        """Select caching strategy based on access patterns"""
        if access_patterns['read_heavy'] and access_patterns['data_locality']:
            return {
                'strategy': 'Cache-Aside',
                'technology': 'Redis',
                'reason': 'High read performance with controlled cache updates'
            }
        elif access_patterns['write_heavy']:
            return {
                'strategy': 'Write-Behind',
                'technology': 'Redis + Message Queue',
                'reason': 'Absorb write spikes with async persistence'
            }
```

## 📊 System Design Metrics

### Key Performance Indicators (KPIs)

```python
class SystemMetrics:
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_availability(self, uptime_hours, total_hours):
        """Calculate system availability percentage"""
        availability = (uptime_hours / total_hours) * 100
        return {
            'availability_percent': availability,
            'downtime_hours': total_hours - uptime_hours,
            'sla_compliance': availability >= 99.9
        }
    
    def calculate_throughput(self, requests_processed, time_period_seconds):
        """Calculate system throughput"""
        qps = requests_processed / time_period_seconds
        return {
            'queries_per_second': qps,
            'requests_per_minute': qps * 60,
            'requests_per_hour': qps * 3600
        }
    
    def calculate_latency_percentiles(self, response_times):
        """Calculate latency percentiles"""
        sorted_times = sorted(response_times)
        n = len(sorted_times)
        
        return {
            'p50': sorted_times[int(n * 0.5)],
            'p90': sorted_times[int(n * 0.9)],
            'p95': sorted_times[int(n * 0.95)],
            'p99': sorted_times[int(n * 0.99)],
            'max': max(sorted_times),
            'avg': sum(sorted_times) / n
        }
```

### Capacity Planning
```python
class CapacityPlanning:
    
    def estimate_server_capacity(self, 
                               expected_qps: int,
                               avg_response_time_ms: int,
                               server_cpu_cores: int = 4) -> dict:
        """Estimate server capacity requirements"""
        
        # Rule of thumb: 1 core can handle ~1000 QPS for typical web apps
        cores_needed = expected_qps / 1000
        servers_needed = cores_needed / server_cpu_cores
        
        # Add 50% buffer for peak traffic
        servers_with_buffer = servers_needed * 1.5
        
        return {
            'servers_needed': int(servers_with_buffer) + 1,
            'cores_needed': cores_needed,
            'estimated_cpu_utilization': (cores_needed / server_cpu_cores) * 100,
            'buffer_capacity': servers_with_buffer - servers_needed
        }
    
    def estimate_storage_growth(self, 
                              daily_data_gb: float,
                              retention_days: int = 365,
                              growth_rate_percent: float = 20) -> dict:
        """Estimate storage requirements with growth"""
        
        base_storage = daily_data_gb * retention_days
        yearly_growth = base_storage * (growth_rate_percent / 100)
        
        return {
            'year_1_storage_gb': base_storage,
            'year_2_storage_gb': base_storage + yearly_growth,
            'year_3_storage_gb': base_storage + (yearly_growth * 2),
            'daily_data_gb': daily_data_gb,
            'retention_days': retention_days
        }
```

## 🔄 Common Architecture Patterns

### Microservices Pattern
```python
class MicroservicesPattern:
    
    def __init__(self):
        self.services = {}
        self.service_registry = ServiceRegistry()
        self.load_balancer = LoadBalancer()
    
    def register_service(self, service_name, service_instance):
        """Register a microservice"""
        self.services[service_name] = service_instance
        self.service_registry.register(service_name, service_instance.endpoint)
        return f"Service {service_name} registered"
    
    def call_service(self, service_name, request):
        """Call another microservice"""
        service_endpoint = self.service_registry.discover(service_name)
        if not service_endpoint:
            raise Exception(f"Service {service_name} not found")
        
        # Add circuit breaker, retry logic, etc.
        return self.make_http_call(service_endpoint, request)
    
    def handle_service_failure(self, failed_service):
        """Handle microservice failures gracefully"""
        # Implement fallback mechanisms
        if failed_service == 'recommendation_service':
            return self.get_default_recommendations()
        elif failed_service == 'notification_service':
            return self.queue_notification_for_later()
        else:
            raise Exception(f"No fallback for {failed_service}")
```

### Event-Driven Architecture
```python
import asyncio
from typing import List, Callable

class EventDrivenArchitecture:
    
    def __init__(self):
        self.event_bus = EventBus()
        self.event_handlers = {}
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def publish_event(self, event_type: str, event_data: dict):
        """Publish event to all subscribers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    # Handle asynchronously to avoid blocking
                    asyncio.create_task(handler(event_data))
                except Exception as e:
                    print(f"Error handling event {event_type}: {e}")
    
    def setup_event_sourcing(self):
        """Example event sourcing setup"""
        # User registration events
        self.subscribe('user_registered', self.send_welcome_email)
        self.subscribe('user_registered', self.create_user_profile)
        self.subscribe('user_registered', self.track_analytics)
        
        # Order events
        self.subscribe('order_placed', self.update_inventory)
        self.subscribe('order_placed', self.process_payment)
        self.subscribe('order_placed', self.notify_fulfillment)
```

## 🎯 Best Practices

### 1. Design for Failure
- Assume everything will fail
- Implement graceful degradation
- Use circuit breakers and bulkheads
- Plan for disaster recovery

### 2. Start Simple, Scale Smart
- Begin with monolithic architecture if appropriate
- Extract microservices when needed
- Use proven technologies
- Measure before optimizing

### 3. Data-Driven Decisions
- Monitor key metrics
- Use A/B testing
- Analyze user behavior
- Make decisions based on data

### 4. Security by Design
- Implement defense in depth
- Use principle of least privilege
- Encrypt sensitive data
- Regular security audits

## ✅ Key Takeaways

1. **Trade-offs are inevitable** - Understand and make conscious decisions
2. **Requirements drive design** - Start with clear functional and non-functional requirements
3. **Simplicity first** - Avoid over-engineering early on
4. **Measure everything** - You can't improve what you don't measure
5. **Design for change** - Systems evolve, design should accommodate change

---

**Next**: [Scalability Concepts →](scalability-concepts.md)

**Master the principles, make better design decisions! 🎯**
