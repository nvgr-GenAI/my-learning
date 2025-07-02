# Scalability Patterns & Techniques 📈

Learn how to design systems that can handle growing loads efficiently and reliably.

## 🎯 Learning Objectives

- Understand horizontal vs vertical scaling trade-offs
- Master load balancing techniques and algorithms
- Implement caching strategies effectively
- Design auto-scaling and elastic systems
- Handle database scalability challenges

## 📚 Topics Overview

<div class="grid cards" markdown>

-   :material-arrow-expand-horizontal: **Horizontal Scaling**
    
    ---
    
    Load balancing, distributed processing, stateless services
    
    [Scale out →](horizontal-scaling.md)

-   :material-arrow-expand-vertical: **Vertical Scaling**
    
    ---
    
    Hardware upgrades, resource optimization, performance tuning
    
    [Scale up →](vertical-scaling.md)

-   :material-scale-balance: **Load Balancing**
    
    ---
    
    Algorithms, health checks, session management
    
    [Balance load →](load-balancing.md)

-   :material-cached: **Caching Strategies**
    
    ---
    
    Cache patterns, invalidation, distributed caching
    
    [Cache effectively →](caching.md)

-   :material-database-arrow-up: **Database Scaling**
    
    ---
    
    Sharding, replication, read replicas, partitioning
    
    [Scale databases →](database-scaling.md)

-   :material-auto-fix: **Auto-scaling**
    
    ---
    
    Elastic scaling, metrics-based scaling, predictive scaling
    
    [Auto-scale →](auto-scaling.md)

</div>

## 📊 Scalability Fundamentals

### Scalability Types

| Type | Definition | Implementation | Best Use Case |
|------|------------|----------------|---------------|
| **Horizontal** | Add more servers | Load balancers, distributed systems | Web services, APIs |
| **Vertical** | Upgrade hardware | CPU, RAM, storage upgrades | Databases, single-threaded apps |
| **Functional** | Split by feature | Microservices, service decomposition | Complex applications |
| **Data** | Partition data | Sharding, data distribution | Large datasets |

### Scaling Patterns Overview

```python
class ScalingPatterns:
    """Common scalability patterns and their implementations"""
    
    def __init__(self):
        self.load_balancer = LoadBalancer()
        self.cache_layer = CacheLayer()
        self.database_cluster = DatabaseCluster()
    
    def handle_request(self, request):
        """Demonstrate multi-layer scaling approach"""
        # 1. Load balancing
        server = self.load_balancer.get_server()
        
        # 2. Caching layer
        cached_result = self.cache_layer.get(request.cache_key)
        if cached_result:
            return cached_result
        
        # 3. Database scaling
        if request.type == 'read':
            db = self.database_cluster.get_read_replica()
        else:
            db = self.database_cluster.get_primary()
        
        result = db.execute(request.query)
        
        # 4. Cache the result
        self.cache_layer.set(request.cache_key, result)
        
        return result
```

## 🔄 Load Balancing Algorithms

### Round Robin
```python
class RoundRobinBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0
    
    def get_server(self):
        """Simple round-robin server selection"""
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server
```

### Weighted Round Robin
```python
class WeightedRoundRobinBalancer:
    def __init__(self, servers_with_weights):
        self.servers = []
        self.weights = []
        self.current_weights = []
        
        for server, weight in servers_with_weights:
            self.servers.append(server)
            self.weights.append(weight)
            self.current_weights.append(0)
    
    def get_server(self):
        """Weighted round-robin based on server capacity"""
        total_weight = sum(self.weights)
        
        # Increase current weights
        for i in range(len(self.current_weights)):
            self.current_weights[i] += self.weights[i]
        
        # Find server with highest current weight
        selected_index = self.current_weights.index(max(self.current_weights))
        self.current_weights[selected_index] -= total_weight
        
        return self.servers[selected_index]
```

### Least Connections
```python
class LeastConnectionsBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.connections = {server: 0 for server in servers}
    
    def get_server(self):
        """Select server with least active connections"""
        return min(self.servers, key=lambda s: self.connections[s])
    
    def on_request_start(self, server):
        """Track connection start"""
        self.connections[server] += 1
    
    def on_request_end(self, server):
        """Track connection end"""
        self.connections[server] -= 1
```

### Consistent Hashing
```python
import hashlib
import bisect

class ConsistentHashBalancer:
    def __init__(self, servers, replicas=3):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        
        for server in servers:
            self.add_server(server)
    
    def _hash(self, key):
        """Hash function for consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_server(self, server):
        """Add server to the hash ring"""
        for i in range(self.replicas):
            key = self._hash(f"{server}:{i}")
            self.ring[key] = server
            bisect.insort(self.sorted_keys, key)
    
    def remove_server(self, server):
        """Remove server from the hash ring"""
        for i in range(self.replicas):
            key = self._hash(f"{server}:{i}")
            del self.ring[key]
            self.sorted_keys.remove(key)
    
    def get_server(self, key):
        """Get server for given key using consistent hashing"""
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        
        # Find the first server clockwise from the hash
        idx = bisect.bisect_right(self.sorted_keys, hash_key)
        if idx == len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]
```

## 🚀 Caching Strategies

### Cache-Aside Pattern
```python
class CacheAsidePattern:
    def __init__(self, cache, database):
        self.cache = cache
        self.database = database
    
    def get(self, key):
        """Cache-aside read pattern"""
        # Try cache first
        value = self.cache.get(key)
        if value is not None:
            return value  # Cache hit
        
        # Cache miss - get from database
        value = self.database.get(key)
        if value is not None:
            # Store in cache for next time
            self.cache.set(key, value, ttl=3600)
        
        return value
    
    def set(self, key, value):
        """Cache-aside write pattern"""
        # Write to database first
        self.database.set(key, value)
        
        # Invalidate cache to maintain consistency
        self.cache.delete(key)
```

### Write-Through Cache
```python
class WriteThroughCache:
    def __init__(self, cache, database):
        self.cache = cache
        self.database = database
    
    def get(self, key):
        """Read from cache (always up-to-date)"""
        return self.cache.get(key)
    
    def set(self, key, value):
        """Write-through pattern - update both cache and database"""
        # Write to database first
        self.database.set(key, value)
        
        # Update cache immediately
        self.cache.set(key, value)
```

### Write-Behind (Write-Back) Cache
```python
import asyncio
from collections import deque
import time

class WriteBehindCache:
    def __init__(self, cache, database, batch_size=100, flush_interval=5):
        self.cache = cache
        self.database = database
        self.write_buffer = deque()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Start background flush task
        asyncio.create_task(self._flush_periodically())
    
    def get(self, key):
        """Read from cache"""
        return self.cache.get(key)
    
    def set(self, key, value):
        """Write-behind pattern - cache immediately, database later"""
        # Update cache immediately
        self.cache.set(key, value)
        
        # Queue for database write
        self.write_buffer.append({
            'key': key,
            'value': value,
            'timestamp': time.time()
        })
        
        # Flush if buffer is full
        if len(self.write_buffer) >= self.batch_size:
            asyncio.create_task(self._flush_buffer())
    
    async def _flush_buffer(self):
        """Flush write buffer to database"""
        if not self.write_buffer:
            return
        
        batch = []
        while self.write_buffer and len(batch) < self.batch_size:
            batch.append(self.write_buffer.popleft())
        
        # Batch write to database
        try:
            await self.database.batch_write(batch)
        except Exception as e:
            # Add back to buffer for retry
            self.write_buffer.extendleft(reversed(batch))
            raise e
    
    async def _flush_periodically(self):
        """Periodic flush task"""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self._flush_buffer()
```

## 🗄️ Database Scaling Patterns

### Read Replicas
```python
class DatabaseWithReadReplicas:
    def __init__(self, primary_db, read_replicas):
        self.primary = primary_db
        self.read_replicas = read_replicas
        self.replica_index = 0
    
    def write(self, query):
        """All writes go to primary"""
        return self.primary.execute(query)
    
    def read(self, query):
        """Distribute reads across replicas"""
        replica = self.read_replicas[self.replica_index]
        self.replica_index = (self.replica_index + 1) % len(self.read_replicas)
        return replica.execute(query)
```

### Database Sharding
```python
class DatabaseSharding:
    def __init__(self, shards):
        self.shards = shards
        self.num_shards = len(shards)
    
    def get_shard_key(self, data):
        """Extract shard key from data"""
        # Common strategies: user_id, timestamp, hash of key
        return data.get('user_id') or data.get('id')
    
    def get_shard(self, shard_key):
        """Determine which shard to use"""
        # Hash-based sharding
        shard_index = hash(shard_key) % self.num_shards
        return self.shards[shard_index]
    
    def query(self, query_data):
        """Route query to appropriate shard"""
        shard_key = self.get_shard_key(query_data)
        shard = self.get_shard(shard_key)
        return shard.execute(query_data['query'])
    
    def cross_shard_query(self, query):
        """Query across all shards (expensive operation)"""
        results = []
        for shard in self.shards:
            try:
                result = shard.execute(query)
                results.extend(result)
            except Exception as e:
                print(f"Shard query failed: {e}")
        
        return results
```

## ⚡ Auto-scaling Implementation

### Metrics-Based Auto-scaling
```python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class ScalingMetrics:
    cpu_utilization: float
    memory_utilization: float
    request_rate: float
    response_time: float
    error_rate: float

class AutoScaler:
    def __init__(self, min_instances=1, max_instances=10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        self.scaling_history = []
    
    def should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Determine if scaling up is needed"""
        conditions = [
            metrics.cpu_utilization > 70,
            metrics.memory_utilization > 80,
            metrics.response_time > 1000,  # ms
            metrics.error_rate > 5,  # percent
        ]
        
        # Scale up if any critical threshold is exceeded
        return any(conditions) and self.current_instances < self.max_instances
    
    def should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Determine if scaling down is needed"""
        conditions = [
            metrics.cpu_utilization < 30,
            metrics.memory_utilization < 40,
            metrics.response_time < 200,  # ms
            metrics.error_rate < 1,  # percent
        ]
        
        # Scale down only if all conditions are met
        return all(conditions) and self.current_instances > self.min_instances
    
    def scale(self, metrics: ScalingMetrics):
        """Execute scaling decision"""
        action = None
        
        if self.should_scale_up(metrics):
            self.current_instances += 1
            action = f"Scaled UP to {self.current_instances} instances"
        elif self.should_scale_down(metrics):
            # Add cooldown period to prevent thrashing
            if self._can_scale_down():
                self.current_instances -= 1
                action = f"Scaled DOWN to {self.current_instances} instances"
        
        if action:
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': action,
                'metrics': metrics
            })
        
        return action
    
    def _can_scale_down(self):
        """Check if enough time has passed since last scaling action"""
        if not self.scaling_history:
            return True
        
        last_scaling = self.scaling_history[-1]['timestamp']
        cooldown_period = 300  # 5 minutes
        
        return time.time() - last_scaling > cooldown_period
```

### Predictive Auto-scaling
```python
import numpy as np
from sklearn.linear_model import LinearRegression

class PredictiveAutoScaler:
    def __init__(self, min_instances=1, max_instances=10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        self.historical_data = []
        self.model = LinearRegression()
    
    def collect_metrics(self, timestamp, metrics):
        """Collect historical metrics for prediction"""
        self.historical_data.append({
            'timestamp': timestamp,
            'cpu_utilization': metrics.cpu_utilization,
            'request_rate': metrics.request_rate,
            'instances': self.current_instances
        })
        
        # Keep only recent data (last 24 hours)
        cutoff_time = timestamp - 86400
        self.historical_data = [
            d for d in self.historical_data 
            if d['timestamp'] > cutoff_time
        ]
    
    def predict_load(self, future_minutes=15):
        """Predict future load based on historical patterns"""
        if len(self.historical_data) < 10:
            return None
        
        # Extract features (time-based patterns)
        X = []
        y = []
        
        for data in self.historical_data:
            timestamp = data['timestamp']
            hour_of_day = (timestamp % 86400) / 3600  # Hour of day (0-23)
            day_of_week = (timestamp // 86400) % 7   # Day of week (0-6)
            
            X.append([hour_of_day, day_of_week])
            y.append(data['request_rate'])
        
        # Train model
        self.model.fit(X, y)
        
        # Predict future load
        future_timestamp = time.time() + (future_minutes * 60)
        future_hour = (future_timestamp % 86400) / 3600
        future_day = (future_timestamp // 86400) % 7
        
        predicted_load = self.model.predict([[future_hour, future_day]])[0]
        return predicted_load
    
    def predictive_scale(self, current_metrics):
        """Scale based on predicted future load"""
        predicted_load = self.predict_load()
        
        if predicted_load is None:
            # Fall back to reactive scaling
            return self.reactive_scale(current_metrics)
        
        # Calculate required instances based on predicted load
        # Assume each instance can handle 1000 requests/minute
        required_instances = max(
            self.min_instances,
            min(self.max_instances, int(predicted_load / 1000) + 1)
        )
        
        if required_instances != self.current_instances:
            old_instances = self.current_instances
            self.current_instances = required_instances
            return f"Predictive scaling: {old_instances} -> {required_instances} instances"
        
        return None
```

## 🎯 Scalability Best Practices

### 1. Design for Statelessness
```python
class StatelessService:
    """Stateless service design for better scalability"""
    
    def __init__(self, external_cache, database):
        self.cache = external_cache
        self.database = database
        # No instance state stored
    
    def process_request(self, request):
        """Process request without maintaining state"""
        # Get any required state from external systems
        user_data = self.cache.get(f"user:{request.user_id}")
        if not user_data:
            user_data = self.database.get_user(request.user_id)
            self.cache.set(f"user:{request.user_id}", user_data, ttl=300)
        
        # Process request
        result = self.business_logic(request, user_data)
        
        # Save any state changes to external systems
        if result.state_changed:
            self.database.update_user(request.user_id, result.new_state)
            self.cache.delete(f"user:{request.user_id}")  # Invalidate cache
        
        return result.response
```

### 2. Implement Circuit Breakers
```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60, success_threshold=3):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self):
        """Check if enough time has passed to attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._reset()
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _reset(self):
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
```

### 3. Implement Rate Limiting
```python
import time
from collections import defaultdict, deque

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, max_tokens=100, refill_rate=10):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second
        self.buckets = defaultdict(lambda: {
            'tokens': max_tokens,
            'last_refill': time.time()
        })
    
    def allow_request(self, client_id, tokens_required=1):
        """Check if request is allowed under rate limit"""
        bucket = self.buckets[client_id]
        now = time.time()
        
        # Refill bucket based on time elapsed
        time_elapsed = now - bucket['last_refill']
        tokens_to_add = time_elapsed * self.refill_rate
        bucket['tokens'] = min(
            self.max_tokens, 
            bucket['tokens'] + tokens_to_add
        )
        bucket['last_refill'] = now
        
        # Check if enough tokens available
        if bucket['tokens'] >= tokens_required:
            bucket['tokens'] -= tokens_required
            return True
        
        return False

class SlidingWindowRateLimiter:
    """Sliding window rate limiter"""
    
    def __init__(self, max_requests=100, window_size=60):
        self.max_requests = max_requests
        self.window_size = window_size
        self.request_logs = defaultdict(deque)
    
    def allow_request(self, client_id):
        """Check if request is allowed in sliding window"""
        now = time.time()
        client_log = self.request_logs[client_id]
        
        # Remove old requests outside window
        while client_log and client_log[0] <= now - self.window_size:
            client_log.popleft()
        
        # Check if under limit
        if len(client_log) < self.max_requests:
            client_log.append(now)
            return True
        
        return False
```

## ✅ Scalability Checklist

### Pre-Launch
- [ ] Identify bottlenecks through load testing
- [ ] Implement horizontal scaling strategy
- [ ] Set up monitoring and alerting
- [ ] Design stateless services
- [ ] Implement caching strategy
- [ ] Plan database scaling approach

### Post-Launch
- [ ] Monitor key metrics continuously
- [ ] Set up auto-scaling policies
- [ ] Implement circuit breakers
- [ ] Add rate limiting
- [ ] Plan capacity for growth
- [ ] Regular performance testing

## 🚀 Next Steps

Ready to master specific scalability techniques?

1. **[Horizontal Scaling](horizontal-scaling.md)** - Learn to scale out effectively
2. **[Load Balancing](load-balancing.md)** - Master load distribution
3. **[Caching Strategies](caching.md)** - Implement effective caching
4. **[Database Scaling](database-scaling.md)** - Scale your data layer
5. **[Auto-scaling](auto-scaling.md)** - Build self-scaling systems

---

**Scale smart, scale efficiently! 📈💪**
