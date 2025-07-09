# Scalability Patterns 📈

Understanding how to scale systems from handling hundreds to millions of users is crucial for modern applications. This guide covers scalability principles, patterns, and implementation strategies.

## 🎯 What is Scalability?

Scalability is the ability of a system to handle increased load by adding resources to the system. It's about maintaining performance characteristics as demand grows.

> **Real-World Analogy**: Think of scalability like a highway system. Adding more lanes (horizontal scaling) or making lanes wider (vertical scaling) helps handle more traffic without slowing down.

## 📊 Types of Scalability

### 1. **Horizontal Scaling (Scale Out)**

**Definition**: Adding more machines to handle increased load

**Characteristics**:

- **Linear scaling**: Performance increases with each added machine
- **Fault tolerance**: System continues if individual machines fail
- **Distributed architecture**: Load spread across multiple nodes
- **Complex coordination**: Requires distributed system patterns

**Examples**:

```text
Web Server Scaling:
Single Server → Load Balancer + Multiple Servers
Database Scaling:
Single DB → Master/Replica + Sharding
```

**Benefits**:

- **Unlimited scaling potential**
- **Cost-effective at scale**
- **Fault tolerance**
- **Incremental growth**

**Challenges**:

- **Distributed system complexity**
- **Data consistency issues**
- **Network latency**
- **Coordination overhead**

### 2. **Vertical Scaling (Scale Up)**

**Definition**: Adding more power to existing machines

**Characteristics**:

- **Hardware upgrade**: Better CPU, more RAM, faster storage
- **Centralized architecture**: Single powerful machine
- **Simple deployment**: No architectural changes needed
- **Physical limits**: Maximum hardware capacity

**Examples**:

```text
CPU Scaling:
4 cores → 8 cores → 16 cores
Memory Scaling:
16GB RAM → 64GB RAM → 256GB RAM
```

**Benefits**:

- **Simple to implement**
- **No architectural changes**
- **Strong consistency**
- **Lower complexity**

**Challenges**:

- **Physical limits**
- **Cost increases exponentially**
- **Single point of failure**
- **Downtime for upgrades**

### 3. **Scaling Comparison**

| Aspect | Horizontal | Vertical |
|--------|-----------|----------|
| **Cost** | Linear growth | Exponential growth |
| **Complexity** | High | Low |
| **Fault Tolerance** | High | Low |
| **Scaling Limit** | Practically unlimited | Hardware dependent |
| **Consistency** | Challenging | Easy |
| **Implementation** | Requires redesign | Drop-in upgrade |

## 🏗️ Scalability Patterns

### 1. **Load Balancing**

**Load Balancer Types**:

| Type | OSI Layer | Use Case |
|------|-----------|----------|
| **Layer 4** | Transport | TCP/UDP routing |
| **Layer 7** | Application | HTTP/HTTPS routing |
| **DNS** | Application | Geographic routing |
| **Global** | Application | Multi-region routing |

**Load Balancing Algorithms**:

```text
Round Robin:
Request 1 → Server A
Request 2 → Server B
Request 3 → Server C
Request 4 → Server A (repeat)

Weighted Round Robin:
Server A (weight 3): 3 requests
Server B (weight 2): 2 requests
Server C (weight 1): 1 request

Least Connections:
Route to server with fewest active connections

IP Hash:
hash(client_ip) % server_count
Ensures session affinity
```

### 2. **Database Scaling**

**Read Replicas**:

```text
Architecture:
Master DB (writes) → Replica 1 (reads)
              → Replica 2 (reads)
              → Replica 3 (reads)

Benefits:
- Distributed read load
- Improved read performance
- Geographic distribution
- Backup/disaster recovery
```

**Database Sharding**:

```text
Horizontal Partitioning:
Users 1-1000   → Shard 1
Users 1001-2000 → Shard 2
Users 2001-3000 → Shard 3

Sharding Strategies:
- Range-based: Partition by value range
- Hash-based: Partition by hash function
- Directory-based: Lookup service for routing
```

**Sharding Challenges**:

- **Cross-shard queries**: Expensive joins
- **Rebalancing**: Moving data between shards
- **Hotspots**: Uneven data distribution
- **Complexity**: Application-level routing

### 3. **Caching Strategies**

**Cache Levels**:

```text
Multi-Level Caching:
Browser Cache (100ms) → CDN (200ms) → App Cache (300ms) → DB Cache (400ms)
```

**Cache Patterns**:

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Cache-Aside** | App manages cache | General purpose |
| **Write-Through** | Write to cache + DB | Strong consistency |
| **Write-Behind** | Write to cache, async to DB | High write volume |
| **Refresh-Ahead** | Proactive cache refresh | Predictable access |

**Cache Invalidation**:

```text
Invalidation Strategies:
- TTL (Time To Live): Expire after time
- LRU (Least Recently Used): Evict old data
- LFU (Least Frequently Used): Evict unpopular data
- Manual: Application-controlled invalidation
```

### 4. **Content Delivery Networks (CDN)**

**CDN Architecture**:

```text
CDN Distribution:
Origin Server → Edge Servers → Users
               ↓
    Global Distribution Points
```

**CDN Benefits**:

- **Reduced latency**: Content closer to users
- **Reduced bandwidth**: Offload from origin
- **Improved availability**: Distributed infrastructure
- **DDoS protection**: Distributed attack mitigation

**CDN Configuration**:

```text
CDN Settings:
- Static assets: Long TTL (1 year)
- Dynamic content: Short TTL (5 minutes)
- Cache headers: Control caching behavior
- Purge API: Invalidate cached content
```

### 5. **Microservices Architecture**

**Service Decomposition**:

```text
Monolith → Microservices:
Single App → User Service + Order Service + Payment Service
```

**Microservices Benefits**:

- **Independent scaling**: Scale services separately
- **Technology diversity**: Different tech stacks
- **Team independence**: Parallel development
- **Fault isolation**: Service failures contained

**Service Communication**:

| Pattern | Coupling | Reliability | Use Case |
|---------|----------|-------------|----------|
| **Synchronous** | High | Lower | Real-time queries |
| **Asynchronous** | Low | Higher | Event processing |
| **Message Queues** | Low | High | Decoupled processing |
| **Event Streaming** | Low | High | Real-time data |

## 📈 Scalability Metrics

### 1. **Performance Metrics**

**Key Indicators**:

```text
Scalability Metrics:
- Throughput: Requests per second
- Latency: Response time percentiles
- Utilization: Resource usage percentage
- Efficiency: Performance per resource unit
```

**Scalability Testing**:

```text
Load Testing Types:
- Baseline: Normal load performance
- Load: Expected peak performance
- Stress: Breaking point identification
- Spike: Sudden load handling
- Volume: Large dataset performance
```

### 2. **Capacity Planning**

**Growth Modeling**:

```text
Capacity Planning:
Current Load → Growth Rate → Future Capacity
100 RPS → 50% monthly growth → 1000 RPS in 6 months
```

**Resource Estimation**:

```text
Resource Calculation:
- CPU: Peak load × safety factor (1.5-2x)
- Memory: Working set × growth factor
- Storage: Data size × replication factor
- Network: Throughput × peak multiplier
```

### 3. **Monitoring and Alerting**

**Scalability Monitoring**:

```text
Monitoring Stack:
- Application metrics: Response time, error rate
- Infrastructure metrics: CPU, memory, disk
- Business metrics: User growth, revenue
- External metrics: Third-party services
```

**Alert Thresholds**:

| Metric | Warning | Critical |
|--------|---------|----------|
| **CPU Usage** | > 70% | > 85% |
| **Memory Usage** | > 80% | > 90% |
| **Response Time** | > 500ms | > 1000ms |
| **Error Rate** | > 1% | > 5% |

## 🔧 Implementation Strategies

### 1. **Gradual Scaling**

**Scaling Phases**:

```text
Phase 1: Single Server
- Simple deployment
- Fast development
- Low complexity

Phase 2: Horizontal Web Layer
- Load balancer
- Multiple web servers
- Shared database

Phase 3: Database Scaling
- Read replicas
- Connection pooling
- Query optimization

Phase 4: Caching Layer
- Application cache
- Database cache
- CDN integration

Phase 5: Microservices
- Service decomposition
- API gateway
- Service mesh
```

### 2. **Auto-Scaling**

**Auto-Scaling Patterns**:

```text
Horizontal Pod Autoscaler:
- Metric: CPU utilization
- Target: 70% average
- Min replicas: 2
- Max replicas: 10
- Scale up: Add pod when > 70%
- Scale down: Remove pod when < 50%
```

**Auto-Scaling Configuration**:

```text
Scaling Rules:
- Scale-up trigger: CPU > 70% for 2 minutes
- Scale-down trigger: CPU < 30% for 5 minutes
- Cooldown period: 5 minutes between actions
- Maximum instances: 20
- Minimum instances: 2
```

### 3. **Data Partitioning**

**Partitioning Strategies**:

| Strategy | Method | Benefits | Challenges |
|----------|--------|----------|------------|
| **Vertical** | Split by columns | Reduced I/O | Complex joins |
| **Horizontal** | Split by rows | Parallel processing | Cross-partition queries |
| **Functional** | Split by feature | Service isolation | Data consistency |
| **Hybrid** | Combination | Optimized for use case | High complexity |

**Partitioning Examples**:

```text
User Data Partitioning:
- Shard key: user_id
- Partition function: hash(user_id) % num_shards
- Routing: Application-level routing
- Rebalancing: Consistent hashing
```

## 🚀 Advanced Scaling Techniques

### 1. **Event-Driven Architecture**

**Event Streaming**:

```text
Event-Driven Scaling:
Producer → Event Stream → Consumer Groups
         ↓
    Automatic Scaling
```

**Stream Processing**:

```text
Processing Patterns:
- Batch processing: Process in chunks
- Stream processing: Real-time processing
- Micro-batch: Small batch processing
- Lambda architecture: Batch + stream
```

### 2. **CQRS (Command Query Responsibility Segregation)**

**CQRS Pattern**:

```text
Traditional:
Application → Single Database

CQRS:
Commands → Write Database
Queries → Read Database (optimized)
```

**CQRS Benefits**:

- **Optimized read/write models**
- **Independent scaling**
- **Better performance**
- **Flexible data models**

### 3. **Event Sourcing**

**Event Store**:

```text
Event Sourcing:
Command → Event → Event Store → Projections
                              ↓
                        Read Models
```

**Scaling Benefits**:

- **Append-only storage**: High write performance
- **Replay capability**: Rebuild read models
- **Audit trail**: Complete history
- **Eventual consistency**: Flexible consistency

## 💡 Best Practices

### 1. **Design for Scale**

**Scalability Principles**:

```text
Scale-First Design:
1. Stateless components
2. Horizontal scaling ready
3. Asynchronous processing
4. Loose coupling
5. Failure isolation
```

### 2. **Performance Optimization**

**Optimization Strategy**:

```text
Performance Optimization:
1. Measure first
2. Optimize algorithms
3. Add caching
4. Scale infrastructure
5. Monitor continuously
```

### 3. **Capacity Planning**

**Planning Process**:

```text
Capacity Planning:
1. Analyze current usage
2. Forecast growth
3. Model resource needs
4. Plan scaling timeline
5. Monitor and adjust
```

## 🔧 Common Scaling Challenges

### 1. **State Management**

**Stateful vs. Stateless**:

```text
Stateful Scaling Issues:
- Session affinity required
- Difficult to scale
- Single point of failure

Stateless Solutions:
- External session store
- JWT tokens
- Database sessions
```

### 2. **Data Consistency**

**Consistency Challenges**:

```text
Distributed Data Issues:
- Split-brain scenarios
- Network partitions
- Eventual consistency
- Conflict resolution
```

### 3. **Operational Complexity**

**Operational Challenges**:

```text
Scaling Complexity:
- Monitoring overhead
- Deployment complexity
- Configuration management
- Incident response
```

## 🧮 Scalability Algorithms

### 1. **Bloom Filters**

**What is a Bloom Filter?**

A space-efficient probabilistic data structure that tells you if an element is "definitely not in the set" or "possibly in the set". Perfect for scalable systems where memory is constrained.

**Use Cases**:
- **Cache optimization**: Check if data exists before expensive lookup
- **Database query optimization**: Avoid unnecessary disk reads
- **Duplicate detection**: Web crawling, distributed systems
- **Content filtering**: Spam detection, malicious URL filtering

**Implementation**:

```python
import hashlib
import math
from typing import Union

class BloomFilter:
    def __init__(self, expected_items: int, false_positive_rate: float = 0.1):
        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        
        # Calculate optimal bit array size and hash functions
        self.bit_array_size = self._calculate_bit_array_size()
        self.hash_count = self._calculate_hash_count()
        
        # Initialize bit array
        self.bit_array = [0] * self.bit_array_size
        self.items_added = 0
    
    def _calculate_bit_array_size(self) -> int:
        """Calculate optimal bit array size"""
        m = -(self.expected_items * math.log(self.false_positive_rate)) / (math.log(2) ** 2)
        return int(m)
    
    def _calculate_hash_count(self) -> int:
        """Calculate optimal number of hash functions"""
        k = (self.bit_array_size / self.expected_items) * math.log(2)
        return int(k)
    
    def _get_hash_values(self, item: str) -> list:
        """Generate multiple hash values for an item"""
        hashes = []
        
        # Use different hash functions
        hash1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        hash2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        
        for i in range(self.hash_count):
            # Double hashing technique
            hash_value = (hash1 + i * hash2) % self.bit_array_size
            hashes.append(hash_value)
        
        return hashes
    
    def add(self, item: str):
        """Add item to bloom filter"""
        hash_values = self._get_hash_values(item)
        
        for hash_value in hash_values:
            self.bit_array[hash_value] = 1
        
        self.items_added += 1
    
    def contains(self, item: str) -> bool:
        """Check if item might be in the set"""
        hash_values = self._get_hash_values(item)
        
        for hash_value in hash_values:
            if self.bit_array[hash_value] == 0:
                return False  # Definitely not in set
        
        return True  # Possibly in set
    
    def current_false_positive_rate(self) -> float:
        """Calculate current false positive rate"""
        if self.items_added == 0:
            return 0.0
        
        filled_bits = sum(self.bit_array)
        probability = (filled_bits / self.bit_array_size) ** self.hash_count
        return probability

# Example usage for cache optimization
class BloomCacheOptimizer:
    def __init__(self, cache_size: int = 10000):
        self.bloom_filter = BloomFilter(cache_size, 0.01)  # 1% false positive
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.bloom_saves = 0
    
    def get(self, key: str):
        """Get item with bloom filter optimization"""
        # First check bloom filter
        if not self.bloom_filter.contains(key):
            # Definitely not in cache
            self.bloom_saves += 1
            return self._fetch_from_database(key)
        
        # Might be in cache, check actual cache
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        else:
            self.cache_misses += 1
            return self._fetch_from_database(key)
    
    def put(self, key: str, value):
        """Put item in cache and bloom filter"""
        self.cache[key] = value
        self.bloom_filter.add(key)
    
    def _fetch_from_database(self, key: str):
        """Simulate expensive database operation"""
        # In real implementation, this would query database
        return f"data_for_{key}"
    
    def get_stats(self):
        """Get performance statistics"""
        total_requests = self.cache_hits + self.cache_misses + self.bloom_saves
        if total_requests == 0:
            return {"cache_hit_rate": 0, "bloom_save_rate": 0}
        
        return {
            "cache_hit_rate": self.cache_hits / total_requests,
            "bloom_save_rate": self.bloom_saves / total_requests,
            "false_positive_rate": self.bloom_filter.current_false_positive_rate()
        }
```

### 2. **Rate Limiting Algorithms**

**Token Bucket Algorithm**:

```python
import time
import threading
from typing import Optional

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens"""
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Refill tokens based on time passed"""
        now = time.time()
        time_passed = now - self.last_refill
        
        # Add tokens based on time passed
        tokens_to_add = time_passed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_available_tokens(self) -> int:
        """Get current available tokens"""
        with self.lock:
            self._refill()
            return int(self.tokens)

class SlidingWindowLog:
    def __init__(self, limit: int, window_size: int):
        """
        Args:
            limit: Maximum requests in window
            window_size: Window size in seconds
        """
        self.limit = limit
        self.window_size = window_size
        self.requests = []
        self.lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside window
            cutoff_time = now - self.window_size
            self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]
            
            # Check if under limit
            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True
            
            return False
    
    def get_remaining_requests(self) -> int:
        """Get remaining requests in current window"""
        with self.lock:
            now = time.time()
            cutoff_time = now - self.window_size
            current_requests = [req_time for req_time in self.requests if req_time > cutoff_time]
            return max(0, self.limit - len(current_requests))

class SlidingWindowCounter:
    def __init__(self, limit: int, window_size: int, num_buckets: int = 10):
        """
        Args:
            limit: Maximum requests in window
            window_size: Window size in seconds
            num_buckets: Number of sub-windows for smoothing
        """
        self.limit = limit
        self.window_size = window_size
        self.num_buckets = num_buckets
        self.bucket_size = window_size / num_buckets
        
        self.buckets = [0] * num_buckets
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            self._update_buckets()
            
            current_requests = sum(self.buckets)
            if current_requests < self.limit:
                current_bucket = self._get_current_bucket()
                self.buckets[current_bucket] += 1
                return True
            
            return False
    
    def _update_buckets(self):
        """Update buckets based on time passed"""
        now = time.time()
        time_passed = now - self.last_update
        
        # Calculate how many buckets to clear
        buckets_to_clear = int(time_passed / self.bucket_size)
        
        if buckets_to_clear > 0:
            # Shift buckets
            if buckets_to_clear >= self.num_buckets:
                self.buckets = [0] * self.num_buckets
            else:
                # Rotate buckets
                self.buckets = self.buckets[buckets_to_clear:] + [0] * buckets_to_clear
            
            self.last_update = now
    
    def _get_current_bucket(self) -> int:
        """Get current bucket index"""
        return int((time.time() / self.bucket_size) % self.num_buckets)

# Distributed rate limiter using Redis
import redis
import json

class DistributedRateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def token_bucket_check(self, key: str, capacity: int, refill_rate: float, tokens: int = 1) -> bool:
        """Distributed token bucket using Redis Lua script"""
        script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local requested_tokens = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now
        
        -- Refill tokens
        local time_passed = now - last_refill
        local tokens_to_add = time_passed * refill_rate
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        -- Check if we can consume
        if tokens >= requested_tokens then
            tokens = tokens - requested_tokens
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, capacity / refill_rate * 2)
            return 1
        else
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, capacity / refill_rate * 2)
            return 0
        end
        """
        
        result = self.redis.eval(script, 1, key, capacity, refill_rate, tokens, time.time())
        return result == 1
    
    def sliding_window_check(self, key: str, limit: int, window_size: int) -> bool:
        """Distributed sliding window using Redis sorted sets"""
        now = time.time()
        cutoff = now - window_size
        
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, cutoff)
        
        # Count current entries
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiration
        pipe.expire(key, window_size)
        
        results = pipe.execute()
        current_count = results[1]
        
        return current_count < limit
```

### 3. **Consistent Hashing (Extended)**

**Advanced Consistent Hashing with Virtual Nodes**:

```python
import hashlib
import bisect
from typing import Dict, List, Any, Optional

class ConsistentHashRing:
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes: set = set()
    
    def _hash(self, key: str) -> int:
        """Generate hash for a key"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: str):
        """Add a node to the ring"""
        if node in self.nodes:
            return
        
        self.nodes.add(node)
        
        # Add virtual nodes
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            
            self.ring[hash_value] = node
            bisect.insort(self.sorted_keys, hash_value)
    
    def remove_node(self, node: str):
        """Remove a node from the ring"""
        if node not in self.nodes:
            return
        
        self.nodes.discard(node)
        
        # Remove virtual nodes
        keys_to_remove = []
        for hash_value, ring_node in self.ring.items():
            if ring_node == node:
                keys_to_remove.append(hash_value)
        
        for key in keys_to_remove:
            del self.ring[key]
            self.sorted_keys.remove(key)
    
    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a key"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find the first node clockwise
        idx = bisect.bisect_right(self.sorted_keys, hash_value)
        
        # Wrap around if necessary
        if idx == len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]
    
    def get_nodes(self, key: str, count: int) -> List[str]:
        """Get multiple nodes for replication"""
        if not self.ring or count <= 0:
            return []
        
        hash_value = self._hash(key)
        nodes = []
        unique_nodes = set()
        
        # Find starting position
        idx = bisect.bisect_right(self.sorted_keys, hash_value)
        
        # Collect unique nodes
        while len(unique_nodes) < count and len(unique_nodes) < len(self.nodes):
            if idx >= len(self.sorted_keys):
                idx = 0
            
            node = self.ring[self.sorted_keys[idx]]
            if node not in unique_nodes:
                unique_nodes.add(node)
                nodes.append(node)
            
            idx += 1
        
        return nodes
    
    def get_load_distribution(self, keys: List[str]) -> Dict[str, int]:
        """Analyze load distribution across nodes"""
        distribution = {node: 0 for node in self.nodes}
        
        for key in keys:
            node = self.get_node(key)
            if node:
                distribution[node] += 1
        
        return distribution

# Jump Consistent Hash (Google's algorithm)
def jump_consistent_hash(key: int, num_buckets: int) -> int:
    """
    Google's Jump Consistent Hash algorithm
    Very fast with O(ln(num_buckets)) time complexity
    """
    b, j = -1, 0
    
    while j < num_buckets:
        b = j
        key = ((key * 2862933555777941757) + 1) & 0xffffffffffffffff
        j = int((b + 1) * (1 << 31) / ((key >> 33) + 1))
    
    return b

class JumpHashRouter:
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.num_nodes = len(nodes)
    
    def get_node(self, key: str) -> str:
        """Get node using jump consistent hash"""
        key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
        bucket = jump_consistent_hash(key_hash, self.num_nodes)
        return self.nodes[bucket]
    
    def add_node(self, node: str):
        """Add new node (requires rehashing)"""
        self.nodes.append(node)
        self.num_nodes += 1
    
    def remove_node(self, node: str):
        """Remove node (requires rehashing)"""
        if node in self.nodes:
            self.nodes.remove(node)
            self.num_nodes -= 1
```

### 4. **Probabilistic Data Structures**

**Count-Min Sketch (for frequency estimation)**:

```python
import hashlib
import numpy as np
from typing import List

class CountMinSketch:
    def __init__(self, width: int, depth: int):
        """
        Args:
            width: Number of buckets per hash function
            depth: Number of hash functions
        """
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=int)
        self.hash_functions = self._generate_hash_functions()
    
    def _generate_hash_functions(self) -> List[callable]:
        """Generate independent hash functions"""
        functions = []
        
        for i in range(self.depth):
            def hash_func(key: str, seed: int = i) -> int:
                hash_obj = hashlib.md5(f"{key}:{seed}".encode())
                return int(hash_obj.hexdigest(), 16) % self.width
            
            functions.append(hash_func)
        
        return functions
    
    def add(self, item: str, count: int = 1):
        """Add item to sketch"""
        for i, hash_func in enumerate(self.hash_functions):
            bucket = hash_func(item)
            self.table[i][bucket] += count
    
    def estimate(self, item: str) -> int:
        """Estimate frequency of item"""
        estimates = []
        
        for i, hash_func in enumerate(self.hash_functions):
            bucket = hash_func(item)
            estimates.append(self.table[i][bucket])
        
        # Return minimum estimate (to reduce overestimation)
        return min(estimates)
    
    def get_heavy_hitters(self, threshold: int) -> List[tuple]:
        """Get items that appear more than threshold times"""
        # This is simplified - in practice you'd need additional tracking
        pass

# HyperLogLog for cardinality estimation
class HyperLogLog:
    def __init__(self, precision: int = 10):
        """
        Args:
            precision: Number of bits for bucket selection (4-16)
        """
        self.precision = precision
        self.num_buckets = 2 ** precision
        self.buckets = [0] * self.num_buckets
        self.alpha = self._get_alpha()
    
    def _get_alpha(self) -> float:
        """Get alpha constant for bias correction"""
        if self.num_buckets >= 128:
            return 0.7213 / (1 + 1.079 / self.num_buckets)
        elif self.num_buckets >= 64:
            return 0.709
        elif self.num_buckets >= 32:
            return 0.697
        else:
            return 0.673
    
    def add(self, item: str):
        """Add item to HyperLogLog"""
        # Hash the item
        hash_value = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        
        # Use first 'precision' bits for bucket selection
        bucket = hash_value & ((1 << self.precision) - 1)
        
        # Use remaining bits to count leading zeros
        remaining_bits = hash_value >> self.precision
        leading_zeros = self._count_leading_zeros(remaining_bits) + 1
        
        # Update bucket with maximum leading zeros seen
        self.buckets[bucket] = max(self.buckets[bucket], leading_zeros)
    
    def _count_leading_zeros(self, value: int) -> int:
        """Count leading zeros in binary representation"""
        if value == 0:
            return 32  # Assuming 32-bit hash
        
        count = 0
        while (value & 0x80000000) == 0 and count < 32:
            count += 1
            value <<= 1
        
        return count
    
    def cardinality(self) -> int:
        """Estimate cardinality"""
        # Raw estimate
        raw_estimate = self.alpha * (self.num_buckets ** 2) / sum(2 ** (-x) for x in self.buckets)
        
        # Apply small range correction
        if raw_estimate <= 2.5 * self.num_buckets:
            zeros = self.buckets.count(0)
            if zeros != 0:
                return self.num_buckets * math.log(self.num_buckets / zeros)
        
        # Apply large range correction
        if raw_estimate <= (1.0/30.0) * (2 ** 32):
            return raw_estimate
        else:
            return -2 ** 32 * math.log(1 - raw_estimate / (2 ** 32))
        
        return int(raw_estimate)
```

### 5. **Load Balancing Algorithms**

**Advanced Load Balancing with Health Checks**:

```python
import time
import random
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class ServerHealth(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class Server:
    id: str
    host: str
    port: int
    weight: int = 1
    health: ServerHealth = ServerHealth.UNKNOWN
    current_connections: int = 0
    response_time: float = 0.0
    last_health_check: float = 0.0
    failure_count: int = 0

class LoadBalancer:
    def __init__(self, health_check_interval: int = 30):
        self.servers: List[Server] = []
        self.health_check_interval = health_check_interval
        self.lock = threading.Lock()
        self.round_robin_index = 0
        
        # Start health checking
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
    
    def add_server(self, server: Server):
        """Add server to load balancer"""
        with self.lock:
            self.servers.append(server)
    
    def remove_server(self, server_id: str):
        """Remove server from load balancer"""
        with self.lock:
            self.servers = [s for s in self.servers if s.id != server_id]
    
    def get_healthy_servers(self) -> List[Server]:
        """Get list of healthy servers"""
        return [s for s in self.servers if s.health == ServerHealth.HEALTHY]
    
    def round_robin(self) -> Optional[Server]:
        """Round robin load balancing"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        with self.lock:
            server = healthy_servers[self.round_robin_index % len(healthy_servers)]
            self.round_robin_index += 1
            return server
    
    def weighted_round_robin(self) -> Optional[Server]:
        """Weighted round robin load balancing"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        # Create weighted list
        weighted_servers = []
        for server in healthy_servers:
            weighted_servers.extend([server] * server.weight)
        
        with self.lock:
            server = weighted_servers[self.round_robin_index % len(weighted_servers)]
            self.round_robin_index += 1
            return server
    
    def least_connections(self) -> Optional[Server]:
        """Least connections load balancing"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        return min(healthy_servers, key=lambda s: s.current_connections)
    
    def weighted_least_connections(self) -> Optional[Server]:
        """Weighted least connections load balancing"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        # Calculate weighted connections
        def weighted_connections(server: Server) -> float:
            if server.weight == 0:
                return float('inf')
            return server.current_connections / server.weight
        
        return min(healthy_servers, key=weighted_connections)
    
    def response_time_weighted(self) -> Optional[Server]:
        """Response time weighted load balancing"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        # Select based on inverse of response time
        weights = []
        for server in healthy_servers:
            # Avoid division by zero
            weight = 1.0 / (server.response_time + 0.001)
            weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for i, server in enumerate(healthy_servers):
            cumulative_weight += weights[i]
            if r <= cumulative_weight:
                return server
        
        return healthy_servers[-1]
    
    def consistent_hash_routing(self, key: str) -> Optional[Server]:
        """Consistent hash-based routing"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        # Simple hash-based selection
        hash_value = hash(key) % len(healthy_servers)
        return healthy_servers[hash_value]
    
    def power_of_two_choices(self) -> Optional[Server]:
        """Power of two choices algorithm"""
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
        
        if len(healthy_servers) == 1:
            return healthy_servers[0]
        
        # Pick two random servers and choose the one with fewer connections
        server1, server2 = random.sample(healthy_servers, 2)
        
        if server1.current_connections <= server2.current_connections:
            return server1
        else:
            return server2
    
    def _health_check_loop(self):
        """Background health checking"""
        while True:
            self._perform_health_checks()
            time.sleep(self.health_check_interval)
    
    def _perform_health_checks(self):
        """Perform health checks on all servers"""
        for server in self.servers:
            try:
                # Simulate health check
                is_healthy = self._check_server_health(server)
                
                if is_healthy:
                    server.health = ServerHealth.HEALTHY
                    server.failure_count = 0
                else:
                    server.failure_count += 1
                    if server.failure_count >= 3:  # Mark unhealthy after 3 failures
                        server.health = ServerHealth.UNHEALTHY
                
                server.last_health_check = time.time()
                
            except Exception:
                server.failure_count += 1
                if server.failure_count >= 3:
                    server.health = ServerHealth.UNHEALTHY
    
    def _check_server_health(self, server: Server) -> bool:
        """Check if server is healthy"""
        # Simulate health check - in practice this would be an HTTP request
        return random.random() > 0.1  # 90% success rate
    
    def update_server_metrics(self, server_id: str, connections: int, response_time: float):
        """Update server metrics"""
        for server in self.servers:
            if server.id == server_id:
                server.current_connections = connections
                server.response_time = response_time
                break
```

### 6. **Geo-Distributed Routing**

```python
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class GeoLocation:
    latitude: float
    longitude: float

@dataclass
class GeoServer:
    id: str
    location: GeoLocation
    capacity: int
    current_load: int = 0

class GeoLoadBalancer:
    def __init__(self):
        self.servers: List[GeoServer] = []
    
    def add_server(self, server: GeoServer):
        """Add geo-distributed server"""
        self.servers.append(server)
    
    def calculate_distance(self, loc1: GeoLocation, loc2: GeoLocation) -> float:
        """Calculate distance between two locations using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(loc1.latitude)
        lat2_rad = math.radians(loc2.latitude)
        delta_lat = math.radians(loc2.latitude - loc1.latitude)
        delta_lon = math.radians(loc2.longitude - loc1.longitude)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        
        return distance
    
    def get_nearest_server(self, client_location: GeoLocation) -> GeoServer:
        """Get nearest available server"""
        available_servers = [s for s in self.servers if s.current_load < s.capacity]
        
        if not available_servers:
            return None
        
        return min(available_servers, 
                  key=lambda s: self.calculate_distance(client_location, s.location))
    
    def get_optimal_server(self, client_location: GeoLocation) -> GeoServer:
        """Get server balancing distance and load"""
        available_servers = [s for s in self.servers if s.current_load < s.capacity]
        
        if not available_servers:
            return None
        
        def score_server(server: GeoServer) -> float:
            distance = self.calculate_distance(client_location, server.location)
            load_factor = server.current_load / server.capacity
            
            # Combine distance and load (lower is better)
            return distance * (1 + load_factor)
        
        return min(available_servers, key=score_server)
```

**Real-World Usage Examples**:

```python
# Example: Using algorithms in a web service
class ScalableWebService:
    def __init__(self):
        # Initialize components
        self.cache_optimizer = BloomCacheOptimizer(10000)
        self.rate_limiter = TokenBucket(100, 10)  # 100 requests, refill 10/sec
        self.hash_ring = ConsistentHashRing(150)
        self.load_balancer = LoadBalancer()
        self.hyperloglog = HyperLogLog(12)  # Track unique visitors
        
        # Add servers to load balancer
        servers = [
            Server("web1", "10.0.1.1", 8080, weight=3),
            Server("web2", "10.0.1.2", 8080, weight=2),
            Server("web3", "10.0.1.3", 8080, weight=1),
        ]
        
        for server in servers:
            self.load_balancer.add_server(server)
            self.hash_ring.add_node(server.id)
    
    def handle_request(self, user_id: str, request_data: dict):
        """Handle incoming request with all algorithms"""
        
        # Rate limiting
        if not self.rate_limiter.consume():
            return {"error": "Rate limit exceeded"}, 429
        
        # Track unique users
        self.hyperloglog.add(user_id)
        
        # Get data with bloom filter optimization
        data = self.cache_optimizer.get(f"user:{user_id}")
        
        # Route to appropriate server
        server = self.load_balancer.weighted_least_connections()
        if not server:
            return {"error": "No servers available"}, 503
        
        # Process request
        response = self.process_request(server, request_data)
        
        return response
    
    def process_request(self, server: Server, request_data: dict):
        """Simulate request processing"""
        # Simulate processing time
        import time
        start_time = time.time()
        time.sleep(0.1)  # Simulate work
        end_time = time.time()
        
        # Update server metrics
        response_time = end_time - start_time
        self.load_balancer.update_server_metrics(
            server.id, 
            server.current_connections + 1, 
            response_time
        )
        
        return {"server": server.id, "data": "processed"}
    
    def get_analytics(self):
        """Get system analytics"""
        return {
            "unique_visitors": self.hyperloglog.cardinality(),
            "cache_stats": self.cache_optimizer.get_stats(),
            "available_tokens": self.rate_limiter.get_available_tokens(),
            "server_distribution": self.hash_ring.get_load_distribution([])
        }
```
