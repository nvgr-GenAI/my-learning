# Database Scaling Patterns 📈

Advanced patterns for scaling databases from thousands to millions of operations per second. This guide covers horizontal scaling, partitioning strategies, and distributed database patterns.

## 🎯 Database Scaling Fundamentals

### Why Scale Databases?

Traditional single-node databases hit limits as applications grow:

**Performance Limits**:
- **CPU bottlenecks**: Query processing capacity
- **Memory constraints**: Working set larger than RAM  
- **I/O bottlenecks**: Disk bandwidth limitations
- **Connection limits**: Maximum concurrent connections

**Capacity Limits**:
- **Storage capacity**: Single disk size limits
- **Network bandwidth**: Single node network capacity
- **Geographic distribution**: Latency for distant users

> **Real-World Example**: A social media platform might start with a single MySQL server handling 1,000 users. As it grows to 1 million users, the database becomes the bottleneck, requiring scaling strategies.

## 🔄 Read Scaling Patterns

### 1. **Read Replicas**

**Pattern**: Create read-only copies of the primary database

**Architecture**:
```text
Write Path:
Application → Primary Database → Replicas (async)

Read Path:
Application → Load Balancer → Read Replicas
```

**Implementation Considerations**:

**Replication Types**:
- **Synchronous**: Replicas updated before write confirmation
- **Asynchronous**: Replicas updated after write confirmation
- **Semi-synchronous**: At least one replica must acknowledge

**Read Replica Strategies**:
```python
class DatabaseRouter:
    def __init__(self):
        self.primary = connect_to_primary()
        self.replicas = [connect_to_replica(i) for i in range(3)]
        self.replica_index = 0
    
    def write(self, query, params):
        # All writes go to primary
        return self.primary.execute(query, params)
    
    def read(self, query, params, consistency='eventual'):
        if consistency == 'strong':
            # Read from primary for strong consistency
            return self.primary.execute(query, params)
        else:
            # Read from replica for eventual consistency
            replica = self.get_next_replica()
            return replica.execute(query, params)
    
    def get_next_replica(self):
        # Round-robin load balancing
        replica = self.replicas[self.replica_index]
        self.replica_index = (self.replica_index + 1) % len(self.replicas)
        return replica
```

**Benefits**:
- **Improved read performance**: Distribute read load
- **Geographic distribution**: Place replicas closer to users
- **High availability**: Failover options if primary fails
- **Backup**: Real-time backup through replication

**Challenges**:
- **Replication lag**: Eventual consistency issues
- **Complexity**: Application must handle read/write routing
- **Consistency**: Read-after-write consistency problems
- **Cost**: Additional infrastructure and maintenance

**Use Cases**:
- **Read-heavy workloads**: 80%+ read operations
- **Geographic distribution**: Global applications
- **Analytics**: Separate analytical queries from transactional
- **Reporting**: Offload reporting to replicas

### 2. **Connection Pooling**

**Pattern**: Share database connections across application instances

**Implementation**:
```python
import threading
from queue import Queue
import time

class ConnectionPool:
    def __init__(self, min_size=5, max_size=20, host='localhost'):
        self.min_size = min_size
        self.max_size = max_size
        self.host = host
        self.pool = Queue(maxsize=max_size)
        self.created_connections = 0
        self.lock = threading.Lock()
        
        # Initialize minimum connections
        for _ in range(min_size):
            self.pool.put(self.create_connection())
    
    def create_connection(self):
        with self.lock:
            self.created_connections += 1
        return connect_to_database(self.host)
    
    def get_connection(self, timeout=10):
        try:
            # Try to get existing connection
            return self.pool.get(timeout=timeout)
        except:
            # Create new connection if under limit
            with self.lock:
                if self.created_connections < self.max_size:
                    return self.create_connection()
            raise Exception("Connection pool exhausted")
    
    def return_connection(self, conn):
        if conn.is_healthy():
            self.pool.put(conn)
        else:
            # Connection is bad, create a new one
            with self.lock:
                self.created_connections -= 1
            new_conn = self.create_connection()
            self.pool.put(new_conn)
```

**Benefits**:
- **Reduced overhead**: Reuse expensive connections
- **Controlled resource usage**: Limit database connections
- **Better performance**: No connection establishment delay
- **Resource management**: Prevent connection exhaustion

## 🗂️ Write Scaling Patterns

### 1. **Database Sharding (Horizontal Partitioning)**

**Pattern**: Split data across multiple database instances

**Sharding Strategies**:

**Range-Based Sharding**:
```text
User ID Ranges:
Shard 1: users 1-100,000
Shard 2: users 100,001-200,000  
Shard 3: users 200,001-300,000
```

```python
class RangeShardRouter:
    def __init__(self):
        self.shards = {
            (1, 100000): connect_to_shard('shard1'),
            (100001, 200000): connect_to_shard('shard2'),
            (200001, 300000): connect_to_shard('shard3')
        }
    
    def get_shard(self, user_id):
        for (start, end), shard in self.shards.items():
            if start <= user_id <= end:
                return shard
        raise Exception(f"No shard found for user_id: {user_id}")
```

**Hash-Based Sharding**:
```text
Hash Function:
shard_id = hash(user_id) % num_shards
```

```python
class HashShardRouter:
    def __init__(self, num_shards=4):
        self.shards = [connect_to_shard(f'shard{i}') 
                      for i in range(num_shards)]
        self.num_shards = num_shards
    
    def get_shard(self, user_id):
        shard_index = hash(str(user_id)) % self.num_shards
        return self.shards[shard_index]
    
    def execute_query(self, user_id, query, params):
        shard = self.get_shard(user_id)
        return shard.execute(query, params)
```

**Directory-Based Sharding**:
```python
class DirectoryShardRouter:
    def __init__(self):
        self.directory = redis.Redis()  # Shard lookup service
        self.shards = {
            'shard1': connect_to_shard('shard1'),
            'shard2': connect_to_shard('shard2'),
            'shard3': connect_to_shard('shard3')
        }
    
    def get_shard(self, user_id):
        shard_name = self.directory.get(f"user:{user_id}:shard")
        if not shard_name:
            # Assign new user to least loaded shard
            shard_name = self.assign_to_shard(user_id)
        return self.shards[shard_name.decode()]
    
    def assign_to_shard(self, user_id):
        # Simple round-robin assignment
        shard_counts = {name: self.directory.get(f"shard:{name}:count") or 0 
                       for name in self.shards.keys()}
        least_loaded = min(shard_counts, key=shard_counts.get)
        
        self.directory.set(f"user:{user_id}:shard", least_loaded)
        self.directory.incr(f"shard:{least_loaded}:count")
        return least_loaded
```

**Sharding Benefits**:
- **Linear scaling**: Add shards to increase capacity
- **Parallel processing**: Queries run across multiple databases
- **Fault isolation**: Failure affects only one shard
- **Cost distribution**: Spread cost across multiple smaller instances

**Sharding Challenges**:
- **Cross-shard queries**: Complex joins across shards
- **Rebalancing**: Moving data when adding/removing shards
- **Hotspots**: Uneven data distribution
- **Transactions**: Cross-shard transactions are complex

### 2. **Vertical Partitioning**

**Pattern**: Split tables by columns or functionality

**Implementation Examples**:

**Column-Based Partitioning**:
```sql
-- Original table
CREATE TABLE users (
    id INT PRIMARY KEY,
    email VARCHAR(255),
    password_hash VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    bio TEXT,
    profile_image BLOB,
    created_at TIMESTAMP
);

-- Split into frequently and rarely accessed columns
-- Frequently accessed (auth database)
CREATE TABLE user_auth (
    id INT PRIMARY KEY,
    email VARCHAR(255),
    password_hash VARCHAR(255),
    created_at TIMESTAMP
);

-- Less frequently accessed (profile database)
CREATE TABLE user_profiles (
    id INT PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    bio TEXT,
    profile_image BLOB
);
```

**Functional Partitioning**:
```text
Service-Based Partitioning:
- User Service: user authentication, profiles
- Order Service: orders, payments, shipping
- Catalog Service: products, categories, inventory
- Analytics Service: user behavior, metrics
```

```python
class DatabaseService:
    def __init__(self):
        self.user_db = connect_to_database('user_service')
        self.order_db = connect_to_database('order_service')
        self.catalog_db = connect_to_database('catalog_service')
        self.analytics_db = connect_to_database('analytics_service')
    
    def get_user_profile(self, user_id):
        return self.user_db.get_user(user_id)
    
    def create_order(self, user_id, items):
        return self.order_db.create_order(user_id, items)
    
    def search_products(self, query):
        return self.catalog_db.search(query)
    
    def track_event(self, user_id, event):
        return self.analytics_db.insert_event(user_id, event)
```

## 🔄 Advanced Scaling Patterns

### 1. **CQRS (Command Query Responsibility Segregation)**

**Pattern**: Separate read and write models for optimal performance

**Architecture**:
```text
Command Side (Writes):
Application → Command Handler → Write Database

Query Side (Reads):  
Application → Query Handler → Read Database (Optimized Views)

Synchronization:
Write Database → Event Stream → Read Database Updates
```

**Implementation**:
```python
# Command side - optimized for writes
class UserCommandHandler:
    def __init__(self):
        self.write_db = connect_to_write_db()
        self.event_bus = EventBus()
    
    def create_user(self, user_data):
        # Write to normalized database
        user_id = self.write_db.insert_user(user_data)
        
        # Publish event for read side
        self.event_bus.publish('user_created', {
            'user_id': user_id,
            'user_data': user_data
        })
        
        return user_id

# Query side - optimized for reads
class UserQueryHandler:
    def __init__(self):
        self.read_db = connect_to_read_db()
        self.event_bus = EventBus()
        self.event_bus.subscribe('user_created', self.handle_user_created)
    
    def handle_user_created(self, event):
        # Create denormalized view for fast reads
        user_data = event['user_data']
        self.read_db.insert_user_view({
            'user_id': event['user_id'],
            'display_name': f"{user_data['first_name']} {user_data['last_name']}",
            'email': user_data['email'],
            'created_at': user_data['created_at']
        })
    
    def get_user_profile(self, user_id):
        # Fast read from denormalized view
        return self.read_db.get_user_view(user_id)
```

### 2. **Event Sourcing**

**Pattern**: Store events instead of current state

**Architecture**:
```text
Command → Event Store → Event Stream → Read Models
                    ↓
              Event History (Immutable)
```

**Implementation**:
```python
class EventStore:
    def __init__(self):
        self.events_db = connect_to_events_db()
    
    def append_event(self, stream_id, event_type, event_data):
        event = {
            'stream_id': stream_id,
            'event_type': event_type,
            'event_data': event_data,
            'timestamp': time.time(),
            'sequence_number': self.get_next_sequence(stream_id)
        }
        return self.events_db.insert_event(event)
    
    def get_events(self, stream_id, from_sequence=0):
        return self.events_db.get_events(stream_id, from_sequence)

class UserAggregate:
    def __init__(self, user_id, event_store):
        self.user_id = user_id
        self.event_store = event_store
        self.state = {}
        self.load_from_events()
    
    def load_from_events(self):
        events = self.event_store.get_events(f"user-{self.user_id}")
        for event in events:
            self.apply_event(event)
    
    def apply_event(self, event):
        if event['event_type'] == 'user_created':
            self.state = event['event_data']
        elif event['event_type'] == 'user_updated':
            self.state.update(event['event_data'])
    
    def create_user(self, user_data):
        self.event_store.append_event(
            f"user-{self.user_id}",
            'user_created',
            user_data
        )
        self.apply_event({'event_type': 'user_created', 'event_data': user_data})
```

### 3. **Database Federation**

**Pattern**: Split databases by function and join at application level

**Architecture**:
```text
Application Layer:
├── User Federation (User DB + Profile DB)
├── Order Federation (Order DB + Payment DB)
└── Catalog Federation (Product DB + Inventory DB)
```

**Implementation**:
```python
class UserFederation:
    def __init__(self):
        self.auth_db = connect_to_auth_db()
        self.profile_db = connect_to_profile_db()
        self.preferences_db = connect_to_preferences_db()
    
    def get_complete_user(self, user_id):
        # Parallel queries to different databases
        auth_future = async_query(self.auth_db, 'get_auth', user_id)
        profile_future = async_query(self.profile_db, 'get_profile', user_id)
        prefs_future = async_query(self.preferences_db, 'get_prefs', user_id)
        
        # Combine results
        return {
            'auth': auth_future.result(),
            'profile': profile_future.result(),
            'preferences': prefs_future.result()
        }
```

## 📊 Database Performance Patterns

### 1. **Query Optimization Patterns**

**Index Strategies**:
```sql
-- Compound indexes for multiple column queries
CREATE INDEX idx_user_status_created 
ON users(status, created_at);

-- Partial indexes for filtered queries
CREATE INDEX idx_active_users 
ON users(created_at) 
WHERE status = 'active';

-- Covering indexes to avoid table lookups
CREATE INDEX idx_user_profile_covering 
ON users(id) INCLUDE (first_name, last_name, email);
```

**Query Patterns**:
```python
class OptimizedQueries:
    def __init__(self, db):
        self.db = db
    
    def get_recent_active_users(self, limit=100):
        # Optimized query with proper indexing
        return self.db.execute("""
            SELECT id, first_name, last_name, email
            FROM users 
            WHERE status = 'active' 
              AND last_login > NOW() - INTERVAL 30 DAY
            ORDER BY last_login DESC 
            LIMIT %s
        """, [limit])
    
    def get_user_order_summary(self, user_id):
        # Single query instead of N+1
        return self.db.execute("""
            SELECT 
                u.id, u.first_name, u.last_name,
                COUNT(o.id) as order_count,
                SUM(o.total) as total_spent
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.id = %s
            GROUP BY u.id, u.first_name, u.last_name
        """, [user_id])
```

### 2. **Connection Management Patterns**

**Connection Lifecycle**:
```python
class ManagedConnection:
    def __init__(self, pool):
        self.pool = pool
        self.connection = None
    
    def __enter__(self):
        self.connection = self.pool.get_connection()
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            # Rollback on exception
            self.connection.rollback()
        else:
            # Commit on success
            self.connection.commit()
        
        # Return connection to pool
        self.pool.return_connection(self.connection)

# Usage
with ManagedConnection(pool) as conn:
    conn.execute("INSERT INTO users (...) VALUES (...)")
    conn.execute("INSERT INTO profiles (...) VALUES (...)")
    # Automatic commit and connection return
```

### 3. **Caching Integration**

**Database + Cache Pattern**:
```python
class CachedDatabase:
    def __init__(self, database, cache):
        self.db = database
        self.cache = cache
    
    def get_user(self, user_id):
        # Try cache first
        cache_key = f"user:{user_id}"
        user = self.cache.get(cache_key)
        
        if user is None:
            # Cache miss - fetch from database
            user = self.db.get_user(user_id)
            if user:
                self.cache.set(cache_key, user, ttl=3600)
        
        return user
    
    def update_user(self, user_id, user_data):
        # Update database
        self.db.update_user(user_id, user_data)
        
        # Invalidate cache
        self.cache.delete(f"user:{user_id}")
    
    def get_user_orders(self, user_id, use_cache=True):
        if not use_cache:
            return self.db.get_user_orders(user_id)
        
        cache_key = f"user:{user_id}:orders"
        orders = self.cache.get(cache_key)
        
        if orders is None:
            orders = self.db.get_user_orders(user_id)
            # Shorter TTL for frequently changing data
            self.cache.set(cache_key, orders, ttl=300)
        
        return orders
```

## 💡 Best Practices

### 1. **Scaling Strategy**

**Incremental Scaling Approach**:
```text
Phase 1: Single Database + Optimization
- Query optimization
- Index tuning
- Connection pooling

Phase 2: Read Scaling
- Add read replicas
- Implement read/write splitting
- Geographic distribution

Phase 3: Write Scaling
- Implement sharding
- Vertical partitioning
- Cache layer

Phase 4: Advanced Patterns
- CQRS implementation
- Event sourcing
- Database federation
```

### 2. **Monitoring and Observability**

**Key Metrics**:
- **Query performance**: Response time, throughput
- **Resource utilization**: CPU, memory, I/O
- **Replication lag**: Primary-replica delay
- **Connection pool**: Active, idle, waiting connections
- **Cache performance**: Hit ratio, miss ratio

### 3. **Data Consistency**

**Consistency Patterns**:
- **Strong consistency**: Use for critical data (financial)
- **Eventual consistency**: Use for user-generated content
- **Session consistency**: Use for user-specific data
- **Causal consistency**: Use for related operations

## 🎓 Summary

Database scaling requires:

- **Read scaling**: Replicas and caching for read-heavy workloads
- **Write scaling**: Sharding and partitioning for write-heavy workloads
- **Pattern selection**: Choose appropriate patterns for your use case
- **Gradual implementation**: Scale incrementally as needed
- **Monitoring**: Track performance and adjust strategies

Remember: **Database scaling is not just about handling more data—it's about maintaining performance and consistency as your system grows.**

---

*"Scale your database not because you can, but because you must."*
