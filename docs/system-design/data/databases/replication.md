# Database Replication Strategies 🔄

Master database replication concepts, patterns, and implementation strategies for building highly available distributed systems. This comprehensive guide covers the theory, trade-offs, and real-world patterns for effective database replication.

## 🎯 Understanding Database Replication

### What is Database Replication?

**Definition:** Database replication is the process of copying and maintaining database objects in multiple databases that make up a distributed database system. The goal is to improve availability, fault tolerance, and performance by creating redundant copies of data across multiple database instances.

**The Problem Replication Solves:**

1. **Single Point of Failure**: If your database goes down, your entire application stops working
2. **Performance Bottlenecks**: A single database can become overloaded with read requests
3. **Geographic Latency**: Users far from the database experience slow response times
4. **Disaster Recovery**: Natural disasters or hardware failures can cause permanent data loss
5. **Maintenance Downtime**: Database maintenance requires taking the system offline

**How Replication Works (Conceptual):**

```
Traditional Single Database:
┌─────────────────────────────┐
│     Application Layer       │
│  ┌─────────────────────────┐│
│  │     Read/Write All      ││  ← Single point of failure
│  │     Operations          ││  ← Performance bottleneck
│  └─────────────────────────┘│  ← No redundancy
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│      Single Database        │  ← If this fails, everything stops
└─────────────────────────────┘

Replicated Database System:
┌─────────────────────────────┐
│     Application Layer       │
│  ┌─────────────────────────┐│
│  │     Writes → Master     ││
│  │     Reads → Replicas    ││  ← Load distribution
│  └─────────────────────────┘│  ← Fault tolerance
└─────────────────────────────┘
              │
              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Master    │───▶│  Replica 1  │    │  Replica 2  │
│ (Read/Write)│    │ (Read Only) │    │ (Read Only) │
└─────────────┘    └─────────────┘    └─────────────┘
   ▲ If master fails, promote replica to master
```

### Core Benefits of Replication

#### 1. High Availability

- **Fault Tolerance**: System continues operating even if some databases fail
- **Automatic Failover**: Promote replicas to master when primary fails
- **Reduced Downtime**: Maintenance can be performed on individual replicas

#### 2. Performance Improvement

- **Read Scaling**: Distribute read queries across multiple replicas
- **Load Distribution**: Reduce load on the primary database
- **Geographic Performance**: Place replicas closer to users

#### 3. Data Protection

- **Redundancy**: Multiple copies protect against data loss
- **Backup Strategy**: Live replicas serve as continuous backups
- **Point-in-Time Recovery**: Historical data preserved across replicas

### Fundamental Challenges

#### 1. Consistency vs. Performance Trade-off

```
Strong Consistency (Synchronous)    vs    High Performance (Asynchronous)
├─ All replicas always have             ├─ Fast writes, eventual consistency
│  identical data                       ├─ Better user experience
├─ Slower writes (wait for all)         ├─ Risk of reading stale data
└─ Lower availability during failures   └─ Complex conflict resolution
```

#### 2. Replication Lag

- **Definition**: Time between a write on master and its appearance on replicas
- **Causes**: Network latency, processing overhead, replica load
- **Impact**: Users might read stale data from replicas

#### 3. Split-Brain Scenarios

- **Problem**: Network partition causes multiple nodes to think they're the master
- **Result**: Data divergence and potential corruption
- **Solution**: Quorum-based consensus and proper failover procedures

## 🏗️ Replication Architectures: Theory and Design

### Master-Slave Architecture

**Core Concept:** One primary database (master) handles all write operations, while multiple secondary databases (slaves/replicas) handle read operations and receive updates from the master.

**Theoretical Foundation:**

```
Write Path:
Client → Application → Master → [Process Write] → Replication Log
                                       ↓
                              Replicas ← Replication Stream

Read Path:
Client → Application → Load Balancer → Replica 1/2/3 → [Process Read]
```

**Data Flow Explanation:**

1. **Write Operations**: 
   - All writes go to the master database
   - Master processes the write and updates its local data
   - Master records the change in a replication log
   - Replication log is streamed to all replicas

2. **Read Operations**:
   - Reads can go to master or any replica
   - Load balancer distributes reads across replicas
   - Applications can choose read source based on consistency needs

**Types of Master-Slave Replication:**

1. **Statement-Based Replication**
   ```sql
   -- Master executes: UPDATE users SET last_login = NOW() WHERE id = 123
   -- Replicas execute the same statement
   -- Problem: NOW() gives different values on each replica
   ```

2. **Row-Based Replication**
   ```sql
   -- Master sends: UPDATE users SET last_login = '2024-01-15 10:30:00' WHERE id = 123
   -- Replicas apply the exact data changes
   -- More reliable but larger replication logs
   ```

3. **Mixed Replication**
   ```sql
   -- Use statement-based for simple operations
   -- Use row-based for non-deterministic operations
   -- Best of both worlds but more complex
   ```

### Master-Master Architecture

**Core Concept:** Multiple database instances can accept both read and write operations, with bidirectional replication between all masters.

**Theoretical Foundation:**

```
Bidirectional Replication:
Master A ←────────────────→ Master B
   ↑                           ↑
   │ Writes from               │ Writes from
   │ Region A                  │ Region B
   ↓                           ↓
Replica A1                  Replica B1
Replica A2                  Replica B2
```

**Conflict Resolution Theory:**

When multiple masters receive conflicting writes, the system needs a way to resolve conflicts:

1. **Last-Write-Wins (LWW)**
   ```python
   # Conflict: Both masters update same record simultaneously
   Master_A: UPDATE user SET name='Alice' WHERE id=1 AT timestamp=100
   Master_B: UPDATE user SET name='Bob'   WHERE id=1 AT timestamp=101
   
   # Resolution: timestamp=101 wins, final value is 'Bob'
   ```

2. **Vector Clocks**
   ```python
   # Track causality relationships between updates
   Master_A: [A:1, B:0] → UPDATE user SET name='Alice' WHERE id=1
   Master_B: [A:0, B:1] → UPDATE user SET name='Bob'   WHERE id=1
   
   # These are concurrent conflicts (neither causally depends on the other)
   # Need application-level resolution
   ```

3. **Application-Level Resolution**
   ```python
   # Business logic determines conflict resolution
   def resolve_user_conflict(user_a, user_b):
       return {
           'name': user_b.name,  # Prefer most recent name
           'email': user_a.email if user_a.email_verified else user_b.email,
           'preferences': merge_preferences(user_a.preferences, user_b.preferences)
       }
   ```

## 📋 Replication Patterns: Detailed Theory

=== "👑 Master-Slave Replication"

    ### Deep Dive: Theory and Implementation
    
    **Fundamental Principle:** Establish a clear hierarchy where one database instance (master) has the authority to process all write operations, while other instances (slaves) serve as read-only copies that are continuously synchronized with the master.
    
    **Theoretical Foundation:**
    
    The master-slave pattern is based on several key principles:
    
    1. **Single Source of Truth**: Master is the authoritative source for all data
    2. **Eventual Consistency**: Slaves eventually reflect master's state
    3. **Read Scaling**: Multiple slaves can serve read traffic simultaneously
    4. **Write Bottleneck**: All writes must go through the single master
    
    **Architecture Components Explained:**
    
    ```
    ┌─────────────────────────────────────────────────────┐
    │                Application Layer                    │
    │  ┌─────────────────┐    ┌─────────────────────────┐ │
    │  │ Write Operations│    │   Read Operations       │ │
    │  │ (INSERT/UPDATE/ │    │   (SELECT queries)      │ │
    │  │  DELETE)        │    │                         │ │
    │  └─────────────────┘    └─────────────────────────┘ │
    └─────────────────────────────────────────────────────┘
                │                           │
                ▼                           ▼
    ┌─────────────────┐           ┌─────────────────┐
    │   Master DB     │──────────▶│ Load Balancer   │
    │                 │           │ (Read Traffic)  │
    │ • Processes     │           └─────────────────┘
    │   writes        │                    │
    │ • Generates     │           ┌────────┼────────┐
    │   redo logs     │           ▼        ▼        ▼
    │ • Replicates    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   to slaves     │  │  Slave 1    │ │  Slave 2    │ │  Slave 3    │
    │                 │  │             │ │             │ │             │
    └─────────────────┘  │ • Read-only │ │ • Read-only │ │ • Read-only │
              │          │ • Applies   │ │ • Applies   │ │ • Applies   │
              │          │   redo logs │ │   redo logs │ │   redo logs │
              │          │ • May lag   │ │ • May lag   │ │ • May lag   │
              │          │   behind    │ │   behind    │ │   behind    │
              │          │   master    │ │   master    │ │   master    │
              │          └─────────────┘ └─────────────┘ └─────────────┘
              │                 ▲                ▲                ▲
              └─────────────────┼────────────────┼────────────────┘
                       Replication Stream
    ```
    
    **Step-by-Step Replication Process:**
    
    1. **Write Operation Initiation**
       - Application sends write request to master
       - Master validates the operation
       - Master acquires necessary locks
    
    2. **Master Processing**
       - Master executes the write operation
       - Changes are committed to master's storage
       - Operation is recorded in the replication log (redo log/binlog)
    
    3. **Replication Log Generation**
       - Master creates a replication event containing:
         - SQL statement (statement-based) or
         - Actual row changes (row-based) or
         - Mixed approach
    
    4. **Replication Transmission**
       - Master sends replication events to all slaves
       - Transmission can be:
         - **Synchronous**: Wait for slave acknowledgment
         - **Asynchronous**: Send and continue (faster)
    
    5. **Slave Processing**
       - Slaves receive replication events
       - Slaves apply changes to their local storage
       - Slaves update their replication position
    
    **Detailed Implementation with Theory:**
    
    ```python
    import asyncio
    import logging
    from enum import Enum
    from dataclasses import dataclass
    from typing import List, Optional, Dict
    
    class ConsistencyLevel(Enum):
        """Define different consistency requirements"""
        STRONG = "strong"      # Read from master
        EVENTUAL = "eventual"  # Read from slaves (may be stale)
        SESSION = "session"    # Read own writes
    
    @dataclass
    class ReplicationEvent:
        """Represents a change that needs to be replicated"""
        event_id: str
        timestamp: float
        query: str
        parameters: tuple
        affected_tables: List[str]
    
    class MasterSlaveReplicator:
        """
        Comprehensive master-slave replication implementation
        with detailed theoretical backing
        """
        
        def __init__(self, master_config: dict, slave_configs: List[dict]):
            self.master = DatabaseConnection(master_config)
            self.slaves = [DatabaseConnection(config) for config in slave_configs]
            self.replication_lag_tracker = ReplicationLagTracker()
            self.health_monitor = DatabaseHealthMonitor()
            self.session_manager = SessionManager()
        
        async def write(self, query: str, *args, 
                       replication_mode: str = "async") -> dict:
            """
            Process write operations with detailed replication
            
            Theory: All writes must go through master to maintain
            consistency and provide a single source of truth.
            """
            start_time = time.time()
            
            try:
                # 1. Validate write operation
                if not self._is_write_operation(query):
                    raise ValueError("Only write operations allowed in write()")
                
                # 2. Execute on master
                result = await self.master.execute(query, *args)
                
                # 3. Create replication event
                event = ReplicationEvent(
                    event_id=self._generate_event_id(),
                    timestamp=time.time(),
                    query=query,
                    parameters=args,
                    affected_tables=self._extract_table_names(query)
                )
                
                # 4. Replicate to slaves based on mode
                if replication_mode == "sync":
                    await self._synchronous_replication(event)
                else:
                    await self._asynchronous_replication(event)
                
                # 5. Track performance metrics
                write_latency = time.time() - start_time
                self._record_metrics("write_latency", write_latency)
                
                return {
                    "success": True,
                    "result": result,
                    "replication_event_id": event.event_id,
                    "latency_ms": write_latency * 1000
                }
                
            except Exception as e:
                logging.error(f"Write operation failed: {e}")
                await self._handle_write_failure(e, query, args)
                raise
        
        async def read(self, query: str, *args,
                      consistency: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
                      session_id: Optional[str] = None) -> dict:
            """
            Process read operations with consistency guarantees
            
            Theory: Read operations can be distributed across replicas
            for better performance, but consistency requirements must
            be considered.
            """
            
            # Determine read source based on consistency requirements
            if consistency == ConsistencyLevel.STRONG:
                # Strong consistency: always read from master
                return await self._read_from_master(query, *args)
            
            elif consistency == ConsistencyLevel.SESSION:
                # Session consistency: read own writes
                return await self._session_consistent_read(
                    query, args, session_id
                )
            
            else:
                # Eventual consistency: can read from slaves
                return await self._eventually_consistent_read(query, *args)
        
        async def _eventually_consistent_read(self, query: str, *args) -> dict:
            """
            Read from slaves with load balancing and failover
            
            Theory: Eventually consistent reads provide better performance
            but may return stale data due to replication lag.
            """
            
            # 1. Select optimal slave based on:
            #    - Health status
            #    - Current load
            #    - Replication lag
            #    - Geographic proximity
            
            optimal_slave = await self._select_optimal_slave()
            
            if optimal_slave:
                try:
                    # 2. Execute query on selected slave
                    result = await optimal_slave.execute(query, *args)
                    
                    # 3. Check if result is acceptably fresh
                    lag = await self.replication_lag_tracker.get_lag(optimal_slave)
                    
                    return {
                        "result": result,
                        "source": "slave",
                        "replication_lag_ms": lag * 1000,
                        "data_freshness": "eventually_consistent"
                    }
                    
                except Exception as e:
                    # 4. Failover to master if slave fails
                    logging.warning(f"Slave read failed, failing over to master: {e}")
                    return await self._read_from_master(query, *args)
            
            # 5. No healthy slaves available, use master
            return await self._read_from_master(query, *args)
        
        async def _session_consistent_read(self, query: str, args: tuple,
                                         session_id: str) -> dict:
            """
            Ensure session consistency: users read their own writes
            
            Theory: Session consistency guarantees that within a user
            session, reads reflect all writes made in that session.
            """
            
            # 1. Check if session has recent writes
            last_write_time = self.session_manager.get_last_write_time(session_id)
            
            if last_write_time:
                # 2. Calculate acceptable replication lag
                acceptable_lag = time.time() - last_write_time
                
                # 3. Find slave that's caught up enough
                for slave in self.slaves:
                    lag = await self.replication_lag_tracker.get_lag(slave)
                    if lag <= acceptable_lag:
                        try:
                            result = await slave.execute(query, *args)
                            return {
                                "result": result,
                                "source": "slave",
                                "consistency": "session",
                                "replication_lag_ms": lag * 1000
                            }
                        except Exception:
                            continue
            
            # 4. Fallback to master for guaranteed consistency
            return await self._read_from_master(query, *args)
        
        async def _synchronous_replication(self, event: ReplicationEvent):
            """
            Synchronous replication: wait for slave acknowledgment
            
            Theory: Provides strong durability guarantees but impacts
            write performance due to network round-trips.
            """
            
            replication_tasks = []
            
            for slave in self.slaves:
                if await self.health_monitor.is_healthy(slave):
                    task = self._replicate_to_slave(slave, event)
                    replication_tasks.append(task)
            
            # Wait for all slaves to acknowledge (or timeout)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*replication_tasks, return_exceptions=True),
                    timeout=5.0  # 5 second timeout
                )
            except asyncio.TimeoutError:
                logging.warning("Synchronous replication timeout")
                # Continue anyway - slaves will catch up eventually
        
        async def _asynchronous_replication(self, event: ReplicationEvent):
            """
            Asynchronous replication: fire-and-forget
            
            Theory: Provides better write performance but slaves
            may lag behind master.
            """
            
            # Queue replication events for background processing
            for slave in self.slaves:
                if await self.health_monitor.is_healthy(slave):
                    # Fire-and-forget: don't wait for completion
                    asyncio.create_task(
                        self._replicate_to_slave(slave, event)
                    )
    
    class ReplicationLagTracker:
        """Monitor and track replication lag across slaves"""
        
        async def get_lag(self, slave: DatabaseConnection) -> float:
            """
            Calculate replication lag for a slave
            
            Theory: Lag is the time difference between when a change
            was made on master vs when it appeared on the slave.
            """
            
            try:
                # Method 1: Compare replication positions
                master_position = await self._get_master_position()
                slave_position = await self._get_slave_position(slave)
                
                # Method 2: Use heartbeat timestamps
                master_heartbeat = await self._get_master_heartbeat()
                slave_heartbeat = await self._get_slave_heartbeat(slave)
                
                # Calculate lag in seconds
                position_lag = master_position - slave_position
                time_lag = master_heartbeat - slave_heartbeat
                
                # Return the more conservative estimate
                return max(position_lag, time_lag)
                
            except Exception as e:
                logging.error(f"Failed to calculate replication lag: {e}")
                return float('inf')  # Assume worst case
    ```
    
    ### Advantages and Challenges
    
    **Advantages:**
    
    ✅ **Simplicity**: Easy to understand and implement
    - Clear separation between read and write operations
    - Single source of truth eliminates conflict resolution
    - Well-established patterns and tooling
    
    ✅ **Read Scalability**: Horizontal scaling for read operations
    - Add more slaves to handle increased read traffic
    - Geographic distribution of read replicas
    - Load balancing across multiple slaves
    
    ✅ **Data Protection**: Built-in redundancy and backup
    - Multiple copies protect against hardware failures
    - Point-in-time recovery from slave snapshots
    - Continuous backup without affecting master performance
    
    **Challenges:**
    
    ❌ **Write Bottleneck**: Single master limits write scalability
    - All writes must go through one database instance
    - Master becomes the bottleneck for write-heavy applications
    - Cannot distribute write load geographically
    
    ❌ **Replication Lag**: Eventual consistency issues
    - Slaves may serve stale data
    - Read-after-write inconsistency
    - Different slaves may have different data at same time
    
    ❌ **Failover Complexity**: Master failure requires careful handling
    - Manual or automated promotion of slave to master
    - Risk of data loss during failover
    - Potential for split-brain scenarios
    
    ### When to Use Master-Slave Replication
    
    **Perfect For:**
    
    - **Read-Heavy Applications**: 80%+ read operations
    - **Eventual Consistency Tolerance**: Application can handle slightly stale data
    - **Simple Architecture**: Team prefers straightforward replication
    - **Established Applications**: Existing apps with clear read/write separation
    
    **Real-World Examples:**
    
    - **Social Media Feeds**: Timeline reads can be eventually consistent
    - **E-commerce Catalogs**: Product information doesn't need real-time updates
    - **Content Management**: Blog posts and articles don't require instant consistency
    - **Analytics Dashboards**: Reports can tolerate some data lag
                if consistency_level == "strong":
                    return await self.replication.read_from_master(query, *args)
                else:
                    return await self.replication.read(query, *args)
        
        def analyze_query_type(self, query):
            write_keywords = ["INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
            query_upper = query.strip().upper()
            
            for keyword in write_keywords:
                if query_upper.startswith(keyword):
                    return "WRITE"
            return "READ"
    ```

    **Consistency Handling:**
    
    ```python
    class ConsistencyManager:
        def __init__(self, replication_manager):
            self.replication = replication_manager
            self.read_after_write_timeout = 5.0  # seconds
        
        async def read_after_write(self, query, *args, write_timestamp=None):
            """Ensure read-after-write consistency"""
            if write_timestamp:
                # Wait for replication lag to catch up
                max_lag = await self.get_max_replication_lag()
                if max_lag > self.read_after_write_timeout:
                    # Read from master if lag is too high
                    return await self.replication.read_from_master(query, *args)
            
            return await self.replication.read(query, *args)
        
        async def get_max_replication_lag(self):
            """Check replication lag across all slaves"""
            max_lag = 0
            for slave in self.replication.slaves:
                lag = await self.check_replication_lag(slave)
                max_lag = max(max_lag, lag)
            return max_lag
    ```

    **Advantages:**
    
    - ✅ **Read Scalability**: Scales read operations horizontally
    - ✅ **Data Redundancy**: Provides backup copies of data
    - ✅ **Simple Architecture**: Easy to understand and implement
    - ✅ **Load Distribution**: Distributes read traffic across multiple nodes
    - ✅ **Disaster Recovery**: Slaves can serve as backup in case of master failure
    - ✅ **Analytics Workloads**: Dedicated slaves for reporting without impacting main workload

    **Challenges:**
    
    - ❌ **Single Point of Failure**: Master failure affects all writes
    - ❌ **Replication Lag**: Slaves may serve stale data
    - ❌ **Write Bottleneck**: All writes still go through single master
    - ❌ **Failover Complexity**: Manual or complex automatic failover process
    - ❌ **Data Consistency**: Eventually consistent reads from slaves
    - ❌ **Split-Brain Risk**: Multiple masters during failover scenarios

=== "⚖️ Master-Master Replication"

    **Multiple nodes accepting both reads and writes**
    
    **Architecture & Challenges:**
    
    Master-master replication allows multiple database instances to accept both read and write operations, providing better write scalability and fault tolerance.

    ```
    Master DB 1 (R/W) ←→ Master DB 2 (R/W)
         ↓                    ↓
    Slave DB 1           Slave DB 2
         ↓                    ↓
    Read Replicas        Read Replicas
    ```

    **Bidirectional Synchronization:**
    
    ```python
    class MasterMasterReplication:
        def __init__(self, master_configs):
            self.masters = [DatabaseConnection(config) for config in master_configs]
            self.conflict_resolver = ConflictResolver()
            self.vector_clock = VectorClock(len(self.masters))
        
        async def write(self, query, *args, preferred_master=None):
            """Write to preferred master with conflict resolution"""
            master_id = preferred_master or self.select_master()
            master = self.masters[master_id]
            
            # Add vector clock to track causality
            timestamp = self.vector_clock.increment(master_id)
            
            try:
                result = await master.execute_with_metadata(query, args, timestamp)
                await self.replicate_to_other_masters(query, args, timestamp, master_id)
                return result
            except ConflictException as e:
                return await self.resolve_and_retry(query, args, e)
        
        async def replicate_to_other_masters(self, query, args, timestamp, source_master_id):
            """Replicate write to all other masters"""
            tasks = []
            for i, master in enumerate(self.masters):
                if i != source_master_id:
                    task = self.replicate_to_master(master, query, args, timestamp)
                    tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        def select_master(self):
            """Select master based on load balancing strategy"""
            # Could be round-robin, least-loaded, geographic, etc.
            return random.randint(0, len(self.masters) - 1)
    ```

    **Conflict Resolution Strategies:**
    
    **Last-Write-Wins (LWW):**
    ```python
    class LastWriteWinsResolver:
        def resolve_conflict(self, record1, record2):
            """Resolve based on timestamp"""
            if record1.timestamp > record2.timestamp:
                return record1
            elif record2.timestamp > record1.timestamp:
                return record2
            else:
                # Same timestamp, use node ID as tiebreaker
                return record1 if record1.node_id > record2.node_id else record2
        
        def merge_records(self, base_record, conflicting_records):
            """Merge multiple conflicting records"""
            latest_record = base_record
            for record in conflicting_records:
                if record.timestamp > latest_record.timestamp:
                    latest_record = record
            return latest_record
    ```

    **Application-Level Resolution:**
    ```python
    class ApplicationLevelResolver:
        def resolve_user_preferences(self, pref1, pref2):
            """Merge user preferences intelligently"""
            merged = {}
            
            # Union of all preference keys
            all_keys = set(pref1.keys()) | set(pref2.keys())
            
            for key in all_keys:
                if key in pref1 and key in pref2:
                    # Handle conflicts based on preference type
                    if key == "theme":
                        merged[key] = pref2[key]  # Most recent theme wins
                    elif key == "notifications":
                        # Merge notification settings
                        merged[key] = {**pref1[key], **pref2[key]}
                    else:
                        merged[key] = pref2[key]  # Default to most recent
                else:
                    # No conflict, take the available value
                    merged[key] = pref1.get(key) or pref2.get(key)
            
            return merged
        
        def resolve_shopping_cart(self, cart1, cart2):
            """Merge shopping carts by combining items"""
            merged_items = {}
            
            # Combine items from both carts
            for item in cart1.items + cart2.items:
                if item.product_id in merged_items:
                    # Sum quantities for same product
                    merged_items[item.product_id].quantity += item.quantity
                else:
                    merged_items[item.product_id] = item
            
            return ShoppingCart(list(merged_items.values()))
    ```

    **Vector Clocks for Causality:**
    ```python
    class VectorClock:
        def __init__(self, num_nodes):
            self.clock = [0] * num_nodes
            self.node_id = None
        
        def increment(self, node_id):
            """Increment clock for this node"""
            self.node_id = node_id
            self.clock[node_id] += 1
            return self.clock.copy()
        
        def update(self, other_clock):
            """Update clock with received timestamp"""
            for i in range(len(self.clock)):
                self.clock[i] = max(self.clock[i], other_clock[i])
            if self.node_id is not None:
                self.clock[self.node_id] += 1
        
        def compare(self, other_clock):
            """Compare two vector clocks"""
            less_than = all(self.clock[i] <= other_clock[i] for i in range(len(self.clock)))
            greater_than = all(self.clock[i] >= other_clock[i] for i in range(len(self.clock)))
            
            if less_than and not greater_than:
                return "before"  # This happened before other
            elif greater_than and not less_than:
                return "after"   # This happened after other
            elif less_than and greater_than:
                return "equal"   # Concurrent/same event
            else:
                return "concurrent"  # Concurrent/conflicting events
    ```

    **Split-Brain Prevention:**
    ```python
    class SplitBrainDetector:
        def __init__(self, masters, quorum_size):
            self.masters = masters
            self.quorum_size = quorum_size
            self.active_masters = set()
        
        async def check_quorum(self):
            """Ensure we have quorum before accepting writes"""
            reachable_masters = await self.count_reachable_masters()
            
            if reachable_masters >= self.quorum_size:
                return True
            else:
                # Enter read-only mode to prevent split-brain
                await self.enter_readonly_mode()
                return False
        
        async def count_reachable_masters(self):
            """Count how many masters are reachable"""
            reachable = 0
            for master in self.masters:
                if await self.ping_master(master):
                    reachable += 1
            return reachable
        
        async def enter_readonly_mode(self):
            """Prevent writes when quorum is lost"""
            for master in self.masters:
                try:
                    await master.set_readonly(True)
                except Exception:
                    pass  # Master may be unreachable
    ```

    **Advantages:**
    
    - ✅ **No Single Point of Failure**: Multiple masters provide redundancy
    - ✅ **Write Scalability**: Distributes write load across masters
    - ✅ **Geographic Distribution**: Masters can be placed in different regions
    - ✅ **High Availability**: Automatic failover between masters
    - ✅ **Load Distribution**: Reads and writes distributed across nodes
    - ✅ **Disaster Recovery**: Natural multi-site disaster recovery

    **Challenges:**
    
    - ❌ **Conflict Resolution**: Complex logic for handling concurrent writes
    - ❌ **Data Consistency**: Risk of inconsistent data across masters
    - ❌ **Operational Complexity**: More complex monitoring and troubleshooting
    - ❌ **Split-Brain Scenarios**: Network partitions can cause data divergence
    - ❌ **Application Complexity**: Apps must handle conflict resolution
    - ❌ **Performance Overhead**: Conflict detection and resolution costs

=== "⚡ Synchronous vs Asynchronous"

    **Understanding replication timing and consistency trade-offs**
    
    **Synchronous Replication:**
    
    Synchronous replication ensures that writes are committed to both master and replicas before acknowledging success to the client.

    **Flow & Implementation:**
    ```python
    class SynchronousReplication:
        def __init__(self, master, replicas, timeout=5.0):
            self.master = master
            self.replicas = replicas
            self.timeout = timeout
            self.min_replicas = len(replicas) // 2 + 1  # Majority
        
        async def write(self, query, *args):
            """Synchronous write to master and replicas"""
            transaction_id = generate_transaction_id()
            
            try:
                # Phase 1: Prepare phase
                prepare_tasks = []
                prepare_tasks.append(self.master.prepare_transaction(transaction_id, query, args))
                
                for replica in self.replicas:
                    task = replica.prepare_transaction(transaction_id, query, args)
                    prepare_tasks.append(task)
                
                # Wait for majority to prepare
                prepared = await asyncio.wait_for(
                    self.wait_for_majority(prepare_tasks),
                    timeout=self.timeout
                )
                
                if len(prepared) < self.min_replicas:
                    raise ReplicationException("Insufficient replicas prepared")
                
                # Phase 2: Commit phase
                commit_tasks = []
                commit_tasks.append(self.master.commit_transaction(transaction_id))
                
                for replica in prepared:
                    task = replica.commit_transaction(transaction_id)
                    commit_tasks.append(task)
                
                # Wait for majority to commit
                committed = await asyncio.wait_for(
                    self.wait_for_majority(commit_tasks),
                    timeout=self.timeout
                )
                
                return {"success": True, "replicas_committed": len(committed)}
                
            except Exception as e:
                # Abort transaction on all nodes
                await self.abort_transaction(transaction_id)
                raise
        
        async def wait_for_majority(self, tasks):
            """Wait for majority of tasks to complete"""
            completed = []
            
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    completed.append(result)
                    
                    if len(completed) >= self.min_replicas:
                        return completed
                except Exception:
                    continue  # Continue waiting for others
            
            return completed
    ```

    **Advantages:**
    - ✅ **Strong Consistency**: All replicas have same data immediately
    - ✅ **Durability**: Data guaranteed to be persisted on multiple nodes
    - ✅ **ACID Compliance**: Maintains transactional guarantees
    - ✅ **No Data Loss**: Failure of master doesn't lose committed data

    **Disadvantages:**
    - ❌ **High Latency**: Must wait for all replicas before acknowledging
    - ❌ **Reduced Availability**: Failure of replicas blocks writes
    - ❌ **Network Dependency**: Sensitive to network latency and partitions
    - ❌ **Throughput Impact**: Lower write throughput due to coordination

    **Asynchronous Replication:**
    
    Asynchronous replication acknowledges writes to the client immediately after the master commits, then replicates to slaves in the background.

    ```python
    class AsynchronousReplication:
        def __init__(self, master, replicas):
            self.master = master
            self.replicas = replicas
            self.replication_queue = asyncio.Queue()
            self.replication_workers = []
            
            # Start background replication workers
            for i in range(len(replicas)):
                worker = asyncio.create_task(self.replication_worker(i))
                self.replication_workers.append(worker)
        
        async def write(self, query, *args):
            """Asynchronous write - fast acknowledgment"""
            # Write to master immediately
            result = await self.master.execute(query, *args)
            
            # Queue replication to slaves (fire-and-forget)
            replication_task = {
                "query": query,
                "args": args,
                "timestamp": time.time(),
                "transaction_id": result.get("transaction_id")
            }
            
            await self.replication_queue.put(replication_task)
            
            # Acknowledge immediately without waiting for replicas
            return result
        
        async def replication_worker(self, worker_id):
            """Background worker for replicating to slaves"""
            replica = self.replicas[worker_id]
            
            while True:
                try:
                    # Get next replication task
                    task = await self.replication_queue.get()
                    
                    # Replicate to this slave
                    await replica.execute(
                        task["query"], 
                        *task["args"],
                        transaction_id=task["transaction_id"]
                    )
                    
                    # Mark task as done
                    self.replication_queue.task_done()
                    
                except Exception as e:
                    await self.handle_replication_failure(replica, task, e)
        
        async def handle_replication_failure(self, replica, task, error):
            """Handle replication failures"""
            # Log the error
            logger.error(f"Replication failed for {replica}: {error}")
            
            # Implement retry logic
            if task.get("retry_count", 0) < 3:
                task["retry_count"] = task.get("retry_count", 0) + 1
                await asyncio.sleep(2 ** task["retry_count"])  # Exponential backoff
                await self.replication_queue.put(task)
            else:
                # Add to dead letter queue for manual intervention
                await self.add_to_dead_letter_queue(task, error)
    ```

    **Lag Monitoring:**
    ```python
    class ReplicationLagMonitor:
        def __init__(self, master, replicas):
            self.master = master
            self.replicas = replicas
        
        async def check_replication_lag(self):
            """Monitor replication lag across all replicas"""
            master_position = await self.get_master_position()
            lag_info = {}
            
            for i, replica in enumerate(self.replicas):
                try:
                    replica_position = await self.get_replica_position(replica)
                    lag_seconds = self.calculate_lag(master_position, replica_position)
                    lag_info[f"replica_{i}"] = {
                        "lag_seconds": lag_seconds,
                        "status": "healthy" if lag_seconds < 5 else "lagging"
                    }
                except Exception as e:
                    lag_info[f"replica_{i}"] = {
                        "lag_seconds": None,
                        "status": "error",
                        "error": str(e)
                    }
            
            return lag_info
        
        async def get_master_position(self):
            """Get current position in master's transaction log"""
            result = await self.master.execute("SELECT pg_current_wal_lsn()")
            return result[0]["pg_current_wal_lsn"]
        
        async def get_replica_position(self, replica):
            """Get current position of replica"""
            result = await replica.execute("SELECT pg_last_wal_replay_lsn()")
            return result[0]["pg_last_wal_replay_lsn"]
    ```

    **Advantages:**
    - ✅ **Low Latency**: Fast write acknowledgment from master only
    - ✅ **High Availability**: Master failure doesn't block writes to other replicas
    - ✅ **High Throughput**: Better write performance
    - ✅ **Network Resilience**: Network issues don't block writes

    **Disadvantages:**
    - ❌ **Eventual Consistency**: Replicas may lag behind master
    - ❌ **Data Loss Risk**: Master failure may lose recent writes
    - ❌ **Read Inconsistency**: Slaves may serve stale data
    - ❌ **Replication Lag**: Time delay in data propagation

    **Semi-Synchronous Replication:**
    
    A hybrid approach that waits for at least one replica to acknowledge before returning to the client.

    ```python
    class SemiSynchronousReplication:
        def __init__(self, master, replicas, min_sync_replicas=1):
            self.master = master
            self.replicas = replicas
            self.min_sync_replicas = min_sync_replicas
        
        async def write(self, query, *args):
            """Semi-synchronous write"""
            # Write to master
            master_result = await self.master.execute(query, *args)
            
            # Replicate synchronously to minimum required replicas
            sync_tasks = []
            for replica in self.replicas[:self.min_sync_replicas]:
                task = replica.execute(query, *args)
                sync_tasks.append(task)
            
            # Wait for minimum replicas to acknowledge
            sync_results = await asyncio.gather(*sync_tasks, return_exceptions=True)
            
            # Continue async replication to remaining replicas
            async_tasks = []
            for replica in self.replicas[self.min_sync_replicas:]:
                task = asyncio.create_task(replica.execute(query, *args))
                async_tasks.append(task)
            
            return {
                "master_result": master_result,
                "sync_replicas": len([r for r in sync_results if not isinstance(r, Exception)]),
                "async_replicas": len(async_tasks)
            }
    ```

## 🛠️ Implementation Patterns

=== "🔧 Failover Strategies"

    **Handling master failures and promoting replicas**
    
    **Automatic Failover:**
    ```python
    class AutomaticFailover:
        def __init__(self, master, replicas, health_check_interval=30):
            self.master = master
            self.replicas = replicas
            self.health_check_interval = health_check_interval
            self.is_monitoring = False
            self.current_master = master
        
        async def start_monitoring(self):
            """Start continuous health monitoring"""
            self.is_monitoring = True
            while self.is_monitoring:
                try:
                    await self.check_master_health()
                    await asyncio.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Health check error: {e}")
        
        async def check_master_health(self):
            """Check if master is healthy"""
            try:
                # Simple health check - try to execute a basic query
                await asyncio.wait_for(
                    self.current_master.execute("SELECT 1"),
                    timeout=10.0
                )
                return True
            except Exception:
                # Master is unhealthy, trigger failover
                await self.trigger_failover()
                return False
        
        async def trigger_failover(self):
            """Promote most up-to-date replica to master"""
            logger.warning("Master failure detected, starting failover process")
            
            # Find the most up-to-date replica
            best_replica = await self.select_best_replica()
            
            if not best_replica:
                raise Exception("No suitable replica found for promotion")
            
            # Promote replica to master
            await self.promote_replica_to_master(best_replica)
            
            # Update application configuration
            await self.update_master_reference(best_replica)
            
            # Reconfigure remaining replicas
            await self.reconfigure_replicas(best_replica)
            
            logger.info(f"Failover completed, new master: {best_replica}")
        
        async def select_best_replica(self):
            """Select replica with most recent data"""
            best_replica = None
            highest_position = None
            
            for replica in self.replicas:
                try:
                    position = await self.get_replica_position(replica)
                    if highest_position is None or position > highest_position:
                        highest_position = position
                        best_replica = replica
                except Exception:
                    continue  # Skip unhealthy replicas
            
            return best_replica
        
        async def promote_replica_to_master(self, replica):
            """Promote replica to accept writes"""
            # Stop replication from old master
            await replica.execute("SELECT pg_promote()")
            
            # Wait for promotion to complete
            await self.wait_for_promotion(replica)
            
            # Update replica to accept writes
            await replica.set_read_only(False)
        
        async def wait_for_promotion(self, replica, timeout=60):
            """Wait for replica promotion to complete"""
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    # Check if replica is now in master mode
                    result = await replica.execute("SELECT pg_is_in_recovery()")
                    if not result[0]["pg_is_in_recovery"]:
                        return True  # Promotion complete
                except Exception:
                    pass
                
                await asyncio.sleep(1)
            
            raise Exception("Replica promotion timeout")
    ```

    **Manual Failover:**
    ```python
    class ManualFailover:
        def __init__(self, master, replicas):
            self.master = master
            self.replicas = replicas
        
        async def planned_failover(self, target_replica):
            """Perform planned failover with minimal data loss"""
            # Step 1: Stop accepting new writes on master
            await self.master.set_read_only(True)
            
            # Step 2: Wait for all replicas to catch up
            await self.wait_for_replication_sync()
            
            # Step 3: Promote target replica
            await self.promote_replica_to_master(target_replica)
            
            # Step 4: Reconfigure other replicas
            remaining_replicas = [r for r in self.replicas if r != target_replica]
            await self.reconfigure_replicas_to_new_master(target_replica, remaining_replicas)
            
            # Step 5: Demote old master to replica (optional)
            await self.demote_master_to_replica(self.master, target_replica)
            
            return target_replica
        
        async def wait_for_replication_sync(self, timeout=300):
            """Wait for all replicas to catch up with master"""
            master_position = await self.get_master_position()
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                all_caught_up = True
                
                for replica in self.replicas:
                    try:
                        replica_position = await self.get_replica_position(replica)
                        if replica_position < master_position:
                            all_caught_up = False
                            break
                    except Exception:
                        all_caught_up = False
                        break
                
                if all_caught_up:
                    return True
                
                await asyncio.sleep(1)
            
            raise Exception("Replication sync timeout")
    ```

=== "🔄 Load Balancing"

    **Distributing read traffic across replicas**
    
    **Round-Robin Load Balancing:**
    ```python
    class RoundRobinLoadBalancer:
        def __init__(self, replicas):
            self.replicas = replicas
            self.current_index = 0
            self.lock = asyncio.Lock()
        
        async def get_next_replica(self):
            """Get next replica using round-robin"""
            async with self.lock:
                replica = self.replicas[self.current_index % len(self.replicas)]
                self.current_index += 1
                return replica
        
        async def execute_read(self, query, *args):
            """Execute read query on next available replica"""
            replica = await self.get_next_replica()
            return await replica.execute(query, *args)
    ```

    **Weighted Load Balancing:**
    ```python
    class WeightedLoadBalancer:
        def __init__(self, replica_weights):
            # replica_weights: [(replica, weight), ...]
            self.replica_weights = replica_weights
            self.total_weight = sum(weight for _, weight in replica_weights)
            self.current_weights = [0] * len(replica_weights)
        
        def get_next_replica(self):
            """Weighted round-robin selection"""
            # Find replica with highest current weight
            best_index = 0
            for i in range(1, len(self.current_weights)):
                if self.current_weights[i] > self.current_weights[best_index]:
                    best_index = i
            
            # Update weights
            replica, weight = self.replica_weights[best_index]
            self.current_weights[best_index] -= self.total_weight
            
            for i, (_, w) in enumerate(self.replica_weights):
                self.current_weights[i] += w
            
            return replica
    ```

    **Health-Based Load Balancing:**
    ```python
    class HealthAwareLoadBalancer:
        def __init__(self, replicas, health_check_interval=30):
            self.replicas = replicas
            self.healthy_replicas = set(replicas)
            self.replica_health = {replica: True for replica in replicas}
            self.health_check_interval = health_check_interval
            self.current_index = 0
        
        async def start_health_monitoring(self):
            """Start continuous health monitoring"""
            while True:
                await self.check_all_replica_health()
                await asyncio.sleep(self.health_check_interval)
        
        async def check_all_replica_health(self):
            """Check health of all replicas"""
            for replica in self.replicas:
                is_healthy = await self.check_replica_health(replica)
                
                if is_healthy and replica not in self.healthy_replicas:
                    self.healthy_replicas.add(replica)
                    logger.info(f"Replica {replica} is now healthy")
                elif not is_healthy and replica in self.healthy_replicas:
                    self.healthy_replicas.discard(replica)
                    logger.warning(f"Replica {replica} is now unhealthy")
        
        async def check_replica_health(self, replica):
            """Check if single replica is healthy"""
            try:
                await asyncio.wait_for(
                    replica.execute("SELECT 1"),
                    timeout=5.0
                )
                return True
            except Exception:
                return False
        
        async def get_healthy_replica(self):
            """Get next healthy replica"""
            if not self.healthy_replicas:
                raise Exception("No healthy replicas available")
            
            healthy_list = list(self.healthy_replicas)
            replica = healthy_list[self.current_index % len(healthy_list)]
            self.current_index += 1
            return replica
    ```

=== "📊 Monitoring & Alerting"

    **Comprehensive replication monitoring system**
    
    **Replication Metrics Collection:**
    ```python
    class ReplicationMonitor:
        def __init__(self, master, replicas):
            self.master = master
            self.replicas = replicas
            self.metrics = {}
        
        async def collect_metrics(self):
            """Collect comprehensive replication metrics"""
            metrics = {
                "timestamp": datetime.utcnow(),
                "master": await self.collect_master_metrics(),
                "replicas": []
            }
            
            for i, replica in enumerate(self.replicas):
                replica_metrics = await self.collect_replica_metrics(replica, i)
                metrics["replicas"].append(replica_metrics)
            
            self.metrics = metrics
            return metrics
        
        async def collect_master_metrics(self):
            """Collect master-specific metrics"""
            try:
                queries = {
                    "wal_position": "SELECT pg_current_wal_lsn()",
                    "active_connections": "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'",
                    "replication_slots": "SELECT slot_name, active, restart_lsn FROM pg_replication_slots",
                    "write_rate": "SELECT sum(tup_inserted + tup_updated + tup_deleted) FROM pg_stat_user_tables"
                }
                
                results = {}
                for metric, query in queries.items():
                    result = await self.master.execute(query)
                    results[metric] = result
                
                return {
                    "status": "healthy",
                    "metrics": results
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        async def collect_replica_metrics(self, replica, replica_id):
            """Collect replica-specific metrics"""
            try:
                queries = {
                    "wal_position": "SELECT pg_last_wal_replay_lsn()",
                    "is_in_recovery": "SELECT pg_is_in_recovery()",
                    "last_replay_timestamp": "SELECT pg_last_xact_replay_timestamp()",
                    "active_connections": "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                }
                
                results = {}
                for metric, query in queries.items():
                    result = await replica.execute(query)
                    results[metric] = result
                
                # Calculate replication lag
                lag = await self.calculate_replication_lag(replica)
                
                return {
                    "replica_id": replica_id,
                    "status": "healthy",
                    "lag_seconds": lag,
                    "metrics": results
                }
            except Exception as e:
                return {
                    "replica_id": replica_id,
                    "status": "error",
                    "error": str(e),
                    "lag_seconds": None
                }
        
        async def calculate_replication_lag(self, replica):
            """Calculate replication lag in seconds"""
            try:
                master_time = await self.master.execute("SELECT EXTRACT(EPOCH FROM now())")
                replica_time = await replica.execute("SELECT EXTRACT(EPOCH FROM pg_last_xact_replay_timestamp())")
                
                if replica_time[0] and master_time[0]:
                    return master_time[0]["extract"] - replica_time[0]["extract"]
                return None
            except Exception:
                return None
    ```

    **Alerting System:**
    ```python
    class ReplicationAlerting:
        def __init__(self, monitor, alert_thresholds):
            self.monitor = monitor
            self.thresholds = alert_thresholds
            self.alert_history = []
        
        async def check_alerts(self):
            """Check for alert conditions"""
            metrics = await self.monitor.collect_metrics()
            alerts = []
            
            # Check master health
            if metrics["master"]["status"] != "healthy":
                alerts.append({
                    "severity": "critical",
                    "type": "master_down",
                    "message": f"Master database is unhealthy: {metrics['master'].get('error', 'Unknown error')}"
                })
            
            # Check replica health and lag
            for replica_metrics in metrics["replicas"]:
                if replica_metrics["status"] != "healthy":
                    alerts.append({
                        "severity": "high",
                        "type": "replica_down",
                        "message": f"Replica {replica_metrics['replica_id']} is unhealthy"
                    })
                
                lag = replica_metrics.get("lag_seconds")
                if lag and lag > self.thresholds.get("max_lag_seconds", 30):
                    alerts.append({
                        "severity": "medium",
                        "type": "high_replication_lag",
                        "message": f"Replica {replica_metrics['replica_id']} lag is {lag:.2f} seconds"
                    })
            
            # Send alerts
            for alert in alerts:
                await self.send_alert(alert)
            
            return alerts
        
        async def send_alert(self, alert):
            """Send alert via configured channels"""
            # Add to history
            alert["timestamp"] = datetime.utcnow()
            self.alert_history.append(alert)
            
            # Send via different channels based on severity
            if alert["severity"] == "critical":
                await self.send_pagerduty_alert(alert)
                await self.send_slack_alert(alert)
                await self.send_email_alert(alert)
            elif alert["severity"] == "high":
                await self.send_slack_alert(alert)
                await self.send_email_alert(alert)
            else:
                await self.send_slack_alert(alert)
    ```

## 🔧 Best Practices

=== "⚙️ Configuration"

    **Optimal replication configuration settings**
    
    **PostgreSQL Configuration:**
    ```sql
    -- Master configuration (postgresql.conf)
    wal_level = replica
    max_wal_senders = 10
    wal_keep_segments = 64
    archive_mode = on
    archive_command = 'cp %p /var/lib/postgresql/archive/%f'
    
    -- Replica configuration
    hot_standby = on
    max_standby_streaming_delay = 30s
    max_standby_archive_delay = 60s
    wal_receiver_timeout = 60s
    ```

    **MySQL Configuration:**
    ```ini
    # Master configuration (my.cnf)
    [mysqld]
    server-id = 1
    log-bin = mysql-bin
    binlog-format = ROW
    gtid-mode = ON
    enforce-gtid-consistency = true
    
    # Replica configuration
    [mysqld]
    server-id = 2
    relay-log = relay-bin
    read-only = 1
    super-read-only = 1
    ```

=== "🛡️ Security"

    **Securing replication channels and access**
    
    **SSL/TLS Configuration:**
    ```python
    class SecureReplication:
        def __init__(self, master_config, replica_configs):
            # Configure SSL for all connections
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            
            self.master = DatabaseConnection(
                **master_config,
                ssl=ssl_context
            )
            
            self.replicas = [
                DatabaseConnection(**config, ssl=ssl_context)
                for config in replica_configs
            ]
        
        async def setup_replication_user(self):
            """Create dedicated replication user with minimal privileges"""
            await self.master.execute("""
                CREATE USER replication_user WITH REPLICATION LOGIN PASSWORD 'secure_password';
                GRANT CONNECT ON DATABASE mydb TO replication_user;
            """)
    ```

=== "⚡ Performance"

    **Optimizing replication performance**
    
    **Replication Tuning:**
    ```python
    class ReplicationOptimizer:
        def __init__(self, replication_manager):
            self.replication = replication_manager
        
        async def optimize_replication_performance(self):
            """Apply performance optimizations"""
            
            # Parallel replication workers
            await self.configure_parallel_replication()
            
            # Optimize network settings
            await self.optimize_network_settings()
            
            # Configure appropriate timeouts
            await self.configure_timeouts()
        
        async def configure_parallel_replication(self):
            """Enable parallel replication where supported"""
            for replica in self.replication.replicas:
                await replica.execute("""
                    SET max_parallel_workers_per_gather = 4;
                    SET max_worker_processes = 8;
                """)
        
        async def optimize_network_settings(self):
            """Optimize network settings for replication"""
            # TCP keepalive settings
            network_config = {
                "tcp_keepalives_idle": 600,
                "tcp_keepalives_interval": 30,
                "tcp_keepalives_count": 3
            }
            
            for replica in self.replication.replicas:
                for setting, value in network_config.items():
                    await replica.execute(f"SET {setting} = {value}")
    ```

## 🎯 Use Cases & Selection Guide

=== "📊 When to Use Each Strategy"

    **Choosing the right replication strategy**
    
    | Use Case | Recommended Strategy | Rationale |
    |----------|---------------------|-----------|
    | **Read-Heavy Applications** | Master-Slave Async | Scale reads, fast writes |
    | **Financial Systems** | Master-Slave Sync | Strong consistency required |
    | **Global Applications** | Master-Master | Geographic distribution |
    | **High Availability SaaS** | Master-Slave with Auto-failover | Balance consistency and availability |
    | **Analytics Workloads** | Master-Slave with dedicated replicas | Isolate analytical queries |
    | **Multi-Region Deployment** | Master-Master with geo-distribution | Reduce latency per region |

=== "⚖️ Trade-off Analysis"

    **Understanding the trade-offs between different approaches**
    
    **Consistency vs Performance:**
    ```
    Strong Consistency (Sync)     ←→     High Performance (Async)
    - ACID guarantees                    - Low latency writes
    - No data loss                       - High throughput  
    - Higher latency                     - Eventual consistency
    - Lower availability                 - Potential data loss
    ```

    **Complexity vs Control:**
    ```
    Simple (Master-Slave)         ←→     Complex (Master-Master)
    - Single write point                 - Multiple write points
    - Clear consistency model            - Conflict resolution needed
    - Easier operations                  - More operational overhead
    - Single point of failure           - No single point of failure
    ```

This comprehensive guide covers all aspects of database replication strategies. For more advanced topics, see the [Sharding Guide](sharding.md) and [Database Scaling Strategies](index.md#database-scaling-strategies).
