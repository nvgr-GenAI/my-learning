# Database Sharding (Horizontal Scaling) 🔀

Master the art of distributing data across multiple database instances for unlimited scalability. This comprehensive guide covers the theory, implementation strategies, and real-world patterns for effective database sharding.

## 🎯 Understanding Sharding: Core Concepts

### What is Database Sharding?

**Definition:** Database sharding is a horizontal scaling technique that involves partitioning a database into smaller, more manageable pieces called "shards." Each shard is a separate database that contains a subset of the total data, distributed across multiple servers or database instances.

**The Problem Sharding Solves:**

Traditional databases face several limitations as they grow:

1. **Storage Limits**: Single servers have finite storage capacity
2. **Performance Bottlenecks**: CPU and memory constraints limit concurrent operations
3. **Network Bandwidth**: Single database connections become saturated
4. **Backup/Recovery Time**: Large databases take too long to backup and restore
5. **Geographic Distribution**: Single location creates latency for global users

**How Sharding Works:**

```
Traditional Single Database:
┌─────────────────────────────┐
│     Single Database         │
│  ┌─────────────────────────┐│
│  │   All User Data         ││
│  │   Users 1-1,000,000     ││
│  │   All Orders            ││
│  │   All Products          ││
│  └─────────────────────────┘│
└─────────────────────────────┘

Sharded Database:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Shard 1   │  │   Shard 2   │  │   Shard 3   │  │   Shard 4   │
│ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │
│ │Users    │ │  │ │Users    │ │  │ │Users    │ │  │ │Users    │ │
│ │1-250K   │ │  │ │250K-500K│ │  │ │500K-750K│ │  │ │750K-1M  │ │
│ │Orders   │ │  │ │Orders   │ │  │ │Orders   │ │  │ │Orders   │ │
│ │Products │ │  │ │Products │ │  │ │Products │ │  │ │Products │ │
│ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

### Key Benefits of Sharding

**1. Unlimited Horizontal Scaling**

- Add more shards as data volume grows
- No upper limit on total data size
- Scale compute and storage independently

**2. Improved Performance**

- Parallel query execution across shards
- Reduced contention and lock conflicts
- Lower memory pressure per shard

**3. Fault Isolation**

- Failure in one shard doesn't affect others
- Better overall system availability
- Easier maintenance and upgrades

**4. Cost Effectiveness**

- Use commodity hardware instead of expensive high-end servers
- Pay-as-you-scale model
- Better resource utilization

### Core Challenges of Sharding

**1. Cross-Shard Operations**

- Joins across shards require application-level logic
- Transactions spanning multiple shards are complex
- Maintaining referential integrity becomes difficult

**2. Data Distribution Challenges**

- Uneven data distribution creates hotspots
- Some shards may become overloaded
- Rebalancing requires careful planning

**3. Application Complexity**

- Applications must be shard-aware
- Routing logic adds complexity
- Error handling becomes more sophisticated

**4. Operational Overhead**

- Multiple databases to monitor and maintain
- Backup and recovery coordination
- Schema evolution across shards

## 🔑 Shard Key Selection: The Foundation

### Understanding Shard Keys

**Definition:** A shard key is the field or combination of fields used to determine which shard contains a particular piece of data. The choice of shard key is crucial for effective sharding.

**Characteristics of a Good Shard Key:**

1. **High Cardinality**: Many possible values to ensure good distribution
2. **Even Distribution**: Values spread evenly across the key space
3. **Query Alignment**: Supports common query patterns
4. **Immutable**: Rarely changes to avoid data movement
5. **Accessible**: Available in most queries to enable routing

**Examples of Good and Bad Shard Keys:**

```python
# Good Shard Keys:
user_id          # High cardinality, even distribution
email_hash       # Good distribution if users are diverse
customer_id      # Natural partition boundary

# Bad Shard Keys:
status           # Low cardinality (active/inactive)
created_date     # Can create hotspots for recent data
country          # Uneven distribution (some countries have more users)
```

### Shard Key Design Patterns

**1. Natural Keys**

```python
# User-based sharding
shard_key = user_id
# Good for: User-centric applications, social media, gaming

# Customer-based sharding  
shard_key = customer_id
# Good for: B2B applications, multi-tenant systems
```

**2. Synthetic Keys**

```python
# UUID-based sharding
import uuid
shard_key = str(uuid.uuid4())
# Good for: Distributed systems, microservices

# Hash-based composite keys
shard_key = hash(f"{user_id}_{timestamp}")
# Good for: Time-series data with user dimension
```

**3. Time-based Keys**

```python
# Date-based sharding
shard_key = created_date.strftime("%Y-%m")
# Good for: Time-series data, logs, events

# Epoch-based sharding
shard_key = int(timestamp / 86400)  # Days since epoch
# Good for: Event data, analytics
```

## 🛠️ Sharding Strategies: Theory and Implementation

=== "🎲 Hash-Based Sharding"

    ### Theory and Concepts
    
    **Core Principle:** Hash-based sharding uses a hash function to determine which shard should store a particular piece of data. This approach provides excellent data distribution and is the most commonly used sharding strategy.
    
    **How It Works Step-by-Step:**
    
    1. **Hash Function Application**: Apply a hash function to the shard key
    2. **Modulo Operation**: Use modulo arithmetic to map to a shard number
    3. **Shard Routing**: Direct the operation to the calculated shard
    4. **Consistent Results**: Same key always maps to same shard
    
    **Mathematical Foundation:**
    ```
    shard_number = hash(shard_key) % number_of_shards
    
    Example:
    hash("user_12345") = 987654321
    987654321 % 4 = 1 → Shard 1
    ```
    
    **Distribution Properties:**
    - **Uniform Distribution**: Good hash functions distribute keys evenly
    - **Deterministic**: Same input always produces same output
    - **Avalanche Effect**: Small input changes create large output changes
    
    ### Basic Implementation
    
    ```python
    import hashlib
    
    class HashBasedSharding:
        def __init__(self, num_shards):
            self.num_shards = num_shards
            
        def get_shard(self, key):
            """Simple hash-based shard selection"""
            return hash(str(key)) % self.num_shards
        
        def get_shard_secure(self, key):
            """More secure hash using MD5"""
            hash_value = hashlib.md5(str(key).encode()).hexdigest()
            return int(hash_value, 16) % self.num_shards
        
        def distribute_keys(self, keys):
            """Show how keys distribute across shards"""
            shard_counts = [0] * self.num_shards
            for key in keys:
                shard = self.get_shard(key)
                shard_counts[shard] += 1
            return shard_counts
    
    # Example usage:
    sharding = HashBasedSharding(4)
    
    # Test distribution
    test_keys = [f"user_{i}" for i in range(1000)]
    distribution = sharding.distribute_keys(test_keys)
    print(f"Shard distribution: {distribution}")
    # Output: [251, 248, 249, 252] - roughly even
    ```
    
    ### Advanced: Consistent Hashing
    
    **Problem with Simple Hashing:**
    When you add or remove shards, most keys need to be remapped:
    
    ```python
    # With 4 shards: hash(key) % 4
    # With 5 shards: hash(key) % 5
    # Most keys map to different shards!
    ```
    
    **Consistent Hashing Solution:**
    
    Consistent hashing minimizes data movement when shards are added or removed by using a circular hash space (ring) with virtual nodes.
    
    **Key Concepts:**
    - **Hash Ring**: Circular address space where both data keys and shards are hashed
    - **Virtual Nodes**: Multiple hash positions per physical shard for better distribution
    - **Clockwise Assignment**: Keys are assigned to the first shard found clockwise on the ring
    - **Minimal Movement**: Only ~1/n of keys move when adding/removing shards
    
    ```python
    class ConsistentHashSharding:
        def __init__(self, shards, virtual_nodes=150):
            self.shards = set(shards)
            self.virtual_nodes = virtual_nodes
            self.ring = {}  # hash_value -> shard_name
            self.sorted_hashes = []
            self._build_ring()
        
        def get_shard(self, key):
            """Find the shard for a given key"""
            hash_value = self._hash(key)
            idx = bisect.bisect_right(self.sorted_hashes, hash_value)
            if idx == len(self.sorted_hashes):
                idx = 0  # Wrap around to first shard
            return self.ring[self.sorted_hashes[idx]]
    
    # Benefits: Only ~25% of keys move when adding shards
    # (instead of ~75% with simple hashing)
    ```
    
    **For comprehensive consistent hashing implementation details**, see the [Consistent Hashing Guide](../consistent-hashing/index.md).
    
    ### Advantages and Disadvantages
    
    **Advantages:**
    - ✅ **Even Distribution**: Good hash functions provide uniform distribution
    - ✅ **Simplicity**: Easy to implement and understand
    - ✅ **Performance**: Fast shard lookup (O(1) for simple hash)
    - ✅ **Scalability**: Works well with many shards
    - ✅ **No Hotspots**: Random distribution prevents sequential access hotspots
    
    **Disadvantages:**
    - ❌ **Range Queries**: Need to query all shards for range operations
    - ❌ **Rebalancing**: Adding/removing shards requires data migration
    - ❌ **No Locality**: Related data may be on different shards
    - ❌ **Cross-Shard Joins**: Complex to join data across shards
    
    ### When to Use Hash-Based Sharding
    
    **Perfect For:**
    - High-volume OLTP applications
    - Key-value access patterns
    - Need for even load distribution
    - Stateless, horizontally scalable applications
    
    **Examples:**
    - User profile data sharded by user_id
    - Session storage sharded by session_id
    - Social media posts sharded by user_id
    - Gaming data sharded by player_id

=== "📊 Range-Based Sharding"

    **Partition data by value ranges for efficient range queries**
    
    **How it Works:**
    
    Range-based sharding divides data based on ranges of the shard key values. This strategy works well when you need to perform range queries or when data has natural ordering.
    
    - Define ranges for each shard based on key values
    - Route data to shards based on which range the key falls into
    - Maintains data locality for sequential keys
    - Enables efficient range queries within a shard
    
    **Example Partitioning:**
    ```
    Shard 1: User IDs 1-1,000,000
    Shard 2: User IDs 1,000,001-2,000,000
    Shard 3: User IDs 2,000,001-3,000,000
    Shard 4: User IDs 3,000,001-4,000,000
    
    # Query for users 500,000-1,500,000 hits Shard 1 and 2
    ```
    
    **Implementation:**
    ```python
    class RangeBasedSharding:
        def __init__(self):
            self.shard_ranges = [
                {"shard": "shard_1", "min": 1, "max": 1000000},
                {"shard": "shard_2", "min": 1000001, "max": 2000000},
                {"shard": "shard_3", "min": 2000001, "max": 3000000},
                {"shard": "shard_4", "min": 3000001, "max": 4000000},
            ]
        
        def get_shard(self, key):
            """Find shard for a given key"""
            for shard_info in self.shard_ranges:
                if shard_info["min"] <= key <= shard_info["max"]:
                    return shard_info["shard"]
            raise ValueError(f"No shard found for key: {key}")
        
        def get_shards_for_range(self, min_key, max_key):
            """Get all shards that contain data in the given range"""
            affected_shards = []
            for shard_info in self.shard_ranges:
                # Check if ranges overlap
                if (min_key <= shard_info["max"] and 
                    max_key >= shard_info["min"]):
                    affected_shards.append(shard_info["shard"])
            return affected_shards
        
        def split_shard(self, shard_name, split_point):
            """Split a shard at the given point"""
            shard_to_split = None
            for i, shard_info in enumerate(self.shard_ranges):
                if shard_info["shard"] == shard_name:
                    shard_to_split = i
                    break
            
            if shard_to_split is None:
                raise ValueError(f"Shard {shard_name} not found")
            
            original = self.shard_ranges[shard_to_split]
            
            # Create two new shards
            shard_1 = {
                "shard": f"{shard_name}_1",
                "min": original["min"],
                "max": split_point
            }
            shard_2 = {
                "shard": f"{shard_name}_2", 
                "min": split_point + 1,
                "max": original["max"]
            }
            
            # Replace original shard with split shards
            self.shard_ranges[shard_to_split:shard_to_split+1] = [shard_1, shard_2]
    ```
    
    **Time-Based Range Sharding:**
    ```python
    from datetime import datetime, timedelta
    
    class TimeBasedSharding:
        def __init__(self):
            # Shard by month
            self.shards = {
                "2024_01": {"start": "2024-01-01", "end": "2024-01-31"},
                "2024_02": {"start": "2024-02-01", "end": "2024-02-29"},
                "2024_03": {"start": "2024-03-01", "end": "2024-03-31"},
                # ... continue for each month
            }
        
        def get_shard(self, timestamp):
            """Get shard for a given timestamp"""
            date_str = timestamp.strftime("%Y-%m")
            for shard_name, shard_info in self.shards.items():
                if shard_name.startswith(date_str):
                    return shard_name
            
            # Create new shard for new month
            return self.create_new_monthly_shard(timestamp)
        
        def get_shards_for_date_range(self, start_date, end_date):
            """Get all shards covering the date range"""
            affected_shards = []
            current = start_date.replace(day=1)  # Start of month
            
            while current <= end_date:
                shard_name = current.strftime("%Y_%m")
                if any(s.startswith(shard_name) for s in self.shards.keys()):
                    affected_shards.append(f"shard_{shard_name}")
                
                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            
            return affected_shards
    ```
    
    **Pros:**
    - ✅ Efficient range queries within and across limited shards
    - ✅ Data locality for sequential operations
    - ✅ Easy to understand and debug
    - ✅ Natural partitioning for time-series data
    - ✅ Predictable query patterns
    
    **Cons:**
    - ❌ Potential for hotspots if data is not evenly distributed
    - ❌ Difficult to maintain balanced shards over time
    - ❌ Manual rebalancing required as data grows
    - ❌ Sequential access patterns can create bottlenecks

=== "📍 Directory-Based Sharding"

    **Use a lookup service for maximum flexibility**
    
    **How it Works:**
    
    Directory-based sharding uses a lookup service that maintains a mapping between data keys and their corresponding shard locations. This provides maximum flexibility but adds complexity.
    
    - Maintain a directory service that maps keys to shard locations
    - Can support any sharding logic (hash, range, custom rules)
    - Allows for dynamic shard assignment and rebalancing
    - Enables complex routing rules based on multiple factors
    
    **Architecture:**
    ```
    Application → Directory Service → Shard Location
                       ↓
    Directory Service Mapping:
    user_id → shard_location
    12345 → Shard-East-1
    67890 → Shard-West-2
    11111 → Shard-Central-1
    ```
    
    **Implementation:**
    ```python
    import redis
    from typing import Dict, List, Optional
    
    class DirectoryService:
        def __init__(self, redis_client):
            self.redis = redis_client
            self.shard_info = {}  # Cache shard information
        
        async def get_shard(self, key: str) -> str:
            """Get shard location for a given key"""
            shard = await self.redis.get(f"shard:{key}")
            if shard:
                return shard.decode()
            
            # If not found, assign to least loaded shard
            return await self.assign_to_optimal_shard(key)
        
        async def assign_to_optimal_shard(self, key: str) -> str:
            """Assign key to the optimal shard based on current load"""
            shard_loads = await self.get_shard_loads()
            optimal_shard = min(shard_loads.items(), key=lambda x: x[1])[0]
            
            # Store mapping
            await self.redis.set(f"shard:{key}", optimal_shard)
            await self.redis.incr(f"load:{optimal_shard}")
            
            return optimal_shard
        
        async def get_shard_loads(self) -> Dict[str, int]:
            """Get current load for all shards"""
            shards = ["shard_1", "shard_2", "shard_3", "shard_4"]
            loads = {}
            
            for shard in shards:
                load = await self.redis.get(f"load:{shard}")
                loads[shard] = int(load) if load else 0
            
            return loads
        
        async def migrate_key(self, key: str, target_shard: str):
            """Migrate a key to a different shard"""
            old_shard = await self.get_shard(key)
            
            # Update directory mapping
            await self.redis.set(f"shard:{key}", target_shard)
            
            # Update load counters
            await self.redis.decr(f"load:{old_shard}")
            await self.redis.incr(f"load:{target_shard}")
            
            return old_shard
        
        async def rebalance_shards(self):
            """Rebalance data across shards"""
            loads = await self.get_shard_loads()
            avg_load = sum(loads.values()) / len(loads)
            
            # Find overloaded and underloaded shards
            overloaded = {k: v for k, v in loads.items() if v > avg_load * 1.2}
            underloaded = {k: v for k, v in loads.items() if v < avg_load * 0.8}
            
            # Migrate keys from overloaded to underloaded shards
            for overloaded_shard in overloaded:
                for underloaded_shard in underloaded:
                    if loads[overloaded_shard] <= avg_load:
                        break
                    
                    # Find a key to migrate
                    keys_pattern = f"shard:{overloaded_shard}:*"
                    keys = await self.redis.keys(keys_pattern)
                    
                    if keys:
                        key_to_migrate = keys[0].decode().split(":")[-1]
                        await self.migrate_key(key_to_migrate, underloaded_shard)
                        loads[overloaded_shard] -= 1
                        loads[underloaded_shard] += 1
    ```
    
    **Custom Routing Rules:**
    ```python
    class CustomRoutingDirectory:
        def __init__(self):
            self.routing_rules = {
                "premium_users": "high_performance_shard",
                "trial_users": "standard_shard",
                "enterprise_users": "dedicated_shard"
            }
            self.geographic_routing = {
                "US": ["us_east_shard", "us_west_shard"],
                "EU": ["eu_central_shard", "eu_west_shard"],
                "ASIA": ["asia_pacific_shard", "asia_southeast_shard"]
            }
        
        def get_shard(self, user_id: str, user_type: str, region: str) -> str:
            """Get shard based on multiple factors"""
            
            # Rule 1: Route by user type
            if user_type in self.routing_rules:
                return self.routing_rules[user_type]
            
            # Rule 2: Route by geography
            if region in self.geographic_routing:
                region_shards = self.geographic_routing[region]
                # Use hash to distribute within region
                shard_index = hash(user_id) % len(region_shards)
                return region_shards[shard_index]
            
            # Rule 3: Default hash-based routing
            return f"default_shard_{hash(user_id) % 4}"
    ```
    
    **Pros:**
    - ✅ Maximum flexibility in data placement
    - ✅ Easy rebalancing and migration
    - ✅ Support for complex routing rules
    - ✅ Can optimize for different data access patterns
    - ✅ Dynamic shard assignment
    
    **Cons:**
    - ❌ Additional complexity and infrastructure
    - ❌ Directory service becomes a potential bottleneck
    - ❌ Need to ensure directory service high availability
    - ❌ Extra network hop for lookups
    - ❌ Directory can become a single point of failure

=== "🌍 Geographic Sharding"

    **Distribute data by geographic regions**
    
    **How it Works:**
    
    Geographic sharding distributes data based on geographical regions to improve performance and comply with data residency requirements.
    
    - Partition data based on geographic regions
    - Place shards close to users for reduced latency
    - Comply with data sovereignty and privacy regulations
    - Often combined with other sharding strategies
    
    **Example Distribution:**
    ```
    US Users → US Database Cluster (Virginia, California)
    EU Users → EU Database Cluster (Ireland, Frankfurt)  
    Asia Users → Asia Database Cluster (Singapore, Tokyo)
    
    Benefits:
    - EU users' data stays in EU (GDPR compliance)
    - Reduced latency for regional users
    - Better disaster recovery per region
    ```
    
    **Implementation:**
    ```python
    class GeographicSharding:
        def __init__(self):
            self.region_mappings = {
                "US": {
                    "primary": "us_east_shard",
                    "secondary": "us_west_shard",
                    "data_center": ["virginia", "california"],
                    "regulations": ["CCPA", "SOX"]
                },
                "EU": {
                    "primary": "eu_central_shard", 
                    "secondary": "eu_west_shard",
                    "data_center": ["frankfurt", "ireland"],
                    "regulations": ["GDPR", "DPA"]
                },
                "ASIA": {
                    "primary": "asia_pacific_shard",
                    "secondary": "asia_southeast_shard", 
                    "data_center": ["singapore", "tokyo"],
                    "regulations": ["PDPA", "PIPEDA"]
                }
            }
        
        def get_shard_by_region(self, user_region: str, operation_type: str = "read"):
            """Get appropriate shard based on region and operation"""
            if user_region not in self.region_mappings:
                # Default to US for unknown regions
                user_region = "US"
            
            region_config = self.region_mappings[user_region]
            
            if operation_type == "write":
                return region_config["primary"]
            else:
                # Load balance reads between primary and secondary
                return region_config["secondary"]
        
        def get_shard_by_ip(self, ip_address: str):
            """Determine region from IP address and get shard"""
            region = self.ip_to_region(ip_address)
            return self.get_shard_by_region(region)
        
        def ip_to_region(self, ip_address: str) -> str:
            """Convert IP address to region (simplified)"""
            # In practice, use a GeoIP service like MaxMind
            ip_ranges = {
                "US": ["192.168.1.0/24", "10.0.0.0/8"],
                "EU": ["172.16.0.0/12"], 
                "ASIA": ["203.0.113.0/24"]
            }
            
            # Simplified IP to region mapping
            # Replace with actual GeoIP lookup
            return "US"  # Default
        
        def ensure_compliance(self, user_region: str, data_type: str) -> bool:
            """Check if data can be stored in region based on regulations"""
            region_config = self.region_mappings.get(user_region, {})
            regulations = region_config.get("regulations", [])
            
            # Define data type compliance requirements
            compliance_matrix = {
                "personal_data": {"required_regions": ["EU"] if "GDPR" in regulations else []},
                "financial_data": {"required_regions": ["US"] if "SOX" in regulations else []},
                "health_data": {"required_regions": ["US"] if "HIPAA" in regulations else []}
            }
            
            requirements = compliance_matrix.get(data_type, {})
            required_regions = requirements.get("required_regions", [])
            
            return user_region in required_regions if required_regions else True
    ```
    
    **Multi-Region Architecture:**
    ```python
    class MultiRegionDatabase:
        def __init__(self):
            self.regional_clusters = {
                "us_east": {
                    "primary": "us-east-primary.db",
                    "replicas": ["us-east-replica-1.db", "us-east-replica-2.db"],
                    "latency_zone": ["us-east-1", "us-east-2", "us-central-1"]
                },
                "eu_central": {
                    "primary": "eu-central-primary.db", 
                    "replicas": ["eu-central-replica-1.db", "eu-west-replica-1.db"],
                    "latency_zone": ["eu-central-1", "eu-west-1", "eu-north-1"]
                },
                "asia_pacific": {
                    "primary": "ap-southeast-primary.db",
                    "replicas": ["ap-southeast-replica-1.db", "ap-northeast-replica-1.db"],
                    "latency_zone": ["ap-southeast-1", "ap-northeast-1", "ap-south-1"]
                }
            }
        
        async def route_request(self, user_region: str, operation: str, query: str):
            """Route database request to optimal regional cluster"""
            cluster = self.regional_clusters.get(user_region)
            
            if not cluster:
                # Fallback to nearest cluster
                cluster = self.find_nearest_cluster(user_region)
            
            if operation in ["SELECT", "READ"]:
                # Route reads to replicas for better performance
                target = random.choice(cluster["replicas"])
            else:
                # Route writes to primary
                target = cluster["primary"]
            
            return await self.execute_query(target, query)
        
        def find_nearest_cluster(self, user_region: str) -> dict:
            """Find nearest cluster based on latency zones"""
            # Simplified nearest cluster logic
            region_proximity = {
                "us_west": "us_east",
                "eu_west": "eu_central", 
                "asia_southeast": "asia_pacific"
            }
            
            nearest = region_proximity.get(user_region, "us_east")
            return self.regional_clusters[nearest]
    ```
    
    **Pros:**
    - ✅ Reduced latency for regional users
    - ✅ Compliance with data residency laws (GDPR, CCPA)
    - ✅ Natural disaster recovery boundaries
    - ✅ Can optimize infrastructure per region
    - ✅ Better user experience through proximity
    
    **Cons:**
    - ❌ Complex cross-region operations
    - ❌ Uneven shard sizes based on user distribution
    - ❌ Higher infrastructure and operational complexity
    - ❌ Difficult to handle users who travel frequently
    - ❌ Network latency for cross-region queries

## 🛠️ Implementation Patterns

=== "🔧 Shard-Aware Application Code"

    **Building applications that understand sharding**
    
    ```python
    class ShardedDatabase:
        def __init__(self, shard_configs, sharding_strategy="hash"):
            self.shards = {}
            self.sharding_strategy = sharding_strategy
            
            # Initialize shard connections
            for shard_id, config in shard_configs.items():
                self.shards[shard_id] = DatabaseConnection(config)
        
        def get_shard_key(self, user_id):
            """Determine which shard contains the data"""
            if self.sharding_strategy == "hash":
                return f"shard_{hash(user_id) % len(self.shards)}"
            elif self.sharding_strategy == "range":
                # Range-based logic
                if user_id <= 1000000:
                    return "shard_0"
                elif user_id <= 2000000:
                    return "shard_1"
                else:
                    return "shard_2"
        
        async def get_user(self, user_id):
            """Get user from appropriate shard"""
            shard_key = self.get_shard_key(user_id)
            shard = self.shards[shard_key]
            
            return await shard.query(
                "SELECT * FROM users WHERE id = ?", user_id
            )
        
        async def create_user(self, user_data):
            """Create user in appropriate shard"""
            user_id = user_data['id']
            shard_key = self.get_shard_key(user_id)
            shard = self.shards[shard_key]
            
            return await shard.execute(
                "INSERT INTO users (id, name, email) VALUES (?, ?, ?)",
                user_id, user_data['name'], user_data['email']
            )
        
        async def get_users_by_range(self, min_id, max_id):
            """Handle range queries across multiple shards"""
            results = []
            
            # Determine which shards might contain data in this range
            affected_shards = set()
            for user_id in range(min_id, max_id + 1, 1000):  # Sample the range
                affected_shards.add(self.get_shard_key(user_id))
            
            # Query each affected shard
            tasks = []
            for shard_key in affected_shards:
                shard = self.shards[shard_key]
                task = shard.query(
                    "SELECT * FROM users WHERE id BETWEEN ? AND ?",
                    min_id, max_id
                )
                tasks.append(task)
            
            # Execute queries in parallel
            shard_results = await asyncio.gather(*tasks)
            
            # Combine and sort results
            for shard_result in shard_results:
                results.extend(shard_result)
            
            return sorted(results, key=lambda x: x['id'])
    ```

=== "🔄 Shard Rebalancing"

    **Strategies for adding and removing shards**
    
    ```python
    class ShardRebalancer:
        def __init__(self, sharded_db):
            self.sharded_db = sharded_db
        
        async def add_new_shard(self, new_shard_config):
            """Add a new shard and rebalance data"""
            new_shard_id = f"shard_{len(self.sharded_db.shards)}"
            new_shard = DatabaseConnection(new_shard_config)
            
            # Create tables in new shard
            await self.create_shard_schema(new_shard)
            
            # Calculate which data should move to new shard
            await self.migrate_data_to_new_shard(new_shard_id, new_shard)
            
            # Add new shard to pool
            self.sharded_db.shards[new_shard_id] = new_shard
            
            print(f"Successfully added {new_shard_id}")
        
        async def create_shard_schema(self, shard):
            """Create necessary tables in new shard"""
            schema_sql = """
                CREATE TABLE users (
                    id INT PRIMARY KEY,
                    name VARCHAR(255),
                    email VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX idx_users_email ON users(email);
            """
            await shard.execute(schema_sql)
        
        async def migrate_data_to_new_shard(self, new_shard_id, new_shard):
            """Migrate appropriate data to the new shard"""
            # For hash-based sharding, need to rehash all data
            num_shards = len(self.sharded_db.shards) + 1  # Including new shard
            
            for old_shard_id, old_shard in self.sharded_db.shards.items():
                # Get all users from old shard
                users = await old_shard.query("SELECT * FROM users")
                
                users_to_migrate = []
                users_to_keep = []
                
                for user in users:
                    # Recalculate shard based on new number of shards
                    new_shard_key = f"shard_{hash(user['id']) % num_shards}"
                    
                    if new_shard_key == new_shard_id:
                        users_to_migrate.append(user)
                    elif new_shard_key == old_shard_id:
                        users_to_keep.append(user)
                    # User belongs to different existing shard - handle separately
                
                # Migrate users to new shard
                if users_to_migrate:
                    await self.bulk_insert_users(new_shard, users_to_migrate)
                    await self.bulk_delete_users(old_shard, [u['id'] for u in users_to_migrate])
        
        async def bulk_insert_users(self, shard, users):
            """Efficiently insert multiple users"""
            if not users:
                return
            
            values = []
            for user in users:
                values.extend([user['id'], user['name'], user['email']])
            
            placeholders = ",".join(["(?,?,?)"] * len(users))
            sql = f"INSERT INTO users (id, name, email) VALUES {placeholders}"
            
            await shard.execute(sql, *values)
        
        async def bulk_delete_users(self, shard, user_ids):
            """Efficiently delete multiple users"""
            if not user_ids:
                return
            
            placeholders = ",".join(["?"] * len(user_ids))
            sql = f"DELETE FROM users WHERE id IN ({placeholders})"
            
            await shard.execute(sql, *user_ids)
    ```

=== "🔍 Cross-Shard Queries"

    **Handling queries that span multiple shards**
    
    ```python
    class CrossShardQueryHandler:
        def __init__(self, sharded_db):
            self.sharded_db = sharded_db
        
        async def cross_shard_join(self, user_ids, include_orders=True):
            """Join user data with orders across shards"""
            
            # Group user IDs by shard
            users_by_shard = {}
            for user_id in user_ids:
                shard_key = self.sharded_db.get_shard_key(user_id)
                if shard_key not in users_by_shard:
                    users_by_shard[shard_key] = []
                users_by_shard[shard_key].append(user_id)
            
            # Fetch users from each shard
            user_tasks = []
            for shard_key, shard_user_ids in users_by_shard.items():
                shard = self.sharded_db.shards[shard_key]
                placeholders = ",".join(["?"] * len(shard_user_ids))
                
                task = shard.query(
                    f"SELECT * FROM users WHERE id IN ({placeholders})",
                    *shard_user_ids
                )
                user_tasks.append(task)
            
            # Execute user queries in parallel
            user_results = await asyncio.gather(*user_tasks)
            
            # Combine user results
            all_users = {}
            for shard_result in user_results:
                for user in shard_result:
                    all_users[user['id']] = user
            
            if not include_orders:
                return list(all_users.values())
            
            # Fetch orders for these users (assuming orders are also sharded by user_id)
            order_tasks = []
            for shard_key, shard_user_ids in users_by_shard.items():
                shard = self.sharded_db.shards[shard_key]
                placeholders = ",".join(["?"] * len(shard_user_ids))
                
                task = shard.query(
                    f"SELECT * FROM orders WHERE user_id IN ({placeholders})",
                    *shard_user_ids
                )
                order_tasks.append(task)
            
            # Execute order queries in parallel
            order_results = await asyncio.gather(*order_tasks)
            
            # Group orders by user
            orders_by_user = {}
            for shard_result in order_results:
                for order in shard_result:
                    user_id = order['user_id']
                    if user_id not in orders_by_user:
                        orders_by_user[user_id] = []
                    orders_by_user[user_id].append(order)
            
            # Combine users with their orders
            result = []
            for user_id, user in all_users.items():
                user['orders'] = orders_by_user.get(user_id, [])
                result.append(user)
            
            return result
        
        async def cross_shard_aggregation(self, metric="count"):
            """Perform aggregations across all shards"""
            # Execute same query on all shards in parallel
            tasks = [shard.query(aggregation_query) for shard in self.shards]
            results = await asyncio.gather(*tasks)
            # Combine results (sum, average, etc.)
            return combine_aggregation_results(results, metric)
        
        async def cross_shard_search(self, search_term, limit=100):
            """Search across all shards and combine results"""
            per_shard_limit = limit // len(self.sharded_db.shards) + 1
            
            # Execute search on all shards in parallel
            search_tasks = [
                shard.search(search_term, per_shard_limit) 
                for shard in self.shards
            ]
            results = await asyncio.gather(*search_tasks)
            
            # Merge and sort by relevance
            return merge_and_sort_results(results, limit)
    ```

## 📊 Monitoring & Maintenance

=== "📈 Shard Health Monitoring"

    **Essential metrics and monitoring strategies for sharded systems**
    
    **Key Metrics to Monitor:**
    
    **Performance Metrics:**
    - **Query Latency**: Average and 95th percentile response times
    - **Throughput**: Queries per second (QPS) per shard
    - **Error Rates**: Failed queries and timeout percentages
    - **Connection Pool**: Active connections and pool utilization
    
    **Resource Metrics:**
    - **CPU Usage**: Processing load per shard
    - **Memory Usage**: RAM utilization and cache hit rates
    - **Disk I/O**: Read/write operations and disk utilization
    - **Storage Growth**: Data size trends and capacity planning
    
    **Distribution Metrics:**
    - **Shard Balance**: Data size and query load distribution
    - **Hotspot Detection**: Identifying overloaded shards
    - **Cross-Shard Operations**: Frequency and performance impact
    
    **Health Check Implementation:**
    ```python
    class ShardMonitor:
        async def check_shard_health(self, shard_id):
            metrics = await self.collect_metrics(shard_id)
            
            # Performance thresholds
            if metrics.avg_query_time > 1000:  # 1 second
                await self.alert("High latency", shard_id)
            
            # Resource thresholds  
            if metrics.cpu_usage > 80:
                await self.alert("High CPU", shard_id)
            
            # Distribution thresholds
            if metrics.load_ratio > 1.5:  # 50% above average
                await self.trigger_rebalancing(shard_id)
    ```
    
    **Monitoring Tools Integration:**
    - **Metrics Collection**: Prometheus, Datadog, CloudWatch
    - **Alerting**: PagerDuty, Slack, email notifications
    - **Dashboards**: Grafana, Kibana for visualization
    - **Log Aggregation**: ELK stack, Splunk for analysis

=== "⚖️ Load Balancing"

    **Strategies for distributing load evenly across shards**
    
    **Load Balancing Concepts:**
    
    **Read Load Distribution:**
    - **Primary-Replica Pattern**: Route reads to replicas when primary is overloaded
    - **Least-Loaded Selection**: Direct queries to shard with lowest current load
    - **Geographic Routing**: Route to nearest shard for better latency
    - **Circuit Breaker**: Fail-fast when shards are unavailable
    
    **Write Load Management:**
    - **Writes to Primary Only**: Ensure data consistency
    - **Async Propagation**: Replicate writes asynchronously to replicas
    - **Write Buffer**: Queue writes during peak loads
    - **Backpressure**: Throttle writes when shards are overloaded
    
    **Load Metrics:**
    ```python
    class LoadMetrics:
        def calculate_shard_load(self, shard_metrics):
            # Normalize metrics to 0-1 scale
            cpu_load = shard_metrics.cpu_usage / 100
            memory_load = shard_metrics.memory_usage / 100
            query_load = min(shard_metrics.avg_response_time / 1000, 1.0)
            
            # Weighted combined score
            return (cpu_load * 0.4 + memory_load * 0.3 + query_load * 0.3)
    ```
    
    **Load Balancing Algorithms:**
    - **Round Robin**: Simple rotation through available shards
    - **Weighted Round Robin**: Consider shard capacity differences
    - **Least Connections**: Route to shard with fewest active connections
    - **Response Time**: Prefer shards with fastest response times
                stats['error_count'] += 1
            
            # Calculate derived metrics
            stats['avg_response_time'] = sum(stats['response_times']) / len(stats['response_times'])
            total_requests = stats['success_count'] + stats['error_count']
            stats['error_rate'] = stats['error_count'] / total_requests if total_requests > 0 else 0
            stats['last_updated'] = time.time()
        
        def is_circuit_broken(self, shard_id):
            """Check if circuit breaker is open for a shard"""
            breaker = self.circuit_breakers.get(shard_id)
            
            if not breaker:
                return False
            
            # Check if circuit should be reset
            if time.time() - breaker['opened_at'] > breaker['timeout']:
                # Try to close circuit
                del self.circuit_breakers[shard_id]
                return False
            
            return breaker['is_open']
        
        async def open_circuit_breaker(self, shard_id, timeout=60):
            """Open circuit breaker for a shard"""
            self.circuit_breakers[shard_id] = {
                'is_open': True,
                'opened_at': time.time(),
                'timeout': timeout
            }
            
            logger.warning(f"Circuit breaker opened for shard {shard_id}")
    ```

## 🚀 Best Practices

=== "✅ Design Guidelines"

    **Essential principles for successful sharding**
    
    **Choose the Right Shard Key:**
    
    - **High Cardinality**: Shard key should have many unique values
    - **Even Distribution**: Data should be distributed evenly across shards  
    - **Query Alignment**: Most queries should use the shard key
    - **Immutable**: Shard key should not change frequently
    
    ```python
    # Good shard keys
    good_shard_keys = [
        "user_id",        # High cardinality, even distribution
        "order_id",       # Unique, immutable
        "device_id",      # Device-specific data
        "tenant_id"       # Multi-tenant applications
    ]
    
    # Problematic shard keys  
    avoid_shard_keys = [
        "created_date",   # Creates hotspots (recent dates get all writes)
        "status",         # Low cardinality (only few unique values)
        "country",        # Uneven distribution (some countries have more users)
        "is_premium"      # Boolean values create uneven shards
    ]
    ```
    
    **Plan for Growth:**
    
    - **Over-shard Initially**: Start with more shards than currently needed
    - **Consistent Hashing**: Use algorithms that minimize data movement
    - **Monitoring**: Track shard performance and capacity metrics
    - **Automation**: Build tools for automated rebalancing
    
    **Design for Cross-Shard Operations:**
    
    - **Denormalize When Needed**: Duplicate data to avoid cross-shard joins
    - **Application-Level Joins**: Implement joins in application code
    - **Async Processing**: Use message queues for cross-shard updates
    - **Eventual Consistency**: Accept eventual consistency for better performance

=== "⚠️ Common Pitfalls"

    **Mistakes to avoid when implementing sharding**
    
    **Hotspot Creation:**
    
    ```python
    # Bad: Sequential IDs create hotspots
    def bad_user_id_generation():
        return get_next_sequential_id()  # All new users go to same shard
    
    # Good: UUID or random IDs
    def good_user_id_generation():
        return str(uuid.uuid4())  # Evenly distributed across shards
    
    # Bad: Time-based sharding for recent data
    def bad_shard_by_date(timestamp):
        return timestamp.strftime("%Y-%m")  # Recent month gets all writes
    
    # Good: Hash of ID + time component
    def good_shard_by_hash_time(user_id, timestamp):
        combined = f"{user_id}_{timestamp.strftime('%Y-%m')}"
        return hash(combined) % num_shards
    ```
    
    **Premature Sharding:**
    
    - **Start Simple**: Use vertical scaling first
    - **Measure First**: Profile your application before sharding
    - **Consider Alternatives**: Read replicas, caching, optimization
    - **Plan Migration**: Have a clear migration strategy
    
    **Ignoring Cross-Shard Queries:**
    
    ```python
    # Bad: Not planning for cross-shard operations
    class BadShardedService:
        def get_user_friends(self, user_id):
            # This requires querying all shards - very expensive!
            friends = []
            for shard in all_shards:
                shard_friends = shard.query(
                    "SELECT * FROM friendships WHERE user_id = ?", user_id
                )
                friends.extend(shard_friends)
            return friends
    
    # Good: Denormalize friend data to user's shard
    class GoodShardedService:
        def get_user_friends(self, user_id):
            user_shard = self.get_shard(user_id)
            # Friend list is stored in user's shard
            return user_shard.query(
                "SELECT * FROM user_friends WHERE user_id = ?", user_id
            )
    ```

=== "🎯 Performance Optimization"

    **Maximizing sharded database performance**
    
    **Connection Pool Sizing:**
    
    ```python
    # Optimize connection pools per shard
    def calculate_optimal_pool_size(shard_load_patterns):
        """Calculate optimal connection pool size for each shard"""
        
        pool_configs = {}
        for shard_id, load_pattern in shard_load_patterns.items():
            peak_concurrent_queries = load_pattern['peak_concurrent']
            avg_query_time = load_pattern['avg_query_time_ms']
            
            # Rule of thumb: pool size = peak_concurrent * (query_time / 1000) * 1.2
            optimal_size = int(peak_concurrent_queries * (avg_query_time / 1000) * 1.2)
            
            # Ensure minimum and maximum bounds
            pool_size = max(5, min(optimal_size, 50))
            
            pool_configs[shard_id] = {
                'min_size': max(2, pool_size // 2),
                'max_size': pool_size,
                'timeout': 30
            }
        
        return pool_configs
    ```
    
    **Query Optimization:**
    
    ```sql
    -- Use shard key in WHERE clauses
    SELECT * FROM orders 
    WHERE user_id = ? AND status = 'pending';  -- Good: uses shard key
    
    -- Avoid queries without shard key
    SELECT * FROM orders 
    WHERE status = 'pending';  -- Bad: hits all shards
    
    -- Create composite indexes including shard key
    CREATE INDEX idx_orders_user_status ON orders(user_id, status);
    CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);
    ```
    
    **Batch Operations:**
    
    ```python
    async def optimized_bulk_insert(self, records):
        """Efficiently insert records across shards"""
        
        # Group records by target shard
        records_by_shard = {}
        for record in records:
            shard_key = self.get_shard_key(record['user_id'])
            
            if shard_key not in records_by_shard:
                records_by_shard[shard_key] = []
            records_by_shard[shard_key].append(record)
        
        # Execute batch inserts in parallel
        insert_tasks = []
        for shard_key, shard_records in records_by_shard.items():
            task = self.batch_insert_to_shard(shard_key, shard_records)
            insert_tasks.append(task)
        
        results = await asyncio.gather(*insert_tasks)
        return sum(results)  # Total inserted records
    
    async def batch_insert_to_shard(self, shard_key, records):
        """Insert multiple records to a single shard efficiently"""
        if not records:
            return 0
        
        shard = self.shards[shard_key]
        
        # Use multi-row insert for better performance
        values = []
        placeholders = []
        
        for record in records:
            values.extend([record['user_id'], record['data'], record['timestamp']])
            placeholders.append("(?,?,?)")
        
        sql = f"INSERT INTO events (user_id, data, timestamp) VALUES {','.join(placeholders)}"
        
        await shard.execute(sql, *values)
        return len(records)
    ```

This comprehensive sharding guide provides you with the knowledge and tools needed to successfully implement horizontal scaling in your database architecture. Each strategy has its own trade-offs, so choose the approach that best fits your specific use case and requirements.
