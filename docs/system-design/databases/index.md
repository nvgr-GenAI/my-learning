# Database Systems 🗄️

Master database design, scaling patterns, and data modeling for distributed systems. This comprehensive guide covers all database types, scaling strategies, and implementation patterns.


## 🎯 Database Fundamentals

Understanding the core principles that govern all database systems.

=== "🔧 ACID Properties"

    **Fundamental guarantees for reliable database transactions**
    
    **Atomicity:**

    - **Definition**: All operations in a transaction succeed or all fail
    - **Example**: Bank transfer - both debit and credit must complete together
    - **Implementation**: Transaction logs, rollback mechanisms
    - **Failure Handling**: Automatic rollback on any operation failure
    ```sql
    BEGIN TRANSACTION;
        UPDATE accounts SET balance = balance - 100 WHERE id = 1;
        UPDATE accounts SET balance = balance + 100 WHERE id = 2;
    COMMIT; -- Both succeed or both rollback
    ```
    
    **Consistency:**

    - **Definition**: Database remains in valid state before and after transaction
    - **Example**: Foreign key constraints, check constraints must be satisfied
    - **Implementation**: Constraint checking, validation rules
    - **Types**: Application-level consistency, database-level consistency
    
    **Isolation:**

    - **Definition**: Concurrent transactions don't interfere with each other
    - **Levels**: Read Uncommitted, Read Committed, Repeatable Read, Serializable
    - **Implementation**: Locking mechanisms, MVCC (Multi-Version Concurrency Control)
    - **Trade-offs**: Higher isolation = better consistency but lower performance
    
    **Durability:**

    - **Definition**: Committed transactions survive system failures
    - **Implementation**: Write-ahead logging (WAL), data persistence to disk
    - **Recovery**: Point-in-time recovery, backup and restore mechanisms
    - **Guarantees**: Data survives crashes, power failures, hardware issues

=== "� System Design Principles"

    **Database systems must balance fundamental distributed systems trade-offs**
    
    Key principles that govern database design decisions:
    
    - **📊 CAP Theorem**: Trade-offs between Consistency, Availability, and Partition Tolerance
    - **⚖️ ACID vs BASE**: Transaction guarantees vs. scalability
    - **🔄 Consistency Models**: Strong vs. eventual vs. weak consistency
    - **🎯 Performance vs. Durability**: Memory vs. disk trade-offs
    
    **📚 For detailed coverage of CAP theorem and other fundamental principles**, see the [System Design Principles Guide](../fundamentals/principles.md).

## 🎭 Database Types & Technologies

Understanding the complete landscape of database technologies and their optimal use cases is crucial for making informed architecture decisions.

**Quick Overview:**

**Relational Databases (SQL):**

- **Best for**: OLTP applications, complex queries, ACID requirements
- **Examples**: PostgreSQL, MySQL, Oracle, SQL Server
- **Strengths**: Strong consistency, mature ecosystem, complex relationships

**Document Databases:**

- **Best for**: Flexible schemas, rapid development, JSON-native apps
- **Examples**: MongoDB, CouchDB, Amazon DocumentDB
- **Strengths**: Schema flexibility, developer-friendly, horizontal scaling

**Key-Value Stores:**

- **Best for**: Caching, session storage, high-performance lookups
- **Examples**: Redis, DynamoDB, Riak
- **Strengths**: Extreme performance, simplicity, linear scaling

**Column-Family:**

- **Best for**: Time-series data, analytics, write-heavy workloads
- **Examples**: Cassandra, HBase, Google Bigtable
- **Strengths**: Write optimization, compression, analytical queries

**Graph Databases:**

- **Best for**: Social networks, recommendations, relationship analysis
- **Examples**: Neo4j, Amazon Neptune, ArangoDB
- **Strengths**: Relationship traversal, pattern matching, graph algorithms

**Time Series:**

- **Best for**: Monitoring, IoT data, metrics collection
- **Examples**: InfluxDB, TimescaleDB, Prometheus
- **Strengths**: Temporal optimization, compression, built-in time functions

**Vector Databases:**

- **Best for**: AI/ML applications, semantic search, recommendations
- **Examples**: Pinecone, Weaviate, Chroma, pgvector
- **Strengths**: Similarity search, AI integration, embedding storage

**📚 For comprehensive coverage of all database types, their characteristics, use cases, and selection criteria, see the dedicated [Database Types & Technologies Guide](types.md).**

The types guide covers:

- Detailed characteristics and features of each database type
- Performance comparisons and trade-off analysis
- Comprehensive use case examples and real-world applications
- Popular database systems and their strengths
- Decision frameworks for database selection
- Cost considerations and operational factors

## 🚀 Database Scaling Strategies

=== "📊 Horizontal Scaling (Sharding)"

    **Distribute data across multiple database instances for unlimited scalability**
    
    **Core Concept:**
    Sharding involves partitioning your database into smaller, more manageable pieces (shards) distributed across multiple servers. Each shard contains a subset of the total data, allowing parallel processing and unlimited scaling potential.
    
    **Key Sharding Strategies:**
    
    - **Hash-Based**: Use hash function for even distribution
    - **Range-Based**: Partition by key value ranges  
    - **Directory-Based**: Use lookup service for flexible routing
    - **Geographic**: Distribute by geographical regions
    
    **Primary Benefits:**
    
    - ✅ **Unlimited Scaling**: Add more shards as data grows
    - ✅ **Parallel Processing**: Queries execute simultaneously across shards
    - ✅ **Fault Isolation**: Issues in one shard don't affect others
    - ✅ **Cost Effective**: Use commodity hardware
    
    **Key Challenges:**
    
    - ❌ **Cross-Shard Operations**: Complex joins and transactions
    - ❌ **Data Rebalancing**: Adding/removing shards requires migration
    - ❌ **Application Complexity**: Code must be shard-aware
    - ❌ **Hotspot Management**: Uneven distribution can overload shards
    
    **📚 For comprehensive coverage including implementation patterns, shard key selection, rebalancing strategies, and real-world examples, see the dedicated [Database Sharding Guide](sharding.md).**

=== "🔄 Replication Strategies"

    **Create copies of data across multiple database instances**
    
    Database replication involves creating copies of data across multiple database instances to improve availability, fault tolerance, and read performance. This section provides an overview of key replication concepts.

    **Quick Overview:**

    **Master-Slave Replication:**
    - One write node (master) with multiple read replicas (slaves)
    - Scales read operations horizontally
    - Simple architecture but single point of failure for writes
    - Common for read-heavy workloads

    **Master-Master Replication:**
    - Multiple nodes accepting both reads and writes
    - Better write scalability and fault tolerance
    - Complex conflict resolution required
    - Suitable for geographically distributed applications

    **Replication Timing:**
    - **Synchronous**: Strong consistency, higher latency
    - **Asynchronous**: Eventually consistent, lower latency
    - **Semi-synchronous**: Balanced approach

    **📚 For comprehensive coverage of replication strategies, implementation patterns, monitoring, and best practices, see the dedicated [Database Replication Guide](replication.md).**

    The replication guide covers:
    - Detailed implementation patterns for master-slave and master-master setups
    - Failover strategies and automatic promotion
    - Load balancing techniques for read replicas
    - Monitoring replication lag and health
    - Security considerations and performance optimization
    - Real-world use cases and selection criteria

=== "📈 Vertical Scaling"

    **Increase the power of existing database hardware**
    
    Vertical scaling (scaling up) involves increasing the resources of a single database server rather than distributing the load across multiple servers. This approach maintains the simplicity of a single database instance while improving performance.

    **CPU Scaling:**

    Enhancing processing power to handle more concurrent operations and complex queries.

    - **More Cores**: Enables better parallel query processing and concurrent transaction handling
    - **Faster Processors**: Reduces CPU-bound operation times and improves single-threaded performance
    - **Specialized Hardware**: NVMe storage controllers, dedicated network processors
    
    **Example Impact:**
    ```
    4-core CPU:   ~100 concurrent connections
    8-core CPU:   ~200 concurrent connections  
    16-core CPU:  ~400 concurrent connections
    32-core CPU:  ~800 concurrent connections
    ```

    **Memory Scaling:**

    Increasing RAM to cache more data in memory and reduce disk I/O operations.

    - **Larger RAM**: More data can be cached in memory, reducing expensive disk reads
    - **Buffer Pools**: Database can keep frequently accessed pages in memory
    - **In-Memory Databases**: Store entire working dataset in RAM for extreme performance
    
    **Memory Performance Benefits:**
    ```
    RAM Access:     ~100 nanoseconds
    SSD Access:     ~100 microseconds (1000x slower)
    HDD Access:     ~10 milliseconds (100,000x slower)
    
    # More RAM = Fewer disk I/O operations = Better performance
    ```

    **Storage Scaling:**

    Upgrading storage systems for faster data access and higher throughput.

    - **SSD/NVMe**: Dramatically faster random access and sequential read/write operations
    - **RAID Configurations**: Combine multiple drives for performance and redundancy
    - **Storage Networks**: High-speed SAN (Storage Area Network) connections
    
    **Storage Performance Comparison:**
    ```
    Traditional HDD:     100-200 IOPS,     ~150 MB/s
    SATA SSD:           10K-20K IOPS,     ~500 MB/s
    NVMe SSD:           100K+ IOPS,       ~3,500 MB/s
    High-end NVMe:      1M+ IOPS,         ~7,000 MB/s
    ```

    **Scaling Progression & Costs:**

    Understanding the cost and performance characteristics at different scaling levels.

    ```
    Entry Level:  4 CPU,   16GB RAM,  500GB SSD     → $1,000
                 Handles: ~1K users, 100 QPS
    
    Mid-Range:    8 CPU,   32GB RAM,  2TB SSD      → $4,000  
                 Handles: ~5K users, 500 QPS
    
    High-End:    16 CPU,  128GB RAM,  8TB NVMe     → $15,000
                 Handles: ~25K users, 2,500 QPS
    
    Enterprise:  32 CPU,  512GB RAM, 32TB NVMe     → $75,000
                 Handles: ~100K users, 10,000 QPS
    
    Maximum:     64 CPU,    2TB RAM, 64TB NVMe     → $200,000+
                 Handles: ~500K users, 50,000 QPS
    ```

    **When to Choose Vertical Scaling:**

    - **ACID Requirements**: Applications requiring strict consistency and transactional integrity
    - **Complex Queries**: Workloads involving complex JOINs across large datasets
    - **Legacy Applications**: Systems that cannot be easily modified for horizontal scaling
    - **Operational Simplicity**: Teams preferring single-instance management
    - **Data Relationships**: Highly interconnected data that benefits from co-location
    - **Initial Growth**: Early-stage applications with predictable scaling needs
    
    **Key Limitations:**

    - **Hardware Limits**: Physical constraints on CPU cores, RAM, and storage
    - **Cost Scaling**: Exponential cost increases at higher performance tiers
    - **Single Point of Failure**: No built-in redundancy or fault tolerance
    - **Downtime Requirements**: Hardware upgrades typically require maintenance windows

=== "🔀 Federation (Functional Partitioning)"

    **Split databases by feature or domain boundaries**
    
    **Core Concept:**

    Database federation (also known as functional partitioning) involves splitting databases by feature boundaries rather than by data volume. Each database is responsible for a specific domain or business function, allowing teams to optimize technology choices and scale independently.

    **Federation vs. Sharding:**
    ```
    Sharding:    Split same data type across multiple databases
                User table → User_Shard_1, User_Shard_2, User_Shard_3
    
    Federation:  Split different data types into separate databases  
                Users → User_DB, Products → Product_DB, Orders → Order_DB
    ```
    
    **Federation Strategies:**
    
    **By Business Domain (Microservices Architecture):**

    Organize databases around business capabilities and team ownership.
    - User Service → User Database (profiles, authentication, preferences)
    - Product Service → Product Database (catalog, inventory, pricing)
    - Order Service → Order Database (transactions, fulfillment)
    - Payment Service → Payment Database (billing, payment methods)

    **Benefits:**
    - ✅ **Team Autonomy**: Each team owns their data and can evolve independently
    - ✅ **Technology Choice**: Different services can use optimal database types
    - ✅ **Failure Isolation**: Issues in one domain don't affect others
    - ✅ **Independent Scaling**: Scale each service based on its specific needs
    
    **By Data Type (Polyglot Persistence):**

    Choose the optimal database technology for each type of data:
    - **Relational Data** → PostgreSQL (user profiles, transactions, billing)
    - **Document Data** → MongoDB (product catalogs, content management)
    - **Cache Data** → Redis (sessions, shopping carts, lookups)
    - **Search Data** → Elasticsearch (product search, content discovery)
    - **Time Series** → InfluxDB (metrics, monitoring, IoT data)
    - **Graph Data** → Neo4j (social networks, recommendations)
    
    **By Access Pattern (Performance Optimization):**

    Organize databases based on how data is accessed and used.

    ```
    Write-Heavy Workloads → Cassandra
    ├─ Event logging and audit trails
    ├─ IoT sensor data collection  
    └─ Real-time analytics ingestion
    
    Read-Heavy Workloads → Read Replicas + CDN
    ├─ Product catalogs and content
    ├─ Reporting and business intelligence
    └─ Historical data analysis
    
    Real-Time Operations → Redis + In-Memory DBs
    ├─ Live chat and messaging
    ├─ Real-time recommendations
    └─ Gaming leaderboards and sessions
    
    Analytical Workloads → Data Warehouse (Snowflake/BigQuery)
    ├─ Historical trend analysis
    ├─ Machine learning feature stores
    └─ Business intelligence and reporting
    ```

    **Data Access Patterns:**
    ```python
    # Example access pattern optimization
    class DataAccessOptimization:
        def __init__(self):
            self.write_heavy_db = CassandraClient()     # High write throughput
            self.read_heavy_db = PostgreSQLReplica()    # Optimized for reads
            self.real_time_db = RedisClient()           # Sub-millisecond access
            self.analytics_db = SnowflakeClient()       # Complex aggregations
        
        async def log_event(self, event):
            # Write-optimized database for high-volume events
            await self.write_heavy_db.insert(event)
        
        async def get_product_catalog(self):
            # Read-optimized replica for catalog browsing
            return await self.read_heavy_db.query("SELECT * FROM products")
        
        async def get_user_session(self, user_id):
            # Real-time access for session data
            return await self.real_time_db.get(f"session:{user_id}")
        
        async def generate_sales_report(self, date_range):
            # Analytics database for complex reporting
            return await self.analytics_db.query(complex_analytics_query)
    ```

    **Implementation Challenges & Solutions:**

    **Cross-Database Queries:**
    ```python
    # Problem: Need to join data across federated databases
    # Solution: Application-level joins or data synchronization
    
    class FederatedDataAccess:
        async def get_user_order_summary(self, user_id):
            # Get user info from User Service
            user = await self.user_service.get_user(user_id)
            
            # Get order history from Order Service  
            orders = await self.order_service.get_user_orders(user_id)
            
            # Get product details from Product Service
            product_ids = [item.product_id for order in orders for item in order.items]
            products = await self.product_service.get_products(product_ids)
            
            # Combine data at application level
            return self.combine_user_order_data(user, orders, products)
    ```

    **Data Consistency Across Services:**
    ```python
    # Saga Pattern for distributed transactions
    class OrderSaga:
        async def process_order(self, order_data):
    **Advantages:**

    - ✅ **Technology Optimization**: Best database for each use case
    - ✅ **Team Independence**: Different teams own different databases
    - ✅ **Failure Isolation**: Problems don't cascade across domains
    - ✅ **Scaling Independence**: Scale each service based on its needs
    - ✅ **Development Velocity**: Teams can move independently

    **Challenges:**

    - ❌ **Cross-Database Queries**: No easy joins across federated databases
    - ❌ **Distributed Transactions**: Complex to maintain consistency across services
    - ❌ **Operational Complexity**: Multiple database technologies to manage
    - ❌ **Data Duplication**: May need to replicate data across services
    - ❌ **Network Latency**: Inter-service communication overhead

## 📋 Database Selection Guide

=== "🎯 Selection Matrix"

    **Choose the right database for your specific use case**
    
    | Use Case | Primary Pattern | Recommended Database | Why This Choice |
    |----------|-----------------|---------------------|-----------------|
    | **OLTP Applications** | High transactions, ACID | PostgreSQL, MySQL | Strong consistency, mature ecosystem |
    | **Content Management** | Flexible schema, documents | MongoDB, CouchDB | Schema flexibility, JSON-native |
    | **Real-time Analytics** | Fast writes, time-series | Cassandra, ClickHouse | Write optimization, columnar storage |
    | **Session Storage** | Fast access, TTL | Redis, Memcached | In-memory speed, automatic expiration |
    | **Social Networks** | Relationships, traversal | Neo4j, Amazon Neptune | Graph algorithms, relationship queries |
    | **Search & Discovery** | Full-text search | Elasticsearch, Solr | Text indexing, relevance scoring |
    | **IoT & Monitoring** | Time-series data | InfluxDB, TimescaleDB | Time-based optimization, compression |
    | **AI/ML Applications** | Vector similarity | Pinecone, Weaviate | Embedding search, similarity algorithms |
    | **Financial Systems** | Strict consistency | PostgreSQL, CockroachDB | ACID compliance, audit trails |
    | **Gaming Leaderboards** | Sorted sets, rankings | Redis, DynamoDB | Fast sorted operations, low latency |

=== "⚡ Performance Characteristics"

    **Understanding performance trade-offs across database types**
    
    **Read Performance:**
    
    | Database Type | Single Record | Range Queries | Aggregations | Full-Text Search |
    |---------------|---------------|---------------|--------------|------------------|
    | **Key-Value** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ❌ |
    | **Document** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
    | **Relational** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
    | **Column-Family** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
    | **Graph** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐ |
    | **Search Engine** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
    
    **Write Performance:**
    
    | Database Type | Single Insert | Bulk Insert | Updates | Deletes |
    |---------------|---------------|-------------|---------|---------|
    | **Key-Value** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
    | **Document** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
    | **Relational** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
    | **Column-Family** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
    | **Time-Series** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ |

=== "🔄 Consistency Models"

    **Choose consistency level based on your requirements**
    
    **Strong Consistency:**

    - **Definition**: All reads receive the most recent write
    - **Databases**: PostgreSQL, MySQL, CockroachDB
    - **Use Cases**: Financial transactions, inventory management
    - **Trade-off**: Higher latency, lower availability
    
    **Eventual Consistency:**

    - **Definition**: System will become consistent over time
    - **Databases**: DynamoDB, Cassandra, MongoDB (default)
    - **Use Cases**: Social media feeds, content distribution
    - **Trade-off**: Higher availability, temporary inconsistency
    
    **Tunable Consistency:**

    - **Definition**: Choose consistency level per operation
    - **Databases**: Cassandra, MongoDB
    - **Examples**: Read/Write quorum settings
    - **Flexibility**: Balance consistency vs performance per use case
    
    **Session Consistency:**

    - **Definition**: Consistency within a user session
    - **Implementation**: Read from same replica, session tokens
    - **Use Cases**: User-facing applications
    - **Balance**: Good user experience with reasonable performance

=== "💰 Cost Considerations"

    **Total cost of ownership across different database types**
    
    **Operational Costs:**
    
    | Factor | Self-Hosted | Managed Service | Serverless |
    |--------|-------------|-----------------|------------|
    | **Hardware** | High | None | None |
    | **Operations** | High | Low | None |
    | **Scaling** | Manual | Semi-automatic | Automatic |
    | **Backup/Recovery** | Manual | Managed | Automatic |
    | **Security Updates** | Manual | Managed | Automatic |
    | **Expertise Required** | High | Medium | Low |
    
    **Scaling Cost Patterns:**
    
    ```
    Traditional RDBMS: Linear to exponential cost growth
    NoSQL Distributed: More linear cost scaling
    Serverless: Pay-per-use, cost scales with actual usage
    Cache-First: High cost for cache tier, but overall efficiency
    ```

## 🛠️ Implementation Patterns

=== "🔌 Connection Management"

    **Efficient database connection patterns**
    
    **Connection Pooling:**
    ```python
    import asyncio
    import asyncpg
    from contextlib import asynccontextmanager
    
    class DatabasePool:
        def __init__(self, database_url: str, min_size: int = 10, max_size: int = 20):
            self.database_url = database_url
            self.min_size = min_size
            self.max_size = max_size
            self.pool = None
        
        async def initialize(self):
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=60,
                server_settings={
                    'application_name': 'my_app',
                    'tcp_keepalives_idle': '600',
                    'tcp_keepalives_interval': '30',
                    'tcp_keepalives_count': '3',
                }
            )
        
        @asynccontextmanager
        async def get_connection(self):
            async with self.pool.acquire() as connection:
                yield connection
        
        async def execute_query(self, query: str, *args):
            async with self.get_connection() as conn:
                return await conn.fetch(query, *args)
        
        async def execute_transaction(self, operations):
            async with self.get_connection() as conn:
                async with conn.transaction():
                    results = []
                    for query, args in operations:
                        result = await conn.fetch(query, *args)
                        results.append(result)
                    return results
    
    # Usage
    db_pool = DatabasePool("postgresql://user:pass@localhost/db")
    await db_pool.initialize()
    ```
    
    **Connection Pool Configuration:**
    
    | Setting | Recommended Value | Purpose |
    |---------|------------------|---------|
    | **min_size** | 5-10 | Minimum connections to maintain |
    | **max_size** | 10-50 | Maximum connections (based on DB limits) |
    | **command_timeout** | 30-60 seconds | Query timeout |
    | **max_inactive_time** | 300 seconds | Close idle connections |

=== "📊 Query Optimization"

    **Best practices for efficient database queries**
    
    **Indexing Strategy:**
    ```sql
    -- Primary key index (automatic)
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Unique index for frequent lookups
    CREATE UNIQUE INDEX idx_users_email ON users(email);
    
    -- Composite index for common query patterns
    CREATE INDEX idx_orders_user_date ON orders(user_id, created_at DESC);
    
    -- Partial index for active records only
    CREATE INDEX idx_users_active ON users(created_at) WHERE active = true;
    
    -- GIN index for JSONB columns
    CREATE INDEX idx_users_metadata ON users USING gin(metadata);
    
    -- Full-text search index
    CREATE INDEX idx_posts_search ON posts USING gin(to_tsvector('english', title || ' ' || content));
    ```
    
    **Query Patterns:**
    ```sql
    -- Efficient pagination with cursor-based approach
    SELECT * FROM posts 
    WHERE created_at < $1 
    ORDER BY created_at DESC 
    LIMIT 20;
    
    -- Avoid N+1 queries with JOINs
    SELECT u.name, COUNT(p.id) as post_count
    FROM users u
    LEFT JOIN posts p ON u.id = p.user_id
    WHERE u.active = true
    GROUP BY u.id, u.name;
    
    -- Use EXISTS instead of IN for better performance
    SELECT * FROM users u
    WHERE EXISTS (
        SELECT 1 FROM orders o 
        WHERE o.user_id = u.id 
        AND o.created_at > '2024-01-01'
    );
    ```
    
    **Query Analysis:**
    ```sql
    -- Analyze query performance
    EXPLAIN (ANALYZE, BUFFERS) 
    SELECT * FROM orders 
    WHERE user_id = 123 
    AND status = 'pending';
    
    -- Monitor slow queries
    SELECT query, calls, total_time, mean_time
    FROM pg_stat_statements
    ORDER BY total_time DESC
    LIMIT 10;
    ```

=== "🔄 Data Migration & Evolution"

    **Strategies for evolving database schemas**
    
    **Schema Migration Pattern:**
    ```python
    class DatabaseMigration:
        def __init__(self, db_pool):
            self.db_pool = db_pool
        
        async def create_migration_table(self):
            await self.db_pool.execute_query("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(255) PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT NOW()
                )
            """)
        
        async def apply_migration(self, version: str, up_sql: str):
            async with self.db_pool.get_connection() as conn:
                async with conn.transaction():
                    # Check if migration already applied
                    existing = await conn.fetchval(
                        "SELECT version FROM schema_migrations WHERE version = $1",
                        version
                    )
                    
                    if existing:
                        print(f"Migration {version} already applied")
                        return
                    
                    # Apply migration
                    await conn.execute(up_sql)
                    
                    # Record migration
                    await conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES ($1)",
                        version
                    )
                    
                    print(f"Migration {version} applied successfully")
    
    # Example migrations
    migrations = [
        {
            "version": "001_create_users",
            "up": """
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """
        },
        {
            "version": "002_add_user_profile",
            "up": """
                ALTER TABLE users 
                ADD COLUMN first_name VARCHAR(100),
                ADD COLUMN last_name VARCHAR(100),
                ADD COLUMN profile_data JSONB;
                
                CREATE INDEX idx_users_profile ON users USING gin(profile_data);
            """
        }
    ]
    ```
    
    **Zero-Downtime Migration Strategies:**
    
    1. **Additive Changes**: Add new columns/tables without removing old ones
    2. **Backward Compatibility**: Support both old and new schema during transition
    3. **Feature Flags**: Control which version of schema to use
    4. **Blue-Green Deployment**: Migrate data to new database, switch traffic
    5. **Shadow Mode**: Run new schema in parallel, compare results

=== "🔐 Security & Compliance"

    **Database security best practices**
    
    **Access Control:**
    ```sql
    -- Create application-specific users
    CREATE USER app_read_only WITH PASSWORD 'secure_password';
    CREATE USER app_read_write WITH PASSWORD 'secure_password';
    
    -- Grant minimal required permissions
    GRANT CONNECT ON DATABASE myapp TO app_read_only;
    GRANT USAGE ON SCHEMA public TO app_read_only;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_read_only;
    
    GRANT CONNECT ON DATABASE myapp TO app_read_write;
    GRANT USAGE ON SCHEMA public TO app_read_write;
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_read_write;
    
    -- Row-level security
    CREATE POLICY user_data_policy ON user_data
        FOR ALL TO app_user
        USING (user_id = current_user_id());
    
    ALTER TABLE user_data ENABLE ROW LEVEL SECURITY;
    ```
    
    **Encryption & Protection:**
    
    - **Encryption at Rest**: Database file encryption (TDE)
    - **Encryption in Transit**: SSL/TLS for all connections
    - **Column Encryption**: Sensitive data encryption at application level
    - **Backup Encryption**: Encrypted backup files
    - **Audit Logging**: Track all database access and changes
    
    **Compliance Considerations:**
    
    | Regulation | Requirements | Implementation |
    |------------|-------------|----------------|
    | **GDPR** | Data privacy, right to deletion | User consent tracking, data anonymization |
    | **HIPAA** | Healthcare data protection | Encryption, access logging, BAAs |
    | **PCI DSS** | Payment card data security | Tokenization, network segmentation |
    | **SOX** | Financial data integrity | Audit trails, change controls |

## 🎯 Advanced Patterns

=== "🔄 CQRS & Event Sourcing"

    **Command Query Responsibility Segregation with Event Sourcing**
    
    **Core Concepts:**
    - **CQRS**: Separate models for reading and writing data
    - **Event Sourcing**: Store sequence of events instead of current state
    - **Event Store**: Append-only log of all events
    - **Projections**: Materialized views built from events
    
    **Implementation Pattern:**
    ```python
    from dataclasses import dataclass
    from typing import List, Any
    import json
    from datetime import datetime
    
    @dataclass
    class Event:
        event_id: str
        event_type: str
        aggregate_id: str
        event_data: dict
        version: int
        timestamp: datetime
    
    class EventStore:
        def __init__(self, db_pool):
            self.db_pool = db_pool
        
        async def append_events(self, aggregate_id: str, events: List[Event], expected_version: int):
            async with self.db_pool.get_connection() as conn:
                async with conn.transaction():
                    # Check current version
                    current_version = await conn.fetchval(
                        "SELECT COALESCE(MAX(version), 0) FROM events WHERE aggregate_id = $1",
                        aggregate_id
                    )
                    
                    if current_version != expected_version:
                        raise Exception(f"Concurrency conflict: expected {expected_version}, got {current_version}")
                    
                    # Append events
                    for event in events:
                        await conn.execute("""
                            INSERT INTO events (event_id, event_type, aggregate_id, event_data, version, timestamp)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, event.event_id, event.event_type, aggregate_id, 
                             json.dumps(event.event_data), event.version, event.timestamp)
        
        async def get_events(self, aggregate_id: str, from_version: int = 0) -> List[Event]:
            events = await self.db_pool.execute_query("""
                SELECT event_id, event_type, aggregate_id, event_data, version, timestamp
                FROM events 
                WHERE aggregate_id = $1 AND version > $2
                ORDER BY version
            """, aggregate_id, from_version)
            
            return [Event(
                event_id=row['event_id'],
                event_type=row['event_type'], 
                aggregate_id=row['aggregate_id'],
                event_data=json.loads(row['event_data']),
                version=row['version'],
                timestamp=row['timestamp']
            ) for row in events]
    
    # Command side (Write model)
    class UserAggregate:
        def __init__(self, user_id: str):
            self.user_id = user_id
            self.version = 0
            self.events = []
        
        def create_user(self, name: str, email: str):
            event = Event(
                event_id=str(uuid.uuid4()),
                event_type="UserCreated",
                aggregate_id=self.user_id,
                event_data={"name": name, "email": email},
                version=self.version + 1,
                timestamp=datetime.utcnow()
            )
            self.events.append(event)
            self.version += 1
        
        def change_email(self, new_email: str):
            event = Event(
                event_id=str(uuid.uuid4()),
                event_type="EmailChanged", 
                aggregate_id=self.user_id,
                event_data={"email": new_email},
                version=self.version + 1,
                timestamp=datetime.utcnow()
            )
            self.events.append(event)
            self.version += 1
    
    # Query side (Read model)
    class UserProjection:
        def __init__(self, db_pool):
            self.db_pool = db_pool
        
        async def handle_user_created(self, event: Event):
            await self.db_pool.execute_query("""
                INSERT INTO user_projections (user_id, name, email, version)
                VALUES ($1, $2, $3, $4)
            """, event.aggregate_id, event.event_data['name'], 
                event.event_data['email'], event.version)
        
        async def handle_email_changed(self, event: Event):
            await self.db_pool.execute_query("""
                UPDATE user_projections 
                SET email = $1, version = $2
                WHERE user_id = $3
            """, event.event_data['email'], event.version, event.aggregate_id)
    ```
    
    **Benefits:**
    - ✅ **Complete Audit Trail**: Every change is recorded
    - ✅ **Time Travel**: Rebuild state at any point in time
    - ✅ **Scalability**: Read and write models can scale independently  
    - ✅ **Flexibility**: Multiple read models from same events
    
    **Challenges:**
    - ❌ **Complexity**: More complex than CRUD operations
    - ❌ **Event Versioning**: Handling schema evolution
    - ❌ **Eventual Consistency**: Read models may lag behind events

=== "🎭 Polyglot Persistence"

    **Using multiple database technologies within the same application**
    
    **Strategy by Data Characteristics:**
    
    ```python
    class PolyglotDataService:
        def __init__(self):
            # Relational for transactional data
            self.postgres = PostgreSQLPool()
            
            # Document store for flexible schemas
            self.mongodb = MongoDBClient()
            
            # Cache for fast access
            self.redis = RedisClient()
            
            # Search engine for full-text search
            self.elasticsearch = ElasticsearchClient()
            
            # Time-series for metrics
            self.influxdb = InfluxDBClient()
            
            # Graph for relationships
            self.neo4j = Neo4jDriver()
        
        async def create_user(self, user_data):
            # Store core user data in PostgreSQL
            user_id = await self.postgres.execute_query("""
                INSERT INTO users (email, name, created_at) 
                VALUES ($1, $2, $3) RETURNING id
            """, user_data['email'], user_data['name'], datetime.utcnow())
            
            # Store profile data in MongoDB
            await self.mongodb.users.insert_one({
                'user_id': user_id,
                'profile': user_data.get('profile', {}),
                'preferences': user_data.get('preferences', {})
            })
            
            # Cache user session
            await self.redis.setex(
                f"user:{user_id}", 3600, 
                json.dumps({'name': user_data['name'], 'email': user_data['email']})
            )
            
            # Index for search
            await self.elasticsearch.index(
                index='users',
                id=user_id,
                document={
                    'name': user_data['name'],
                    'email': user_data['email'],
                    'bio': user_data.get('bio', '')
                }
            )
            
            return user_id
        
        async def get_user_complete(self, user_id):
            # Get core data from PostgreSQL
            user = await self.postgres.execute_query(
                "SELECT * FROM users WHERE id = $1", user_id
            )
            
            # Get profile from MongoDB
            profile = await self.mongodb.users.find_one({'user_id': user_id})
            
            # Get recent activity from InfluxDB
            activity = await self.influxdb.query(f"""
                SELECT * FROM user_activity 
                WHERE user_id = '{user_id}' 
                AND time > now() - 24h
            """)
            
            return {
                **user[0],
                'profile': profile.get('profile', {}),
                'preferences': profile.get('preferences', {}),
                'recent_activity': activity
            }
    ```
    
    **Database Selection by Domain:**
    
    | Domain | Database Choice | Rationale |
    |--------|----------------|-----------|
    | **User Authentication** | PostgreSQL | ACID compliance, security |
    | **Product Catalog** | MongoDB | Flexible product schemas |
    | **Shopping Cart** | Redis | Fast access, automatic expiration |
    | **Order History** | PostgreSQL | Complex queries, reporting |
    | **Product Search** | Elasticsearch | Full-text search, faceting |
    | **User Activity** | InfluxDB | Time-series data, analytics |
    | **Recommendations** | Neo4j | Graph relationships, algorithms |
    | **Content Management** | MongoDB | Document structure, versioning |

=== "🔄 Database Migrations"

    **Managing schema evolution in production systems**
    
    **Migration Strategies:**
    
    **1. Forward-Only Migrations:**
    ```python
    class Migration:
        def __init__(self, version: str, description: str):
            self.version = version
            self.description = description
        
        async def up(self, db_pool):
            raise NotImplementedError
        
        async def validate(self, db_pool):
            """Validate migration was applied correctly"""
            return True
    
    class AddUserProfileMigration(Migration):
        def __init__(self):
            super().__init__("002", "Add user profile fields")
        
        async def up(self, db_pool):
            await db_pool.execute_query("""
                ALTER TABLE users 
                ADD COLUMN IF NOT EXISTS profile_data JSONB,
                ADD COLUMN IF NOT EXISTS last_login TIMESTAMP;
                
                CREATE INDEX IF NOT EXISTS idx_users_profile 
                ON users USING gin(profile_data);
            """)
        
        async def validate(self, db_pool):
            # Check if columns exist
            result = await db_pool.execute_query("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'users' 
                AND column_name IN ('profile_data', 'last_login')
            """)
            return len(result) == 2
    ```
    
    **2. Backward-Compatible Changes:**
    ```sql
    -- Safe migrations (no downtime)
    ALTER TABLE users ADD COLUMN phone VARCHAR(20);  -- Add nullable column
    CREATE INDEX CONCURRENTLY idx_users_phone ON users(phone);  -- Non-blocking index
    
    -- Risky migrations (require careful planning)
    ALTER TABLE users DROP COLUMN old_field;  -- Data loss
    ALTER TABLE users ALTER COLUMN email TYPE TEXT;  -- Type change
    ```
    
    **3. Multi-Phase Migrations:**
    ```
    Phase 1: Add new column (nullable)
    Phase 2: Backfill data in new column
    Phase 3: Update application to use new column
    Phase 4: Make column non-nullable
    Phase 5: Remove old column
    ```

=== "🔗 Connection Pooling Strategies"

    **Advanced connection management patterns**
    
    **Connection Pool Configuration:**
    ```python
    class AdvancedConnectionPool:
        def __init__(self, config):
            self.config = config
            self.pools = {}
        
        async def get_read_pool(self):
            if 'read' not in self.pools:
                self.pools['read'] = await asyncpg.create_pool(
                    self.config['read_replica_url'],
                    min_size=10,
                    max_size=50,
                    command_timeout=30,
                    server_settings={
                        'default_transaction_isolation': 'read_committed',
                        'statement_timeout': '30s'
                    }
                )
            return self.pools['read']
        
        async def get_write_pool(self):
            if 'write' not in self.pools:
                self.pools['write'] = await asyncpg.create_pool(
                    self.config['master_url'],
                    min_size=5,
                    max_size=20,
                    command_timeout=60,
                    server_settings={
                        'default_transaction_isolation': 'read_committed'
                    }
                )
            return self.pools['write']
        
        async def execute_read(self, query, *args):
            pool = await self.get_read_pool()
            async with pool.acquire() as conn:
                return await conn.fetch(query, *args)
        
        async def execute_write(self, query, *args):
            pool = await self.get_write_pool()
            async with pool.acquire() as conn:
                return await conn.fetch(query, *args)
    ```
    
    **Pool Sizing Guidelines:**
    
    | Application Type | Pool Size Formula | Example |
    |-----------------|-------------------|---------|
    | **Web Application** | 2-3x CPU cores | 8 cores = 16-24 connections |
    | **API Service** | 1-2x CPU cores | 8 cores = 8-16 connections |
    | **Background Workers** | 0.5-1x CPU cores | 8 cores = 4-8 connections |
    | **Analytics** | 10-20 connections | Long-running queries |

## 🎓 Learning Path & Resources

**📚 Progressive Learning Track:**

=== "🌱 Beginner (Weeks 1-2)"

    **Foundation Concepts:**

    - [ ] **ACID Properties**: Understand database guarantees
    - [ ] **SQL Basics**: SELECT, INSERT, UPDATE, DELETE operations
    - [ ] **Database Design**: Normalization, relationships, constraints  
    - [ ] **Indexing Fundamentals**: Primary keys, simple indexes
    
    **Hands-on Projects:**

    - Build a simple blog with PostgreSQL
    - Create user registration/login system
    - Implement basic CRUD operations
    
    **Resources:**

    - [PostgreSQL Tutorial](https://www.postgresqltutorial.com/)
    - [SQLBolt Interactive Learning](https://sqlbolt.com/)

=== "🚀 Intermediate (Weeks 3-6)"

    **Scaling Concepts:**

    - [ ] **Read Replicas**: Master-slave replication setup
    - [ ] **Connection Pooling**: Efficient connection management
    - [ ] **Query Optimization**: EXPLAIN plans, performance tuning
    - [ ] **NoSQL Introduction**: Document and key-value stores
    
    **Advanced Projects:**

    - Implement read replica architecture
    - Build caching layer with Redis
    - Create document-based product catalog
    
    **Resources:**

    - [High Performance MySQL](https://www.oreilly.com/library/view/high-performance-mysql/9781449332471/)
    - [MongoDB University](https://university.mongodb.com/)

=== "🎯 Advanced (Weeks 7-12)"

    **Distributed Systems:**

    - [ ] **Database Sharding**: Horizontal partitioning strategies
    - [ ] **CAP Theorem**: Consistency vs availability trade-offs
    - [ ] **Event Sourcing**: Event-driven architectures
    - [ ] **Polyglot Persistence**: Multi-database strategies
    
    **Production Projects:**

    - Design and implement sharding strategy
    - Build CQRS system with event sourcing
    - Create monitoring and alerting system
    
    **Resources:**

    - [Designing Data-Intensive Applications](https://dataintensive.net/)
    - [Database Internals](https://databass.dev/)

**🔗 Related Topics:**

- **[Caching Strategies](../caching/index.md)** - Reduce database load with intelligent caching
- **[System Design Patterns](../fundamentals/index.md)** - Architectural patterns for scalable systems  
- **[Performance Optimization](../performance/index.md)** - Fine-tune system performance
- **[Monitoring & Observability](../reliability-security/monitoring.md)** - Track database health and performance

---

Master the data layer - the foundation of every great system! 🗄️💪
