# Database Systems 🗄️

Master database design, scaling patterns, and data modeling for distributed systems.

## 📋 Core Topics

### Database Types

- **[Relational Databases](relational.md)** - ACID properties, normalization, SQL optimization
- **[NoSQL Databases](nosql.md)** - Document, key-value, column-family, graph databases
- **[Time Series Databases](timeseries.md)** - InfluxDB, TimescaleDB for metrics and events
- **[Search Engines](search.md)** - Elasticsearch, Solr for full-text search

### Scaling Strategies

- **[Database Sharding](sharding.md)** - Horizontal partitioning strategies
- **[Read Replicas](replication.md)** - Master-slave, master-master replication
- **[Database Federation](federation.md)** - Splitting databases by function
- **[Denormalization](denormalization.md)** - Trading consistency for performance

### Advanced Patterns

- **[CQRS & Event Sourcing](cqrs.md)** - Command Query Responsibility Segregation
- **[Polyglot Persistence](polyglot.md)** - Using multiple database types
- **[Database Migrations](migrations.md)** - Schema evolution strategies
- **[Connection Pooling](pooling.md)** - Managing database connections

## 🔍 Quick Reference

### Database Selection Matrix

| Use Case | Recommended DB Type | Examples |
|----------|-------------------|----------|
| OLTP Applications | Relational | PostgreSQL, MySQL |
| Real-time Analytics | Column Store | Cassandra, HBase |
| Content Management | Document Store | MongoDB, CouchDB |
| Session Storage | Key-Value | Redis, DynamoDB |
| Social Networks | Graph Database | Neo4j, Amazon Neptune |
| Time Series Data | Time Series DB | InfluxDB, TimescaleDB |

### CAP Theorem Trade-offs

```mermaid
graph TD
    A[CAP Theorem] --> B[Consistency]
    A --> C[Availability]
    A --> D[Partition Tolerance]
    
    B --> E[ACID Databases<br/>PostgreSQL, MySQL]
    C --> F[Eventually Consistent<br/>DynamoDB, Cassandra]
    D --> G[Distributed Systems<br/>Required for scale]
```

## 🛠️ Implementation Examples

### Database Connection Pattern

```python
import asyncio
import asyncpg
from typing import Optional

class DatabasePool:
    def __init__(self, database_url: str, min_size: int = 10, max_size: int = 20):
        self.database_url = database_url
        self.min_size = min_size
        self.max_size = max_size
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=self.min_size,
            max_size=self.max_size,
            command_timeout=60
        )
    
    async def execute_query(self, query: str, *args):
        """Execute a query with connection pooling"""
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def execute_transaction(self, queries: list):
        """Execute multiple queries in a transaction"""
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                results = []
                for query, args in queries:
                    result = await connection.fetch(query, *args)
                    results.append(result)
                return results

# Usage example
async def main():
    db_pool = DatabasePool("postgresql://user:pass@localhost/db")
    await db_pool.initialize()
    
    # Simple query
    users = await db_pool.execute_query(
        "SELECT * FROM users WHERE active = $1", True
    )
    
    # Transaction example
    transaction_queries = [
        ("INSERT INTO orders (user_id, total) VALUES ($1, $2)", [1, 100.0]),
        ("UPDATE users SET last_order = NOW() WHERE id = $1", [1])
    ]
    await db_pool.execute_transaction(transaction_queries)
```

### Database Sharding Strategy

```python
import hashlib
from typing import List, Dict, Any

class DatabaseSharding:
    def __init__(self, shard_configs: List[Dict[str, Any]]):
        self.shards = {}
        for i, config in enumerate(shard_configs):
            self.shards[i] = DatabasePool(config['url'])
    
    def get_shard_key(self, user_id: int) -> int:
        """Hash-based sharding"""
        return hash(user_id) % len(self.shards)
    
    def get_range_shard(self, user_id: int) -> int:
        """Range-based sharding"""
        if user_id <= 1000000:
            return 0
        elif user_id <= 2000000:
            return 1
        else:
            return 2
    
    async def get_user(self, user_id: int):
        """Get user from appropriate shard"""
        shard_key = self.get_shard_key(user_id)
        shard = self.shards[shard_key]
        return await shard.execute_query(
            "SELECT * FROM users WHERE id = $1", user_id
        )
    
    async def create_user(self, user_data: dict):
        """Create user in appropriate shard"""
        user_id = user_data['id']
        shard_key = self.get_shard_key(user_id)
        shard = self.shards[shard_key]
        
        return await shard.execute_query(
            "INSERT INTO users (id, name, email) VALUES ($1, $2, $3)",
            user_data['id'], user_data['name'], user_data['email']
        )
```

## 📊 Performance Optimization

### Database Indexing Strategy

```sql
-- Primary index for user lookups
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);

-- Composite index for common query patterns
CREATE INDEX CONCURRENTLY idx_orders_user_status 
ON orders(user_id, status, created_at);

-- Partial index for active records only
CREATE INDEX CONCURRENTLY idx_users_active 
ON users(created_at) WHERE active = true;

-- GIN index for JSONB columns
CREATE INDEX CONCURRENTLY idx_users_metadata 
ON users USING gin(metadata);

-- Analyze index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### Query Optimization Checklist

- [ ] **Use appropriate indexes** for WHERE, JOIN, and ORDER BY clauses
- [ ] **Analyze query plans** with EXPLAIN ANALYZE
- [ ] **Limit result sets** with proper pagination
- [ ] **Use prepared statements** to prevent SQL injection
- [ ] **Batch operations** to reduce round trips
- [ ] **Monitor slow queries** and optimize regularly
- [ ] **Use read replicas** for read-heavy workloads
- [ ] **Implement connection pooling** to manage connections

## 🔗 Related Topics

- [Caching Strategies](../caching/index.md) - Reduce database load
- [Scalability Patterns](../scalability/index.md) - Scale database layer
- [Data Consistency](../distributed-systems/consistency.md) - Consistency patterns
- [Monitoring](../performance/monitoring.md) - Database monitoring

## 📚 Additional Resources

- [Database Internals Book](https://databass.dev/) - Deep dive into database systems
- [High Performance MySQL](https://www.oreilly.com/library/view/high-performance-mysql/9781449332471/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MongoDB Best Practices](https://docs.mongodb.com/manual/administration/production-notes/)
