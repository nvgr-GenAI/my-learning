# Databases

**Master database concepts for system design** | 💾 SQL/NoSQL | 🔄 Replication | 📊 Sharding

## Overview

Database design is crucial for system scalability, reliability, and performance. Choose the right database type and patterns for your use case.

---

## Quick Decision Guide

| Database Type | Use Case | Scale | Consistency | Examples |
|--------------|----------|-------|-------------|----------|
| **Relational (SQL)** | Structured data, ACID transactions | Vertical | Strong | PostgreSQL, MySQL |
| **Document (NoSQL)** | Flexible schema, JSON documents | Horizontal | Eventual | MongoDB, CouchDB |
| **Key-Value** | Simple lookups, caching | Horizontal | Eventual | Redis, DynamoDB |
| **Wide-Column** | Time-series, analytics | Horizontal | Tunable | Cassandra, HBase |
| **Graph** | Relationships, social networks | Medium | Strong | Neo4j, JanusGraph |
| **Time-Series** | Metrics, monitoring | Horizontal | Eventual | InfluxDB, TimescaleDB |

---

## Topics

| Topic | Status | Description |
|-------|--------|-------------|
| [SQL vs NoSQL](sql-vs-nosql.md) | 📝 Planned | When to use each type |
| [Replication](replication.md) | 📝 Planned | Master-slave, multi-master |
| [Sharding](sharding.md) | 📝 Planned | Horizontal partitioning |
| [Indexing](indexing.md) | 📝 Planned | B-tree, hash, composite indexes |
| [Transactions](transactions.md) | 📝 Planned | ACID, isolation levels |
| [Query Optimization](query-optimization.md) | 📝 Planned | EXPLAIN, query plans |

---

## Database Selection

**ACID Transactions Needed?**
- Yes → SQL (PostgreSQL, MySQL)
- No → NoSQL (flexibility + scale)

**Data Structure?**
- Structured, relational → SQL
- Documents, flexible → MongoDB
- Key-value pairs → Redis, DynamoDB
- Graph relationships → Neo4j
- Time-series → InfluxDB

**Scale Requirements?**
- <1M records → Any database works
- 1M-100M records → SQL with replication or NoSQL
- 100M-1B+ records → NoSQL with sharding

---

## Further Reading

**Related Topics:**
- [Caching](../caching/index.md) - Reduce database load
- [Scalability](../../scalability/index.md) - Scale databases
- [Data Consistency](../../fundamentals/data-consistency.md) - Consistency models

---

**Choose the right database for your needs! 💾**
