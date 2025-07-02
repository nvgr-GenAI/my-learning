# System Design Learning Path 🗺️

Your complete roadmap from beginner to system design expert. This structured path takes you through essential concepts, hands-on practice, and real-world applications.

## 🎯 Choose Your Journey

=== "🚀 Quick Start (30 mins)"

    **Perfect for**: Beginners, interview prep, or getting a high-level overview
    
    Get the essential concepts to start thinking like a system architect:
    
    **1. Design Principles - The 5 Pillars** 🏛️
    
    - **Reliability**: System works correctly even when things go wrong
    - **Scalability**: Handles increased load gracefully  
    - **Availability**: Remains operational and accessible
    - **Consistency**: All nodes see the same data
    - **Performance**: Fast response times and high throughput
    
    **2. Trade-offs - Why There's No Perfect Solution** ⚖️
    
    - **CAP Theorem**: You can't have Consistency, Availability, and Partition tolerance all at once
    - **Performance vs Cost**: Faster systems require more resources
    - **Simplicity vs Flexibility**: Simple systems are easier to manage but harder to extend
    
    **3. Scaling Basics - From 1 to 1 Million Users** 📈
    
    - **Vertical Scaling**: Add more power to existing machines (CPU, RAM)
    - **Horizontal Scaling**: Add more machines to your resource pool
    - **Load Balancing**: Distribute requests across multiple servers
    - **Caching**: Store frequently accessed data in fast storage
    
    **Quick Win**: After 30 minutes, you'll understand why Instagram chose simple architecture initially and how they scaled to billions of users.

=== "📚 Deep Dive (2-3 hours)"

    **Perfect for**: Developers ready to design production systems, senior engineers
    
    Build comprehensive understanding with hands-on practice:
    
    **4. Architecture Patterns - Monoliths vs Microservices** 🏗️
    
    **Monolithic Architecture:**
    
    - Single deployable unit with all functionality
    - **Best for**: Small teams (2-8 people), simple domains, MVP stage
    - **Examples**: Early Twitter, Instagram, GitHub started as monoliths
    - **When to choose**: Fast development, simple deployment, consistent data
    
    **Microservices Architecture:**
    
    - Multiple small services that communicate over network
    - **Best for**: Large teams (50+ engineers), complex domains, mature products
    - **Examples**: Netflix (600+ services), Amazon, Uber
    - **When to choose**: Independent scaling, technology diversity, team autonomy
    
    **5. Performance - Latency, Throughput, Optimization** 🚀
    
    **Key Metrics:**
    
    - **Latency**: Time to process single request (<100ms for web apps)
    - **Throughput**: Requests processed per second (1000+ RPS for popular apps)
    - **Percentiles**: 95th percentile response time matters more than average
    
    **Optimization Strategies:**
    
    - **Database Indexing**: Speed up queries by 10-100x
    - **Connection Pooling**: Reuse database connections
    - **Async Processing**: Handle heavy tasks in background
    - **CDN**: Serve static content from locations close to users
    
    **6. Data Strategy - Storage, Consistency, ACID vs BASE** 💾
    
    **SQL Databases (ACID):**
    
    - **Use for**: Financial transactions, user accounts, inventory
    - **Examples**: PostgreSQL, MySQL, SQL Server
    - **Guarantees**: Atomicity, Consistency, Isolation, Durability
    
    **NoSQL Databases (BASE):**
    
    - **Use for**: Social feeds, analytics, logs, caching
    - **Examples**: MongoDB (documents), Redis (key-value), Cassandra (wide-column)
    - **Trade-off**: Eventually consistent but highly available and scalable
    
    **Real-World Application**: Design a complete e-commerce system with user service (SQL), product catalog (NoSQL), and order processing (message queues).

=== "🛠️ Hands-on (1-2 weeks)"

    **Perfect for**: Experienced developers, system architects, interview preparation
    
    Apply your knowledge with real-world system design challenges:
    
    **7. Practice Problems - Build Real Systems** 🎯
    
    **URL Shortener (like bit.ly):**
    
    - **Challenge**: Handle 100M URLs, 10:1 read/write ratio
    - **Learn**: Database design, caching strategies, API design
    - **Key decisions**: How to generate short URLs? SQL vs NoSQL? Caching strategy?
    
    **Chat System (like WhatsApp basics):**
    
    - **Challenge**: Real-time messaging, online presence, message history
    - **Learn**: WebSockets, message queues, data partitioning
    - **Key decisions**: How to handle message delivery? Store message history? Scale to millions of users?
    
    **Social Feed (like Twitter timeline):**
    
    - **Challenge**: Generate personalized feeds for millions of users in real-time
    - **Learn**: Fan-out strategies, timeline generation, content ranking
    - **Key decisions**: Push vs pull model? How to handle celebrity users? Cache invalidation?
    
    **8. Case Studies - How Big Tech Solves Real Problems** 🏢
    
    **Netflix Streaming Architecture:**
    
    - **Challenge**: Stream video to 200M+ users globally
    - **Solutions**: Microservices (600+), AWS cloud, CDN, chaos engineering
    - **Lessons**: How to handle massive scale, global distribution, fault tolerance
    
    **Instagram Photo Sharing:**
    
    - **Challenge**: Store and serve billions of photos efficiently
    - **Solutions**: Sharded MySQL, S3 storage, CDN, simple Python backend
    - **Lessons**: Start simple, scale gradually, focus on user experience
    
    **Uber Real-time Matching:**
    
    - **Challenge**: Match riders with drivers in real-time across the globe
    - **Solutions**: Geospatial indexing, event streaming, machine learning
    - **Lessons**: Handle real-time data, geographic challenges, predictive systems
    
    **9. Interview Prep - System Design for Technical Interviews** 💼
    
    **Common Interview Questions:**
    
    - Design a URL shortener (45 minutes)
    - Design a chat system (45 minutes)  
    - Design Instagram/Twitter feed (45 minutes)
    - Design Uber/ride-sharing system (45 minutes)
    
    **Interview Framework:**
    
    1. **Clarify Requirements** (5 min): Ask about scale, features, constraints
    2. **High-level Design** (10 min): Draw main components and data flow
    3. **Detailed Design** (20 min): Deep dive into specific components
    4. **Scale the Design** (10 min): Handle bottlenecks and scaling challenges
    
    **Success Tips:**
    
    - Start simple and iterate
    - Think out loud and explain your reasoning
    - Consider trade-offs and alternatives
    - Ask questions throughout the process
    
    **Practice Schedule**: 1 system design problem every 2 days, focusing on different domains (social media, e-commerce, real-time systems, data processing).

## 🚀 30-Day Structured Learning Plan

### Week 1: Core Foundations

**Goal**: Build fundamental understanding of system design principles

**Day 1-2: Design Principles**

- [ ] Read [Fundamentals](fundamentals/index.md) - Focus on the 5 pillars
- [ ] Study reliability patterns (redundancy, failover)
- [ ] Practice: Identify single points of failure in a simple web app

**Day 3-4: Trade-offs & CAP Theorem**

- [ ] Deep dive into CAP theorem with real examples
- [ ] Compare consistency models (strong vs eventual)
- [ ] Practice: Design a banking system vs social media feed

**Day 5-7: Scaling Basics**

- [ ] Learn vertical vs horizontal scaling patterns
- [ ] Study load balancing strategies
- [ ] Practice: Scale a web app from 1K to 100K users

### Week 2: Architecture & Performance

**Goal**: Master architecture patterns and performance optimization

**Day 8-10: Architecture Patterns**

- [ ] Compare monolith vs microservices architectures
- [ ] Study service communication patterns
- [ ] Practice: Convert a monolith to microservices

**Day 11-13: Performance Optimization**

- [ ] Learn caching strategies (browser, CDN, application, database)
- [ ] Study database optimization (indexing, query optimization)
- [ ] Practice: Optimize a slow web application

**Day 14: Data Strategy**

- [ ] Compare SQL vs NoSQL databases
- [ ] Learn data modeling for different use cases
- [ ] Practice: Choose databases for an e-commerce system

### Week 3: Real-World Systems

**Goal**: Apply knowledge through practical system design problems

**Day 15-17: URL Shortener System**

- [ ] Design bit.ly from scratch
- [ ] Handle 100M URLs with 10:1 read/write ratio
- [ ] Focus: Database design, caching, API design

**Day 18-20: Chat System**

- [ ] Design WhatsApp-like messaging system
- [ ] Handle real-time messaging and presence
- [ ] Focus: WebSockets, message queues, data partitioning

**Day 21: Social Media Feed**

- [ ] Design Twitter timeline system
- [ ] Handle personalized feeds for millions of users
- [ ] Focus: Fan-out strategies, cache invalidation

### Week 4: Advanced Topics & Interview Prep

**Goal**: Master advanced concepts and interview techniques

**Day 22-24: Distributed Systems**

- [ ] Study consensus algorithms (Raft, Paxos basics)
- [ ] Learn about event streaming and message queues
- [ ] Practice: Design a distributed database

**Day 25-27: Case Studies**

- [ ] Study Netflix architecture (microservices, chaos engineering)
- [ ] Analyze Instagram's evolution (simple to complex)
- [ ] Learn from Uber's real-time matching system

**Day 28-30: Interview Practice**

- [ ] Practice system design interviews with friends
- [ ] Focus on communication and trade-off discussions
- [ ] Review and strengthen weak areas

## 📚 Recommended Reading Order

### Core Resources

1. **Start Here**: [System Design Fundamentals](fundamentals/index.md)
2. **Deep Dive**: [Scalability Patterns](scalability/index.md)
3. **Data Focus**: [Data & Storage](data-storage/index.md)
4. **Advanced**: [Distributed Systems](distributed-systems/index.md)

### Supplementary Materials

- **Real Examples**: [Case Studies](case-studies/index.md)
- **Interview Ready**: [Interview Preparation](interviews/index.md)
- **Performance**: [Performance & Optimization](performance/index.md)
- **Infrastructure**: [Load Balancing](load-balancing/index.md), [Caching](caching/index.md)

## 🎯 Learning Milestones

### 🥉 Bronze Level (Week 1-2)
- Understand the 5 system design pillars
- Can explain CAP theorem with examples
- Know when to scale vertically vs horizontally
- Can design a simple 3-tier web application

### 🥈 Silver Level (Week 3)
- Can design URL shortener, chat system, social feed
- Understand microservices trade-offs
- Know caching strategies and when to use them
- Can handle system design problem constraints

### 🥇 Gold Level (Week 4)
- Can design complex distributed systems
- Understand consensus and consistency models
- Can discuss real-world architecture trade-offs
- Ready for senior-level system design interviews

## 🛠️ Hands-on Projects

Build these systems to reinforce your learning:

### Beginner Projects
1. **Personal Blog System** - Single server, database, basic caching
2. **Task Management API** - REST API, authentication, data persistence

### Intermediate Projects  
3. **URL Shortener** - High read/write ratio, caching, analytics
4. **Real-time Chat** - WebSockets, message persistence, user presence

### Advanced Projects
5. **Social Media Feed** - Complex algorithms, personalization, scale
6. **Video Streaming Platform** - CDN, encoding, global distribution

## 🔗 What's Next?

After completing this learning path:

- **Specialize**: Pick an area (distributed systems, data engineering, security)
- **Build**: Create your own large-scale project
- **Share**: Write blog posts about your learnings
- **Interview**: Apply for senior engineering roles
- **Teach**: Help others learn system design

---

**Remember**: System design is learned through practice. Don't just read—build, experiment, and learn from failures. Every senior engineer started exactly where you are now.
