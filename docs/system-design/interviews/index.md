# System Design Interviews 🎯

Master the art of system design interviews with structured approaches, common questions, and proven frameworks.

## 🎯 Learning Objectives

- Master the system design interview framework
- Learn to handle common system design questions
- Understand how to communicate design decisions effectively
- Practice estimation and capacity planning
- Build confidence for technical interviews

## 📚 Topics Overview

<div class="grid cards" markdown>

-   :material-clipboard-text: **Interview Framework**
    
    ---
    
    Step-by-step approach to tackle any system design question
    
    [Learn framework →](framework.md)

-   :material-frequently-asked-questions: **Common Questions**
    
    ---
    
    Popular system design interview questions with solutions
    
    [Practice questions →](common-questions.md)

-   :material-calculator: **Estimation Techniques**
    
    ---
    
    Back-of-envelope calculations and capacity planning
    
    [Master estimation →](estimation.md)

-   :material-message-text: **Communication Skills**
    
    ---
    
    How to present your design and handle follow-up questions
    
    [Improve communication →](communication.md)

-   :material-target: **Company-Specific Tips**
    
    ---
    
    FAANG and other top companies' interview patterns
    
    [Company insights →](company-specific.md)

-   :material-school: **Practice & Mock Interviews**
    
    ---
    
    Practice problems and mock interview resources
    
    [Practice now →](practice.md)

</div>

## 🏗️ System Design Interview Framework

### The RADOS Framework

**R** - Requirements Clarification
**A** - Architecture Design  
**D** - Deep Dive
**O** - Optimization
**S** - Summary

```python
class SystemDesignFramework:
    """Step-by-step framework for system design interviews"""
    
    def __init__(self, problem_statement):
        self.problem = problem_statement
        self.requirements = {}
        self.architecture = {}
        self.estimates = {}
    
    def step_1_requirements(self):
        """Clarify functional and non-functional requirements"""
        return {
            'functional': [
                'What are the core features?',
                'Who are the users?',
                'What are the main use cases?',
                'What are the inputs and outputs?'
            ],
            'non_functional': [
                'How many users do we expect?',
                'What is the expected read/write ratio?',
                'What are the latency requirements?',
                'What are the availability requirements?'
            ],
            'constraints': [
                'Any specific technology constraints?',
                'Budget limitations?',
                'Timeline constraints?'
            ]
        }
    
    def step_2_estimation(self):
        """Back-of-envelope calculations"""
        return {
            'traffic': 'Calculate QPS, peak traffic',
            'storage': 'Estimate data storage needs',
            'bandwidth': 'Network bandwidth requirements',
            'memory': 'Caching and memory needs'
        }
    
    def step_3_high_level_design(self):
        """Create high-level system architecture"""
        return {
            'components': [
                'Load Balancers',
                'Web Servers',
                'Application Servers', 
                'Databases',
                'Caches',
                'CDN'
            ],
            'data_flow': 'How data flows through the system',
            'apis': 'Key API endpoints'
        }
    
    def step_4_detailed_design(self):
        """Deep dive into specific components"""
        return {
            'database_design': 'Schema, indexing, partitioning',
            'scalability': 'How to handle growth',
            'reliability': 'Fault tolerance, redundancy',
            'security': 'Authentication, authorization'
        }
    
    def step_5_scale_and_optimize(self):
        """Address bottlenecks and optimize"""
        return {
            'bottlenecks': 'Identify potential bottlenecks',
            'solutions': 'Scaling strategies',
            'trade_offs': 'Discuss trade-offs made'
        }
```

## 📊 Estimation Cheat Sheet

### Common Numbers Every Programmer Should Know

```python
class SystemEstimationHelper:
    """Common numbers and estimation techniques"""
    
    # Latency numbers
    L1_CACHE_REFERENCE = 0.5  # ns
    BRANCH_MISPREDICT = 5  # ns
    L2_CACHE_REFERENCE = 7  # ns
    MUTEX_LOCK_UNLOCK = 25  # ns
    MAIN_MEMORY_REFERENCE = 100  # ns
    COMPRESS_1K_WITH_SNAPPY = 3000  # ns
    SEND_1K_OVER_1GBPS_NETWORK = 10000  # ns
    READ_4K_FROM_SSD = 150000  # ns
    READ_1MB_SEQUENTIALLY_FROM_MEMORY = 250000  # ns
    ROUND_TRIP_WITHIN_DATACENTER = 500000  # ns
    READ_1MB_SEQUENTIALLY_FROM_SSD = 1000000  # ns (1ms)
    DISK_SEEK = 10000000  # ns (10ms)
    READ_1MB_SEQUENTIALLY_FROM_DISK = 20000000  # ns (20ms)
    SEND_PACKET_CA_TO_NETHERLANDS = 150000000  # ns (150ms)
    
    # Storage and bandwidth
    SECONDS_PER_DAY = 86400
    BYTES_PER_KB = 1024
    BYTES_PER_MB = 1024 * 1024
    BYTES_PER_GB = 1024 * 1024 * 1024
    
    def calculate_qps(self, daily_requests):
        """Calculate queries per second"""
        avg_qps = daily_requests / self.SECONDS_PER_DAY
        peak_qps = avg_qps * 2  # Assume 2x peak traffic
        return {
            'average_qps': avg_qps,
            'peak_qps': peak_qps
        }
    
    def calculate_storage(self, items_per_day, item_size_bytes, retention_years=5):
        """Calculate storage requirements"""
        daily_storage = items_per_day * item_size_bytes
        yearly_storage = daily_storage * 365
        total_storage = yearly_storage * retention_years
        
        return {
            'daily_gb': daily_storage / self.BYTES_PER_GB,
            'yearly_gb': yearly_storage / self.BYTES_PER_GB,
            'total_gb': total_storage / self.BYTES_PER_GB,
            'total_tb': total_storage / (self.BYTES_PER_GB * 1024)
        }
    
    def calculate_bandwidth(self, qps, avg_response_size_kb):
        """Calculate bandwidth requirements"""
        bytes_per_second = qps * avg_response_size_kb * self.BYTES_PER_KB
        
        return {
            'bytes_per_second': bytes_per_second,
            'mb_per_second': bytes_per_second / self.BYTES_PER_MB,
            'gb_per_second': bytes_per_second / self.BYTES_PER_GB
        }

# Example usage for Twitter-like system
estimator = SystemEstimationHelper()

# Assumptions
daily_tweets = 500_000_000  # 500M tweets per day
avg_tweet_size = 280  # bytes
daily_active_users = 200_000_000  # 200M DAU
read_write_ratio = 100  # 100:1 read to write ratio

# Calculations
write_qps = estimator.calculate_qps(daily_tweets)
read_qps = estimator.calculate_qps(daily_tweets * read_write_ratio)
storage = estimator.calculate_storage(daily_tweets, avg_tweet_size)
bandwidth = estimator.calculate_bandwidth(read_qps['peak_qps'], 1)  # 1KB avg response
```

### Estimation Templates

#### Social Media Platform (Twitter/Instagram)
```python
class SocialMediaEstimation:
    def estimate_twitter_like_system(self):
        # User base
        total_users = 1_000_000_000  # 1B total users
        daily_active_users = 200_000_000  # 200M DAU
        
        # Content generation
        tweets_per_day = 500_000_000  # 500M tweets/day
        avg_tweet_size = 280  # bytes
        photos_per_day = 200_000_000  # 200M photos/day
        avg_photo_size = 2 * 1024 * 1024  # 2MB
        
        # Read patterns
        timeline_requests_per_user = 50  # per day
        total_timeline_requests = daily_active_users * timeline_requests_per_user
        
        # QPS calculations
        tweet_write_qps = tweets_per_day / 86400  # ~5,787 QPS
        timeline_read_qps = total_timeline_requests / 86400  # ~115,741 QPS
        
        # Storage calculations
        text_storage_per_year = tweets_per_day * 365 * avg_tweet_size
        photo_storage_per_year = photos_per_day * 365 * avg_photo_size
        
        return {
            'write_qps': tweet_write_qps,
            'read_qps': timeline_read_qps,
            'text_storage_gb_per_year': text_storage_per_year / (1024**3),
            'photo_storage_tb_per_year': photo_storage_per_year / (1024**4)
        }
```

#### Chat Application (WhatsApp/Slack)
```python
class ChatApplicationEstimation:
    def estimate_whatsapp_like_system(self):
        # User base
        total_users = 2_000_000_000  # 2B users
        daily_active_users = 500_000_000  # 500M DAU
        
        # Message patterns
        messages_per_user_per_day = 40
        total_messages_per_day = daily_active_users * messages_per_user_per_day
        avg_message_size = 100  # bytes
        
        # Media sharing
        media_messages_percent = 10  # 10% of messages contain media
        avg_media_size = 5 * 1024 * 1024  # 5MB
        
        # QPS calculations
        message_qps = total_messages_per_day / 86400
        
        # Storage calculations
        text_storage_per_day = total_messages_per_day * avg_message_size
        media_messages_per_day = total_messages_per_day * (media_messages_percent / 100)
        media_storage_per_day = media_messages_per_day * avg_media_size
        
        return {
            'message_qps': message_qps,
            'text_storage_gb_per_day': text_storage_per_day / (1024**3),
            'media_storage_gb_per_day': media_storage_per_day / (1024**3)
        }
```

#### Video Streaming (YouTube/Netflix)
```python
class VideoStreamingEstimation:
    def estimate_youtube_like_system(self):
        # User base
        total_users = 2_000_000_000  # 2B users
        daily_active_users = 30_000_000  # 30M DAU
        
        # Video consumption
        videos_watched_per_user = 5  # per day
        avg_video_length_minutes = 10
        video_bitrate_mbps = 1  # 1 Mbps average
        
        # Video upload
        videos_uploaded_per_day = 500_000  # 500K videos/day
        avg_upload_size_gb = 1  # 1GB raw video
        
        # Calculations
        total_watch_time_minutes = daily_active_users * videos_watched_per_user * avg_video_length_minutes
        bandwidth_requirement_gbps = (total_watch_time_minutes * video_bitrate_mbps) / (60 * 8)  # Convert to Gbps
        
        storage_per_day_tb = videos_uploaded_per_day * avg_upload_size_gb / 1024
        
        return {
            'total_watch_minutes_per_day': total_watch_time_minutes,
            'bandwidth_gbps': bandwidth_requirement_gbps,
            'storage_tb_per_day': storage_per_day_tb
        }
```

## 🗣️ Communication Framework

### How to Present Your Design

```python
class CommunicationStrategy:
    """Framework for effective communication during interviews"""
    
    def opening_approach(self):
        """How to start the interview"""
        return [
            "Thank you for the problem. Let me start by asking a few clarifying questions.",
            "I want to make sure I understand the requirements correctly.",
            "Should I assume this is similar to [existing system]?",
            "What scale are we designing for?"
        ]
    
    def requirement_clarification_questions(self):
        """Questions to ask during requirements phase"""
        return {
            'functional': [
                "What are the core features we need to support?",
                "Who are the primary users?",
                "What are the main user flows?",
                "Are there any specific constraints or assumptions?"
            ],
            'scale': [
                "How many users do we expect?",
                "What's the expected read/write ratio?",
                "How much data will we store?",
                "What are the latency requirements?"
            ],
            'technical': [
                "Are there any preferred technologies?",
                "Should we focus on any specific aspects?",
                "What's the budget/timeline?"
            ]
        }
    
    def design_presentation_flow(self):
        """How to present your design"""
        return [
            "Based on our requirements, here's my high-level approach...",
            "Let me start with the overall architecture and then dive deeper.",
            "The main components will be...",
            "For the data flow, requests will go through...",
            "Let me explain why I chose this approach...",
            "The trade-offs here are..."
        ]
    
    def handling_follow_up_questions(self):
        """How to handle interviewer questions"""
        return {
            'clarification': "That's a great question. Let me clarify...",
            'alternative': "You're right, another approach would be...",
            'trade_off': "The trade-off between these options is...",
            'scaling': "If we need to scale this further, we could...",
            'improvement': "A potential optimization would be..."
        }
```

## 📋 Common System Design Questions

### Tier 1: Fundamental Systems
1. **URL Shortener** (bit.ly, tinyurl)
2. **Pastebin** (pastebin.com)
3. **Chat System** (WhatsApp, Slack)
4. **Notification System** (Push notifications)

### Tier 2: Intermediate Systems
5. **News Feed** (Facebook, Twitter)
6. **Rate Limiter** (API rate limiting)
7. **Key-Value Store** (Redis, DynamoDB)
8. **Unique ID Generator** (Distributed system)

### Tier 3: Advanced Systems
9. **Web Crawler** (Google crawler)
10. **Search Engine** (Google, Elasticsearch)
11. **Video Streaming** (YouTube, Netflix)
12. **Ride Sharing** (Uber, Lyft)

### Example: URL Shortener Design

```python
class URLShortenerDesign:
    """Complete URL shortener system design"""
    
    def __init__(self):
        self.requirements = self.define_requirements()
        self.estimates = self.calculate_estimates()
        self.architecture = self.design_architecture()
    
    def define_requirements(self):
        """Step 1: Requirements clarification"""
        return {
            'functional': [
                'Shorten long URLs',
                'Redirect to original URL',
                'Custom short URLs (optional)',
                'URL expiration',
                'Analytics (click tracking)'
            ],
            'non_functional': [
                '100M URLs shortened per day',
                '100:1 read/write ratio',
                'Low latency < 100ms',
                '99.9% availability',
                '5 years data retention'
            ]
        }
    
    def calculate_estimates(self):
        """Step 2: Capacity estimation"""
        # Write operations
        write_qps = 100_000_000 / 86400  # ~1160 QPS
        
        # Read operations  
        read_qps = write_qps * 100  # ~116,000 QPS
        
        # Storage (5 years)
        urls_per_year = 100_000_000 * 365
        total_urls = urls_per_year * 5  # 182.5B URLs
        
        # Assuming 500 bytes per URL object
        storage_gb = (total_urls * 500) / (1024**3)  # ~85TB
        
        return {
            'write_qps': write_qps,
            'read_qps': read_qps,
            'storage_gb': storage_gb
        }
    
    def design_architecture(self):
        """Step 3: High-level design"""
        return {
            'components': [
                'Load Balancer',
                'Web Servers',
                'URL Shortening Service',
                'URL Redirect Service', 
                'Cache (Redis)',
                'Database (MySQL/PostgreSQL)',
                'Analytics Service',
                'CDN'
            ],
            'apis': {
                'shorten_url': 'POST /api/v1/shorten',
                'redirect_url': 'GET /{short_url}',
                'get_analytics': 'GET /api/v1/analytics/{short_url}'
            }
        }
    
    def detailed_design(self):
        """Step 4: Detailed component design"""
        return {
            'url_encoding': self.design_url_encoding(),
            'database_schema': self.design_database(),
            'caching_strategy': self.design_caching(),
            'scalability': self.design_scalability()
        }
    
    def design_url_encoding(self):
        """URL encoding algorithm"""
        return {
            'approach': 'Base62 encoding',
            'characters': 'a-z, A-Z, 0-9 (62 characters)',
            'short_url_length': 7,  # 62^7 = 3.5 trillion combinations
            'algorithm': 'Hash function + Base62 encoding',
            'collision_handling': 'Increment counter and retry'
        }
    
    def design_database(self):
        """Database schema design"""
        return {
            'url_table': {
                'id': 'BIGINT PRIMARY KEY',
                'long_url': 'VARCHAR(2048)',
                'short_url': 'VARCHAR(16) UNIQUE',
                'user_id': 'BIGINT',
                'created_at': 'TIMESTAMP',
                'expires_at': 'TIMESTAMP'
            },
            'analytics_table': {
                'id': 'BIGINT PRIMARY KEY',
                'short_url': 'VARCHAR(16)',
                'click_timestamp': 'TIMESTAMP',
                'user_agent': 'VARCHAR(512)',
                'ip_address': 'VARCHAR(45)'
            },
            'indexing': [
                'Index on short_url for fast lookups',
                'Index on user_id for user queries',
                'Index on created_at for cleanup'
            ]
        }
    
    def design_caching(self):
        """Caching strategy"""
        return {
            'cache_type': 'Redis',
            'cache_policy': 'LRU eviction',
            'ttl': '1 hour for popular URLs',
            'cache_size': '20% of daily reads (hot data)',
            'cache_pattern': 'Cache-aside',
            'cache_key': 'url:{short_url}'
        }
    
    def design_scalability(self):
        """Scalability considerations"""
        return {
            'database_scaling': [
                'Read replicas for read-heavy workload',
                'Vertical scaling initially',
                'Horizontal sharding if needed (by hash of short_url)'
            ],
            'application_scaling': [
                'Stateless web servers',
                'Load balancer with health checks',
                'Auto-scaling based on CPU/memory'
            ],
            'caching_scaling': [
                'Redis cluster for distributed caching',
                'Consistent hashing for cache partitioning'
            ]
        }
```

## 🎯 Interview Tips

### Do's
- ✅ Ask clarifying questions
- ✅ Start with high-level design
- ✅ Explain your thought process
- ✅ Discuss trade-offs
- ✅ Consider scalability from the start
- ✅ Think about failure scenarios
- ✅ Be open to feedback

### Don'ts  
- ❌ Jump into details immediately
- ❌ Design in silence
- ❌ Ignore non-functional requirements
- ❌ Over-engineer the solution
- ❌ Forget about data consistency
- ❌ Ignore monitoring and logging
- ❌ Be defensive about your design

### Time Management (45-60 minute interview)

| Phase | Time | Activities |
|-------|------|------------|
| **Requirements** | 5-10 min | Clarify functional and non-functional requirements |
| **Estimation** | 5-10 min | Back-of-envelope calculations |
| **High-level Design** | 10-15 min | Overall architecture, main components |
| **Detailed Design** | 15-20 min | Deep dive into specific components |
| **Scale & Optimize** | 5-10 min | Handle bottlenecks, optimization |
| **Q&A** | 5-10 min | Answer follow-up questions |

## 🏆 Success Stories Framework

### The STAR Method for Behavioral Questions

**S**ituation - Set the context
**T**ask - Describe what needed to be done  
**A**ction - Explain what you did
**R**esult - Share the outcome

Example for "Tell me about a time you designed a scalable system":

```
Situation: At my previous company, our e-commerce platform was struggling with Black Friday traffic spikes.

Task: I was tasked with redesigning the system to handle 10x normal traffic without downtime.

Action: I implemented a multi-tier caching strategy with Redis, added read replicas for the database, and introduced a CDN for static assets. I also implemented circuit breakers to prevent cascading failures.

Result: The system successfully handled Black Friday traffic with 99.99% uptime and 50% improvement in response times.
```

## 🚀 Next Steps

Ready to master system design interviews?

1. **[Learn the Framework](framework.md)** - Master the structured approach
2. **[Practice Common Questions](common-questions.md)** - Work through popular problems
3. **[Master Estimation](estimation.md)** - Perfect your calculation skills
4. **[Improve Communication](communication.md)** - Present designs effectively
5. **[Company-Specific Prep](company-specific.md)** - Target specific companies
6. **[Practice Mock Interviews](practice.md)** - Get real interview experience

---

**Ace your system design interviews! 🎯💪**
