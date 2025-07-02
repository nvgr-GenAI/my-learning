# Reliability & Security 🛡️

Build resilient, fault-tolerant, and secure distributed systems that can handle failures gracefully and protect against threats.

## 🎯 Learning Objectives

- **Reliability**: Design systems that continue working despite failures
- **Fault Tolerance**: Handle and recover from various failure modes
- **Security**: Protect systems from threats and vulnerabilities
- **Monitoring**: Observe system health and performance
- **Disaster Recovery**: Plan for and recover from catastrophic failures

## 📚 Topics Overview

### 🔧 Reliability Patterns

| Pattern | Purpose | Use Cases |
|---------|---------|-----------|
| **Circuit Breaker** | Prevent cascade failures | Service dependencies |
| **Bulkhead** | Isolate critical resources | Resource protection |
| **Timeout & Retry** | Handle transient failures | Network calls |
| **Graceful Degradation** | Maintain core functionality | Service unavailability |

### 🛡️ Security Fundamentals

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Authentication** | Verify identity | OAuth, JWT, SAML |
| **Authorization** | Control access | RBAC, ABAC |
| **Encryption** | Protect data | TLS, AES, RSA |
| **Rate Limiting** | Prevent abuse | Token bucket, sliding window |

## 🗂️ Section Navigation

### Core Reliability
- [Fault Tolerance Patterns](fault-tolerance.md)
- [Circuit Breaker Pattern](circuit-breaker.md)
- [Retry Strategies](retry-strategies.md)
- [Graceful Degradation](graceful-degradation.md)
- [Chaos Engineering](chaos-engineering.md)

### Security Architecture
- [Authentication Systems](authentication.md)
- [Authorization Patterns](authorization.md)
- [Data Protection](data-protection.md)
- [API Security](api-security.md)
- [Security Threats](security-threats.md)

### Monitoring & Observability
- [Monitoring Strategies](monitoring.md)
- [Logging Best Practices](logging.md)
- [Metrics & Alerting](metrics-alerting.md)
- [Distributed Tracing](distributed-tracing.md)
- [Health Checks](health-checks.md)

### Disaster Recovery
- [Backup Strategies](backup-strategies.md)
- [Disaster Recovery Planning](disaster-recovery.md)
- [Business Continuity](business-continuity.md)
- [Incident Response](incident-response.md)

## 🔧 Reliability Patterns

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
```

### Retry with Exponential Backoff

```python
import time
import random

def retry_with_backoff(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt)
            jitter = random.uniform(0, delay * 0.1)
            time.sleep(delay + jitter)
```

## 🛡️ Security Implementations

### JWT Authentication

```python
import jwt
import datetime

class JWTAuth:
    def __init__(self, secret_key):
        self.secret_key = secret_key
    
    def generate_token(self, user_id, expires_in_hours=24):
        payload = {
            'user_id': user_id,
            'exp': datetime.datetime.utcnow() + 
                   datetime.timedelta(hours=expires_in_hours),
            'iat': datetime.datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload['user_id']
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")
```

### Rate Limiting

```python
import time
from collections import defaultdict

class TokenBucketRateLimiter:
    def __init__(self, capacity=10, refill_rate=1):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets = defaultdict(lambda: {
            'tokens': capacity,
            'last_refill': time.time()
        })
    
    def is_allowed(self, key):
        bucket = self.buckets[key]
        now = time.time()
        
        # Refill tokens
        time_passed = now - bucket['last_refill']
        tokens_to_add = time_passed * self.refill_rate
        bucket['tokens'] = min(self.capacity, 
                              bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now
        
        # Check if request is allowed
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        return False
```

## 📊 Monitoring & Observability

### Health Check Implementation

```python
from flask import Flask, jsonify
import psutil
import requests

app = Flask(__name__)

class HealthChecker:
    def __init__(self):
        self.dependencies = []
    
    def add_dependency(self, name, check_func):
        self.dependencies.append((name, check_func))
    
    def check_system_health(self):
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        # System metrics
        health_status['checks']['cpu'] = {
            'status': 'healthy' if psutil.cpu_percent() < 80 else 'warning',
            'value': psutil.cpu_percent()
        }
        
        health_status['checks']['memory'] = {
            'status': 'healthy' if psutil.virtual_memory().percent < 85 else 'warning',
            'value': psutil.virtual_memory().percent
        }
        
        # Custom dependencies
        for name, check_func in self.dependencies:
            try:
                result = check_func()
                health_status['checks'][name] = {
                    'status': 'healthy',
                    'details': result
                }
            except Exception as e:
                health_status['checks'][name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['status'] = 'degraded'
        
        return health_status

@app.route('/health')
def health_check():
    checker = HealthChecker()
    return jsonify(checker.check_system_health())
```

## 🎯 Best Practices

### Reliability
- **Design for Failure**: Assume components will fail
- **Graceful Degradation**: Maintain core functionality
- **Idempotency**: Make operations repeatable safely
- **Timeouts**: Set appropriate timeouts for all operations
- **Bulkheads**: Isolate critical resources

### Security
- **Defense in Depth**: Multiple layers of security
- **Least Privilege**: Minimal necessary permissions
- **Input Validation**: Validate all user inputs
- **Encryption**: Encrypt data at rest and in transit
- **Regular Updates**: Keep dependencies updated

### Monitoring
- **Golden Signals**: Latency, traffic, errors, saturation
- **Proactive Alerting**: Alert before issues impact users
- **Distributed Tracing**: Track requests across services
- **Log Aggregation**: Centralize logs for analysis
- **SLA/SLO Monitoring**: Track service level objectives

## 🔗 Related Topics

- [Scalability Patterns](../scalability/index.md)
- [Distributed Systems](../distributed-systems/index.md)
- [Performance Optimization](../performance/index.md)
- [System Design Interviews](../interviews/index.md)

## 📖 Further Reading

- [Site Reliability Engineering](https://sre.google/books/)
- [Designing Data-Intensive Applications](https://dataintensive.net/)
- [Building Security In](https://www.us-cert.gov/bsi)
- [The Phoenix Project](https://itrevolution.com/the-phoenix-project/)
