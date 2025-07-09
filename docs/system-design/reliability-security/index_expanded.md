# Reliability & Security 🛡️

Build systems that remain operational and secure under adverse conditions. This comprehensive guide covers fault tolerance, monitoring, security patterns, and disaster recovery strategies with practical implementations.

## 🎯 What You'll Learn

Reliable and secure systems are essential for production environments. Learn how to design for failure, implement comprehensive security, and ensure your systems can recover from disasters.

> **Engineering Truth**: "Everything fails, all the time. The question is not if, but when, and how your system responds to failure."

<div class="grid cards" markdown>

- :material-shield-check: **Fault Tolerance**

    ---

    Graceful degradation, redundancy, and failure isolation patterns

    [Build resilient systems →](fault-tolerance.md)

- :material-monitor-dashboard: **Monitoring & Observability**

    ---

    Metrics, logging, tracing, and alerting strategies

    [Observe your systems →](monitoring.md)

- :material-security: **Security Patterns**

    ---

    Authentication, authorization, encryption, and security best practices

    [Secure your systems →](security.md)

- :material-backup-restore: **Disaster Recovery**

    ---

    Backup strategies, failover mechanisms, and business continuity

    [Plan for disasters →](disaster-recovery.md)

</div>

## 🛡️ Reliability Fundamentals

### **Understanding System Failures**

**Types of Failures**:

| Type | Description | Example | Impact |
|------|-------------|---------|---------|
| **Transient** | Temporary, self-correcting | Network blip, memory spike | Brief degradation |
| **Intermittent** | Sporadic, hard to reproduce | Race condition, timing issue | Unpredictable behavior |
| **Permanent** | Persistent until fixed | Hardware failure, software bug | Complete service failure |
| **Cascade** | One failure triggers others | Database overload causing timeouts | System-wide outage |

**Failure Modes**:

```text
Common Failure Patterns:
┌─────────────────────────────────────────────────────────────┐
│ Fail-Fast: Detect errors early and fail immediately        │
│ Fail-Safe: Default to safe state when error occurs         │
│ Fail-Secure: Maintain security even during failures        │
│ Fail-Operational: Continue operating with degraded service │
└─────────────────────────────────────────────────────────────┘
```

### **Fault Tolerance Patterns**

#### 1. **Circuit Breaker Pattern**

**States and Transitions**:

```text
Circuit Breaker States:
CLOSED (Normal) → OPEN (Failure) → HALF_OPEN (Testing) → CLOSED/OPEN
    ↓               ↓                    ↓
 Success       Failure Count      Test Request
 Continue      Exceeds Threshold  Determines State
```

**Implementation Strategy**:

```python
import time
import threading
from enum import Enum
from typing import Any, Callable, Dict, Optional

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN - requests failing fast")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        return (time.time() - self.last_failure_time) > self.recovery_timeout
```

#### 2. **Retry Strategies**

**Retry Types**:

| Strategy | Description | Use Case | Implementation |
|----------|-------------|----------|----------------|
| **Fixed Delay** | Constant wait between retries | Network timeouts | `time.sleep(1)` |
| **Exponential Backoff** | Increasing delay (2^n) | API rate limits | `delay = base * (2 ** attempt)` |
| **Linear Backoff** | Linearly increasing delay | Database locks | `delay = base * attempt` |
| **Random Jitter** | Random delay component | Avoid thundering herd | `delay + random(0, jitter)` |

**Advanced Retry Implementation**:

```python
import random
import time
from typing import Callable, Any, Type, Tuple
from functools import wraps

class RetryConfig:
    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter_factor: float = 0.1,
                 retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter_factor = jitter_factor
        self.retry_exceptions = retry_exceptions

def retry_with_backoff(config: RetryConfig):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retry_exceptions as e:
                    if attempt == config.max_retries:
                        raise
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    jitter = random.uniform(0, delay * config.jitter_factor)
                    total_delay = delay + jitter
                    
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {total_delay:.2f}s")
                    time.sleep(total_delay)
            
            return None
        return wrapper
    return decorator

# Usage example
@retry_with_backoff(RetryConfig(max_retries=3, base_delay=1.0))
def unreliable_api_call():
    # Simulate API call that might fail
    import requests
    return requests.get("https://api.example.com/data")
```

#### 3. **Bulkhead Pattern**

**Resource Isolation**:

```python
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

class ResourceBulkhead:
    def __init__(self, pools: Dict[str, int]):
        self.pools = {}
        for pool_name, pool_size in pools.items():
            self.pools[pool_name] = ThreadPoolExecutor(max_workers=pool_size)
    
    def execute(self, pool_name: str, func: Callable, *args, **kwargs):
        if pool_name not in self.pools:
            raise ValueError(f"Pool {pool_name} not found")
        
        pool = self.pools[pool_name]
        future = pool.submit(func, *args, **kwargs)
        return future.result()
    
    def shutdown(self):
        for pool in self.pools.values():
            pool.shutdown(wait=True)

# Usage
bulkhead = ResourceBulkhead({
    'database': 10,    # Database operations
    'external_api': 5,  # External API calls
    'file_io': 3       # File operations
})

# Different operations use different thread pools
def process_user_request(user_id):
    # Database operations isolated to database pool
    user_data = bulkhead.execute('database', get_user_data, user_id)
    
    # External API calls isolated to API pool
    preferences = bulkhead.execute('external_api', get_user_preferences, user_id)
    
    # File operations isolated to file pool
    bulkhead.execute('file_io', log_user_activity, user_id)
```

### **Reliability Metrics**

**Key Metrics**:

| Metric | Definition | Calculation | Industry Standard |
|--------|------------|-------------|-------------------|
| **Availability** | Percentage of time system is operational | `(Total time - Downtime) / Total time * 100` | 99.9% (8.76 hours/year) |
| **MTTR** | Mean Time To Recovery | `Total recovery time / Number of incidents` | < 1 hour |
| **MTBF** | Mean Time Between Failures | `Total operational time / Number of failures` | > 1000 hours |
| **RTO** | Recovery Time Objective | Maximum acceptable downtime | Business dependent |
| **RPO** | Recovery Point Objective | Maximum acceptable data loss | Business dependent |

**SLA Levels**:

```text
Availability Levels:
99.9% (8.76 hours/year)    - Standard web service
99.95% (4.38 hours/year)   - Enterprise application
99.99% (52.56 minutes/year) - Critical business system
99.999% (5.26 minutes/year) - High-availability system
```

## 🔒 Security Fundamentals

### **Security Principles**

#### **CIA Triad**

| Component | Definition | Implementation | Real-World Example |
|-----------|------------|----------------|-------------------|
| **Confidentiality** | Data accessible only to authorized users | Encryption, access controls | SSL/TLS, database encryption |
| **Integrity** | Data remains accurate and unmodified | Checksums, digital signatures | Hash functions, certificate validation |
| **Availability** | System accessible when needed | Redundancy, DDoS protection | Load balancers, CDN |

#### **Security Architecture (Defense in Depth)**

```text
Security Layers:
┌─────────────────────────────────────────────────────────────┐
│ Layer 7: Application Security                              │
│ ├─ Input validation, secure coding practices              │
│ ├─ SQL injection prevention, XSS protection               │
│ └─ Business logic security                                │
├─────────────────────────────────────────────────────────────┤
│ Layer 6: Authentication & Authorization                    │
│ ├─ Multi-factor authentication (MFA)                      │
│ ├─ OAuth 2.0, JWT tokens                                  │
│ └─ Role-based access control (RBAC)                       │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: Transport Security                               │
│ ├─ TLS/SSL encryption                                     │
│ ├─ API security (rate limiting, API keys)                │
│ └─ Certificate management                                 │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: Infrastructure Security                          │
│ ├─ Web Application Firewall (WAF)                        │
│ ├─ DDoS protection                                        │
│ └─ Security monitoring and logging                       │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Network Security                                 │
│ ├─ Firewall rules and network segmentation               │
│ ├─ VPN and private networks                              │
│ └─ Intrusion detection systems (IDS)                     │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Host Security                                    │
│ ├─ Operating system hardening                            │
│ ├─ Endpoint protection                                    │
│ └─ Patch management                                       │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Physical Security                                │
│ ├─ Data center security                                  │
│ ├─ Hardware security modules (HSM)                       │
│ └─ Secure disposal of hardware                           │
└─────────────────────────────────────────────────────────────┘
```

### **Common Security Threats and Mitigation**

#### **Web Application Threats**

| Threat | Description | Impact | Mitigation |
|--------|-------------|---------|------------|
| **SQL Injection** | Malicious SQL code injection | Data breach, database corruption | Parameterized queries, input validation |
| **Cross-Site Scripting (XSS)** | Malicious script injection | Session hijacking, data theft | Input sanitization, Content Security Policy |
| **Cross-Site Request Forgery (CSRF)** | Unauthorized actions on behalf of user | Unauthorized transactions | CSRF tokens, SameSite cookies |
| **Insecure Direct Object References** | Direct access to internal objects | Unauthorized data access | Access control checks, indirect references |
| **Security Misconfiguration** | Insecure default configurations | Various vulnerabilities | Security hardening, regular audits |

**SQL Injection Prevention**:

```python
import sqlite3
from typing import List, Dict, Any

class SecureDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_user_by_id(self, user_id: int) -> Dict[str, Any]:
        """Secure way to query user by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # SECURE: Using parameterized queries
            cursor.execute(
                "SELECT id, username, email FROM users WHERE id = ?",
                (user_id,)
            )
            
            result = cursor.fetchone()
            if result:
                return {
                    'id': result[0],
                    'username': result[1],
                    'email': result[2]
                }
            return None
    
    def search_users(self, search_term: str) -> List[Dict[str, Any]]:
        """Secure user search with input validation"""
        # Input validation
        if not search_term or len(search_term) < 2:
            return []
        
        # Sanitize input
        search_term = search_term.replace('%', '').replace('_', '')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # SECURE: Using parameterized queries with LIKE
            cursor.execute(
                "SELECT id, username, email FROM users WHERE username LIKE ?",
                (f"%{search_term}%",)
            )
            
            results = cursor.fetchall()
            return [
                {'id': row[0], 'username': row[1], 'email': row[2]}
                for row in results
            ]
```

**XSS Prevention**:

```python
import html
import re
from typing import Dict, Any

class XSSProtection:
    @staticmethod
    def sanitize_input(user_input: str) -> str:
        """Sanitize user input to prevent XSS attacks"""
        if not user_input:
            return ""
        
        # HTML escape
        sanitized = html.escape(user_input)
        
        # Remove potentially dangerous tags
        dangerous_tags = [
            '<script>', '</script>', '<iframe>', '</iframe>',
            '<object>', '</object>', '<embed>', '</embed>',
            '<link>', '<style>', '</style>'
        ]
        
        for tag in dangerous_tags:
            sanitized = re.sub(tag, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    @staticmethod
    def generate_csp_header() -> str:
        """Generate Content Security Policy header"""
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' 'unsafe-inline'; "
            "connect-src 'self'"
        )

# Usage in Flask
from flask import Flask, request, make_response

app = Flask(__name__)
xss_protection = XSSProtection()

@app.route('/comment', methods=['POST'])
def post_comment():
    comment = request.form.get('comment', '')
    
    # Sanitize input
    safe_comment = xss_protection.sanitize_input(comment)
    
    # Store safe comment
    # ... save to database
    
    response = make_response("Comment posted")
    response.headers['Content-Security-Policy'] = xss_protection.generate_csp_header()
    return response
```

### **Authentication and Authorization**

#### **Multi-Factor Authentication (MFA)**

```python
import pyotp
import qrcode
from typing import Dict, Any
import secrets
import time

class MFAManager:
    def __init__(self):
        self.backup_codes = {}
        self.failed_attempts = {}
        self.rate_limit = 5  # max attempts per minute
    
    def generate_secret(self, user_id: str) -> str:
        """Generate TOTP secret for user"""
        secret = pyotp.random_base32()
        return secret
    
    def generate_qr_code(self, user_email: str, secret: str) -> str:
        """Generate QR code for authenticator app"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name="Your App Name"
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        return totp_uri
    
    def verify_totp(self, user_id: str, secret: str, token: str) -> bool:
        """Verify TOTP token"""
        # Rate limiting
        if not self._check_rate_limit(user_id):
            return False
        
        totp = pyotp.TOTP(secret)
        
        # Allow for time drift (±30 seconds)
        for time_offset in [-1, 0, 1]:
            if totp.verify(token, for_time=time.time() + (time_offset * 30)):
                self._reset_failed_attempts(user_id)
                return True
        
        self._record_failed_attempt(user_id)
        return False
    
    def generate_backup_codes(self, user_id: str) -> List[str]:
        """Generate backup codes for account recovery"""
        codes = []
        for _ in range(10):
            code = secrets.token_hex(4).upper()
            codes.append(code)
        
        self.backup_codes[user_id] = codes
        return codes
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit"""
        now = time.time()
        attempts = self.failed_attempts.get(user_id, [])
        
        # Remove attempts older than 1 minute
        recent_attempts = [attempt for attempt in attempts if now - attempt < 60]
        self.failed_attempts[user_id] = recent_attempts
        
        return len(recent_attempts) < self.rate_limit
    
    def _record_failed_attempt(self, user_id: str):
        """Record failed authentication attempt"""
        now = time.time()
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        self.failed_attempts[user_id].append(now)
    
    def _reset_failed_attempts(self, user_id: str):
        """Reset failed attempts after successful authentication"""
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
```

#### **JWT Token Implementation**

```python
import jwt
import datetime
from typing import Dict, Any, Optional
import secrets

class JWTManager:
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.blacklisted_tokens = set()
    
    def generate_token(self, 
                      user_id: str, 
                      role: str = 'user',
                      expires_in_hours: int = 24) -> str:
        """Generate JWT token for user"""
        now = datetime.datetime.utcnow()
        
        payload = {
            'user_id': user_id,
            'role': role,
            'iat': now,
            'exp': now + datetime.timedelta(hours=expires_in_hours),
            'jti': secrets.token_hex(16)  # JWT ID for blacklisting
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise jwt.InvalidTokenError("Token has been revoked")
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")
    
    def refresh_token(self, token: str) -> str:
        """Refresh JWT token"""
        payload = self.verify_token(token)
        
        # Generate new token with same claims
        new_token = self.generate_token(
            user_id=payload['user_id'],
            role=payload['role']
        )
        
        # Blacklist old token
        self.blacklist_token(token)
        
        return new_token
    
    def blacklist_token(self, token: str):
        """Add token to blacklist"""
        self.blacklisted_tokens.add(token)
```

### **Rate Limiting and DDoS Protection**

#### **Advanced Rate Limiting**

```python
import time
import redis
from typing import Dict, Any, Optional
from collections import defaultdict

class RateLimiter:
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.local_buckets = defaultdict(dict)
    
    def sliding_window_log(self, 
                          key: str, 
                          limit: int, 
                          window_size: int = 60) -> bool:
        """Sliding window log rate limiter"""
        now = time.time()
        
        if self.redis:
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(key, 0, now - window_size)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, window_size)
            results = pipe.execute()
            
            current_requests = results[1]
            return current_requests < limit
        else:
            # Local implementation
            if key not in self.local_buckets:
                self.local_buckets[key] = []
            
            # Remove old entries
            bucket = self.local_buckets[key]
            bucket[:] = [timestamp for timestamp in bucket if now - timestamp < window_size]
            
            if len(bucket) < limit:
                bucket.append(now)
                return True
            return False
    
    def token_bucket(self, 
                    key: str, 
                    capacity: int, 
                    refill_rate: float) -> bool:
        """Token bucket rate limiter"""
        now = time.time()
        
        if self.redis:
            # Redis-based implementation
            bucket_key = f"bucket:{key}"
            
            pipe = self.redis.pipeline()
            pipe.get(bucket_key)
            results = pipe.execute()
            
            bucket_data = results[0]
            if bucket_data:
                bucket = eval(bucket_data.decode())
            else:
                bucket = {'tokens': capacity, 'last_refill': now}
            
            # Refill tokens
            time_passed = now - bucket['last_refill']
            tokens_to_add = time_passed * refill_rate
            bucket['tokens'] = min(capacity, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = now
            
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                self.redis.setex(bucket_key, 3600, str(bucket))
                return True
            else:
                self.redis.setex(bucket_key, 3600, str(bucket))
                return False
        else:
            # Local implementation
            if key not in self.local_buckets:
                self.local_buckets[key] = {'tokens': capacity, 'last_refill': now}
            
            bucket = self.local_buckets[key]
            time_passed = now - bucket['last_refill']
            tokens_to_add = time_passed * refill_rate
            bucket['tokens'] = min(capacity, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = now
            
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True
            return False
```

## 📊 Monitoring & Observability

### **Golden Signals**

The Four Golden Signals of monitoring:

| Signal | Description | Metric Examples | Alerting Thresholds |
|--------|-------------|-----------------|-------------------|
| **Latency** | Time to process requests | Response time, P95/P99 | > 500ms (warning), > 1s (critical) |
| **Traffic** | Demand on your system | RPS, concurrent users | > 80% of capacity |
| **Errors** | Rate of failed requests | 5xx errors, exceptions | > 1% (warning), > 5% (critical) |
| **Saturation** | How full your service is | CPU, memory, disk usage | > 70% (warning), > 85% (critical) |

### **Comprehensive Health Monitoring**

```python
from flask import Flask, jsonify
import psutil
import requests
import time
import threading
from typing import Dict, Any, List, Callable

class HealthChecker:
    def __init__(self):
        self.checks = {}
        self.thresholds = {
            'cpu_warning': 70,
            'cpu_critical': 85,
            'memory_warning': 80,
            'memory_critical': 90,
            'disk_warning': 80,
            'disk_critical': 95
        }
    
    def add_check(self, name: str, check_func: Callable, critical: bool = False):
        """Add custom health check"""
        self.checks[name] = {
            'func': check_func,
            'critical': critical,
            'last_result': None,
            'last_run': None
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {},
            'metrics': {}
        }
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # CPU check
        if cpu_percent > self.thresholds['cpu_critical']:
            cpu_status = 'critical'
            health_status['status'] = 'unhealthy'
        elif cpu_percent > self.thresholds['cpu_warning']:
            cpu_status = 'warning'
            if health_status['status'] == 'healthy':
                health_status['status'] = 'degraded'
        else:
            cpu_status = 'healthy'
        
        health_status['checks']['cpu'] = {
            'status': cpu_status,
            'value': cpu_percent,
            'threshold_warning': self.thresholds['cpu_warning'],
            'threshold_critical': self.thresholds['cpu_critical']
        }
        
        # Memory check
        memory_percent = memory.percent
        if memory_percent > self.thresholds['memory_critical']:
            memory_status = 'critical'
            health_status['status'] = 'unhealthy'
        elif memory_percent > self.thresholds['memory_warning']:
            memory_status = 'warning'
            if health_status['status'] == 'healthy':
                health_status['status'] = 'degraded'
        else:
            memory_status = 'healthy'
        
        health_status['checks']['memory'] = {
            'status': memory_status,
            'value': memory_percent,
            'available': memory.available,
            'used': memory.used,
            'total': memory.total
        }
        
        # Disk check
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent > self.thresholds['disk_critical']:
            disk_status = 'critical'
            health_status['status'] = 'unhealthy'
        elif disk_percent > self.thresholds['disk_warning']:
            disk_status = 'warning'
            if health_status['status'] == 'healthy':
                health_status['status'] = 'degraded'
        else:
            disk_status = 'healthy'
        
        health_status['checks']['disk'] = {
            'status': disk_status,
            'value': disk_percent,
            'free': disk.free,
            'used': disk.used,
            'total': disk.total
        }
        
        # Process information
        process = psutil.Process()
        health_status['metrics']['process'] = {
            'pid': process.pid,
            'cpu_percent': process.cpu_percent(),
            'memory_rss': process.memory_info().rss,
            'memory_vms': process.memory_info().vms,
            'num_threads': process.num_threads(),
            'create_time': process.create_time()
        }
        
        # Custom checks
        for check_name, check_config in self.checks.items():
            try:
                start_time = time.time()
                result = check_config['func']()
                end_time = time.time()
                
                health_status['checks'][check_name] = {
                    'status': 'healthy',
                    'response_time': end_time - start_time,
                    'result': result,
                    'critical': check_config['critical']
                }
                
                check_config['last_result'] = result
                check_config['last_run'] = time.time()
                
            except Exception as e:
                health_status['checks'][check_name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'critical': check_config['critical']
                }
                
                # Mark overall status as unhealthy if critical check fails
                if check_config['critical']:
                    health_status['status'] = 'unhealthy'
                elif health_status['status'] == 'healthy':
                    health_status['status'] = 'degraded'
        
        return health_status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics for monitoring"""
        metrics = {
            'timestamp': time.time(),
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100,
                'boot_time': psutil.boot_time(),
                'users': len(psutil.users())
            },
            'network': {
                'connections': len(psutil.net_connections()),
                'io_counters': psutil.net_io_counters()._asdict()
            },
            'processes': {
                'count': len(psutil.pids()),
                'top_cpu': self._get_top_processes_by_cpu(),
                'top_memory': self._get_top_processes_by_memory()
            }
        }
        return metrics
    
    def _get_top_processes_by_cpu(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top processes by CPU usage"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                proc_info = proc.info
                if proc_info['cpu_percent'] > 0:
                    processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:limit]
    
    def _get_top_processes_by_memory(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top processes by memory usage"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
            try:
                proc_info = proc.info
                processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:limit]

# Flask integration
app = Flask(__name__)
health_checker = HealthChecker()

# Add custom health checks
def check_database():
    # Database connectivity check
    try:
        # Your database connection test
        return {'status': 'connected', 'latency': 0.1}
    except Exception as e:
        raise Exception(f"Database connection failed: {e}")

def check_external_api():
    # External API health check
    try:
        response = requests.get('https://api.example.com/health', timeout=5)
        return {'status_code': response.status_code}
    except Exception as e:
        raise Exception(f"External API unreachable: {e}")

health_checker.add_check('database', check_database, critical=True)
health_checker.add_check('external_api', check_external_api, critical=False)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify(health_checker.check_system_health())

@app.route('/metrics')
def metrics():
    """Detailed metrics endpoint"""
    return jsonify(health_checker.get_metrics())

@app.route('/ready')
def readiness_check():
    """Readiness check for Kubernetes"""
    health = health_checker.check_system_health()
    if health['status'] == 'unhealthy':
        return jsonify(health), 503
    return jsonify({'status': 'ready'})

@app.route('/live')
def liveness_check():
    """Liveness check for Kubernetes"""
    return jsonify({'status': 'alive'})
```

## 🔄 Disaster Recovery

### **Backup Strategies**

**Backup Types**:

| Type | Description | RTO | RPO | Storage Cost |
|------|-------------|-----|-----|--------------|
| **Hot Backup** | Real-time replication | Minutes | Seconds | High |
| **Warm Backup** | Periodic synchronization | Hours | Minutes | Medium |
| **Cold Backup** | Offline backups | Days | Hours | Low |

**3-2-1 Backup Rule**:

```text
3-2-1 Backup Strategy:
3 copies of important data
2 different storage types
1 offsite backup

Example:
- Production database (primary)
- Local replica (second copy, different storage)
- Cloud backup (third copy, offsite)
```

### **Database Backup Implementation**

```python
import subprocess
import boto3
import gzip
import os
from datetime import datetime, timedelta
from typing import Dict, Any

class DatabaseBackupManager:
    def __init__(self, 
                 db_host: str, 
                 db_name: str, 
                 db_user: str, 
                 db_password: str,
                 s3_bucket: str):
        self.db_host = db_host
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
    
    def create_backup(self) -> Dict[str, Any]:
        """Create database backup"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{self.db_name}_backup_{timestamp}.sql"
        compressed_filename = f"{backup_filename}.gz"
        
        try:
            # Create database dump
            dump_command = [
                'pg_dump',
                f'--host={self.db_host}',
                f'--username={self.db_user}',
                f'--dbname={self.db_name}',
                '--format=custom',
                '--compress=9',
                f'--file={backup_filename}'
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_password
            
            subprocess.run(dump_command, env=env, check=True)
            
            # Compress backup
            with open(backup_filename, 'rb') as f_in:
                with gzip.open(compressed_filename, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Upload to S3
            s3_key = f"backups/{compressed_filename}"
            self.s3_client.upload_file(compressed_filename, self.s3_bucket, s3_key)
            
            # Get file size and checksum
            file_size = os.path.getsize(compressed_filename)
            
            # Cleanup local files
            os.remove(backup_filename)
            os.remove(compressed_filename)
            
            return {
                'success': True,
                'filename': compressed_filename,
                's3_key': s3_key,
                'size': file_size,
                'timestamp': timestamp
            }
            
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'error': f"Database dump failed: {e}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Backup failed: {e}"
            }
    
    def restore_backup(self, backup_s3_key: str) -> Dict[str, Any]:
        """Restore database from backup"""
        try:
            # Download backup from S3
            backup_filename = os.path.basename(backup_s3_key)
            self.s3_client.download_file(self.s3_bucket, backup_s3_key, backup_filename)
            
            # Decompress backup
            sql_filename = backup_filename.replace('.gz', '')
            with gzip.open(backup_filename, 'rb') as f_in:
                with open(sql_filename, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Restore database
            restore_command = [
                'pg_restore',
                f'--host={self.db_host}',
                f'--username={self.db_user}',
                f'--dbname={self.db_name}',
                '--clean',
                '--if-exists',
                sql_filename
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_password
            
            subprocess.run(restore_command, env=env, check=True)
            
            # Cleanup
            os.remove(backup_filename)
            os.remove(sql_filename)
            
            return {
                'success': True,
                'restored_from': backup_s3_key
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Restore failed: {e}"
            }
    
    def cleanup_old_backups(self, retention_days: int = 30):
        """Remove backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # List all backups
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix='backups/'
            )
            
            deleted_count = 0
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        self.s3_client.delete_object(
                            Bucket=self.s3_bucket,
                            Key=obj['Key']
                        )
                        deleted_count += 1
            
            return {
                'success': True,
                'deleted_count': deleted_count
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Cleanup failed: {e}"
            }
```

## 🎯 Best Practices Summary

### **Reliability Best Practices**

1. **Design for Failure**
   - Assume components will fail
   - Build in redundancy and failover mechanisms
   - Use circuit breakers and timeouts

2. **Graceful Degradation**
   - Maintain core functionality when components fail
   - Provide meaningful error messages
   - Implement fallback mechanisms

3. **Monitoring and Alerting**
   - Monitor the four golden signals
   - Set up proactive alerting
   - Use distributed tracing for complex systems

4. **Testing for Reliability**
   - Implement chaos engineering
   - Regular disaster recovery drills
   - Load testing and stress testing

### **Security Best Practices**

1. **Defense in Depth**
   - Multiple layers of security
   - No single point of failure
   - Regular security audits

2. **Principle of Least Privilege**
   - Grant minimum necessary permissions
   - Regular access reviews
   - Role-based access control

3. **Secure Development**
   - Security by design
   - Regular security training
   - Automated security testing

4. **Incident Response**
   - Documented response procedures
   - Regular practice exercises
   - Post-incident reviews

### **Operational Excellence**

1. **Documentation**
   - Runbooks and procedures
   - Architecture documentation
   - Troubleshooting guides

2. **Automation**
   - Automated deployments
   - Automated testing
   - Automated monitoring

3. **Continuous Improvement**
   - Regular post-mortems
   - Metrics-driven improvements
   - Feedback loops

---

*"The goal of reliability engineering is not to prevent all failures, but to ensure that when failures occur, they are handled gracefully and the system continues to provide value to users."*
