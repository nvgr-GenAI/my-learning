# Design a URL Validator Service

Design a URL validation service that checks if URLs are properly formatted, safe, and reachable. The service should validate syntax, check for malicious content, and verify URL accessibility.

**Difficulty:** 🟢 Easy | **Frequency:** ⭐⭐⭐ Medium | **Time:** 30-40 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10K validations/sec, 100M URLs/day |
| **Key Challenges** | Syntax validation, security checks, reachability testing, rate limiting |
| **Core Concepts** | Regex, DNS lookup, HTTP HEAD request, URL parsing, blacklist/whitelist |
| **Companies** | Security companies, browsers, link shorteners, social media platforms |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Syntax Validation** | Check if URL format is correct | P0 (Must have) |
    | **Protocol Check** | Validate HTTP/HTTPS protocols | P0 (Must have) |
    | **Domain Validation** | Verify domain exists (DNS lookup) | P0 (Must have) |
    | **Reachability Check** | Verify URL is accessible | P1 (Should have) |
    | **Security Screening** | Check against malicious URL blacklist | P1 (Should have) |
    | **Response Caching** | Cache validation results | P1 (Should have) |

    **Explicitly Out of Scope:**

    - Deep content analysis
    - Phishing detection (ML-based)
    - SSL certificate validation
    - Performance testing
    - Link following/redirects

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency** | < 100ms p95 | Fast validation for real-time use |
    | **Availability** | 99.9% | Service must be reliable |
    | **Throughput** | 10K requests/sec | Handle high validation load |
    | **Accuracy** | > 99% | Minimize false positives/negatives |
    | **Cache Hit Rate** | > 70% | Reduce repeated validations |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily validations: 100 million
    QPS: 100M / 86,400 = 1,160 QPS
    Peak QPS: 3x average = 3,500 QPS

    Cache hit rate: 70%
    - Cache hits: 1,160 × 0.7 = 812 QPS (from cache)
    - Cache misses: 1,160 × 0.3 = 348 QPS (full validation)
    ```

    ### Storage Estimates

    ```
    Validation cache:
    - URL: 200 bytes average
    - Result: 50 bytes (valid/invalid, reason, timestamp)
    - Total per entry: 250 bytes

    Cache 10M most recent URLs:
    - 10M × 250 bytes = 2.5 GB

    Blacklist storage:
    - 1M malicious URLs × 200 bytes = 200 MB

    Total storage: ~2.7 GB
    ```

    ### Bandwidth Estimates

    ```
    Request bandwidth:
    - 1,160 QPS × 200 bytes = 232 KB/sec ≈ 2 Mbps

    Response bandwidth:
    - 1,160 QPS × 100 bytes = 116 KB/sec ≈ 1 Mbps

    External checks (DNS, HTTP):
    - 348 QPS × 500 bytes = 174 KB/sec ≈ 1.4 Mbps

    Total bandwidth: ~4.4 Mbps
    ```

    ---

    ## Key Assumptions

    1. Most URLs are HTTP/HTTPS
    2. 70% of validations are repeated URLs (cache-friendly)
    3. DNS lookup: 10-50ms
    4. HTTP HEAD request: 100-500ms
    5. Blacklist updated hourly

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Multi-layer validation** - Syntax → DNS → Reachability → Security
    2. **Fail fast** - Stop at first validation failure
    3. **Caching** - Cache valid/invalid results with TTL
    4. **Async checks** - DNS and HTTP in parallel when possible
    5. **Rate limiting** - Prevent abuse

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Client[Client Applications]
        end

        subgraph "API Layer"
            API[API Server<br/>Rate Limiter]
        end

        subgraph "Validation Pipeline"
            Syntax[1. Syntax Validator<br/>Regex Check]
            DNS[2. DNS Validator<br/>Domain Lookup]
            HTTP[3. Reachability Check<br/>HTTP HEAD]
            Security[4. Security Check<br/>Blacklist]
        end

        subgraph "Data Layer"
            Cache[Redis Cache<br/>Validation Results]
            Blacklist[(Malicious URLs<br/>PostgreSQL)]
        end

        subgraph "External Services"
            DNSServer[DNS Servers]
            WebServer[Target Web Servers]
        end

        Client --> API
        API --> Cache

        Cache --> Syntax
        Syntax --> DNS
        DNS --> DNSServer
        DNS --> HTTP
        HTTP --> WebServer
        HTTP --> Security
        Security --> Blacklist

        style Cache fill:#fff4e1
        style Security fill:#ffe1e1
    ```

    ---

    ## API Design

    ### 1. Validate URL

    **Request:**
    ```http
    POST /api/v1/validate
    Content-Type: application/json

    {
      "url": "https://example.com/path?query=value",
      "checks": ["syntax", "dns", "reachability", "security"]
    }
    ```

    **Response (Valid):**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "valid": true,
      "url": "https://example.com/path?query=value",
      "checks": {
        "syntax": {
          "passed": true,
          "message": "Valid URL format"
        },
        "dns": {
          "passed": true,
          "message": "Domain resolves to 93.184.216.34",
          "latency_ms": 15
        },
        "reachability": {
          "passed": true,
          "status_code": 200,
          "latency_ms": 145
        },
        "security": {
          "passed": true,
          "message": "Not in blacklist"
        }
      },
      "cached": false,
      "validated_at": "2026-02-04T10:30:00Z"
    }
    ```

    **Response (Invalid):**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "valid": false,
      "url": "htp://invalid..com",
      "checks": {
        "syntax": {
          "passed": false,
          "message": "Invalid protocol: htp",
          "error_code": "INVALID_PROTOCOL"
        }
      },
      "validated_at": "2026-02-04T10:30:00Z"
    }
    ```

    ---

    ### 2. Batch Validate

    **Request:**
    ```http
    POST /api/v1/validate/batch
    Content-Type: application/json

    {
      "urls": [
        "https://example.com",
        "https://google.com",
        "http://malicious-site.com"
      ]
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "results": [
        {
          "url": "https://example.com",
          "valid": true,
          "cached": true
        },
        {
          "url": "https://google.com",
          "valid": true,
          "cached": false
        },
        {
          "url": "http://malicious-site.com",
          "valid": false,
          "reason": "URL in security blacklist"
        }
      ]
    }
    ```

    ---

    ## Validation Implementation

    ### 1. Syntax Validation

    ```python
    import re
    from urllib.parse import urlparse

    class SyntaxValidator:
        """Validate URL syntax and structure"""

        # Comprehensive URL regex
        URL_REGEX = re.compile(
            r'^(?:http|https)://'  # Protocol
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # Domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # Optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )

        def validate(self, url: str) -> dict:
            """
            Validate URL syntax

            Returns:
                dict with passed (bool), message (str), error_code (str)
            """
            # Check length
            if len(url) > 2048:
                return {
                    'passed': False,
                    'message': 'URL exceeds maximum length (2048 characters)',
                    'error_code': 'URL_TOO_LONG'
                }

            # Check regex match
            if not self.URL_REGEX.match(url):
                return {
                    'passed': False,
                    'message': 'Invalid URL format',
                    'error_code': 'INVALID_FORMAT'
                }

            # Parse URL components
            try:
                parsed = urlparse(url)

                # Validate protocol
                if parsed.scheme not in ['http', 'https']:
                    return {
                        'passed': False,
                        'message': f'Invalid protocol: {parsed.scheme}',
                        'error_code': 'INVALID_PROTOCOL'
                    }

                # Validate domain
                if not parsed.netloc:
                    return {
                        'passed': False,
                        'message': 'Missing domain',
                        'error_code': 'MISSING_DOMAIN'
                    }

                # Check for suspicious patterns
                if '..' in parsed.netloc:
                    return {
                        'passed': False,
                        'message': 'Invalid domain format',
                        'error_code': 'INVALID_DOMAIN'
                    }

                return {
                    'passed': True,
                    'message': 'Valid URL format'
                }

            except Exception as e:
                return {
                    'passed': False,
                    'message': f'Parse error: {str(e)}',
                    'error_code': 'PARSE_ERROR'
                }
    ```

    ### 2. DNS Validation

    ```python
    import socket
    import time

    class DNSValidator:
        """Validate domain via DNS lookup"""

        def validate(self, url: str) -> dict:
            """
            Check if domain resolves via DNS

            Returns:
                dict with passed (bool), message (str), latency_ms (int)
            """
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc.split(':')[0]  # Remove port if present

            start_time = time.time()

            try:
                # DNS lookup
                ip_address = socket.gethostbyname(domain)
                latency_ms = int((time.time() - start_time) * 1000)

                return {
                    'passed': True,
                    'message': f'Domain resolves to {ip_address}',
                    'ip_address': ip_address,
                    'latency_ms': latency_ms
                }

            except socket.gaierror as e:
                latency_ms = int((time.time() - start_time) * 1000)
                return {
                    'passed': False,
                    'message': f'DNS lookup failed: {str(e)}',
                    'error_code': 'DNS_LOOKUP_FAILED',
                    'latency_ms': latency_ms
                }
    ```

    ### 3. Reachability Check

    ```python
    import requests

    class ReachabilityValidator:
        """Check if URL is reachable"""

        def __init__(self, timeout: int = 5):
            self.timeout = timeout

        def validate(self, url: str) -> dict:
            """
            Send HTTP HEAD request to check reachability

            Returns:
                dict with passed (bool), status_code (int), latency_ms (int)
            """
            start_time = time.time()

            try:
                # Send HEAD request (faster than GET)
                response = requests.head(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True,
                    headers={'User-Agent': 'URLValidator/1.0'}
                )

                latency_ms = int((time.time() - start_time) * 1000)

                # Consider 2xx and 3xx as valid
                if 200 <= response.status_code < 400:
                    return {
                        'passed': True,
                        'status_code': response.status_code,
                        'latency_ms': latency_ms
                    }
                else:
                    return {
                        'passed': False,
                        'message': f'HTTP error: {response.status_code}',
                        'status_code': response.status_code,
                        'error_code': 'HTTP_ERROR',
                        'latency_ms': latency_ms
                    }

            except requests.Timeout:
                return {
                    'passed': False,
                    'message': f'Request timeout after {self.timeout}s',
                    'error_code': 'TIMEOUT',
                    'latency_ms': self.timeout * 1000
                }

            except requests.RequestException as e:
                latency_ms = int((time.time() - start_time) * 1000)
                return {
                    'passed': False,
                    'message': f'Request failed: {str(e)}',
                    'error_code': 'REQUEST_FAILED',
                    'latency_ms': latency_ms
                }
    ```

    ### 4. Security Check

    ```python
    class SecurityValidator:
        """Check URL against security blacklist"""

        def __init__(self, blacklist_db):
            self.blacklist_db = blacklist_db

        def validate(self, url: str) -> dict:
            """
            Check if URL is in malicious URL blacklist

            Returns:
                dict with passed (bool), message (str)
            """
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc

            # Check exact URL
            if self.blacklist_db.is_blacklisted(url):
                return {
                    'passed': False,
                    'message': 'URL in security blacklist',
                    'error_code': 'BLACKLISTED_URL',
                    'severity': 'HIGH'
                }

            # Check domain
            if self.blacklist_db.is_domain_blacklisted(domain):
                return {
                    'passed': False,
                    'message': 'Domain in security blacklist',
                    'error_code': 'BLACKLISTED_DOMAIN',
                    'severity': 'HIGH'
                }

            return {
                'passed': True,
                'message': 'Not in blacklist'
            }
    ```

=== "🔍 Step 3: Deep Dive"

    ## Orchestrating Validation Pipeline

    ```python
    class URLValidatorService:
        """Orchestrate URL validation pipeline"""

        def __init__(self, cache, blacklist_db):
            self.syntax_validator = SyntaxValidator()
            self.dns_validator = DNSValidator()
            self.reachability_validator = ReachabilityValidator(timeout=5)
            self.security_validator = SecurityValidator(blacklist_db)
            self.cache = cache

        def validate_url(self, url: str, checks: list = None) -> dict:
            """
            Validate URL through pipeline

            Args:
                url: URL to validate
                checks: List of checks to perform (default: all)

            Returns:
                Validation result dict
            """
            if checks is None:
                checks = ['syntax', 'dns', 'reachability', 'security']

            # Check cache first
            cache_key = f"url_validation:{url}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                cached_result['cached'] = True
                return cached_result

            result = {
                'valid': True,
                'url': url,
                'checks': {},
                'cached': False
            }

            # 1. Syntax validation (fast, no external calls)
            if 'syntax' in checks:
                syntax_result = self.syntax_validator.validate(url)
                result['checks']['syntax'] = syntax_result

                if not syntax_result['passed']:
                    result['valid'] = False
                    self._cache_result(url, result, ttl=3600)  # Cache 1 hour
                    return result

            # 2. DNS validation
            if 'dns' in checks:
                dns_result = self.dns_validator.validate(url)
                result['checks']['dns'] = dns_result

                if not dns_result['passed']:
                    result['valid'] = False
                    self._cache_result(url, result, ttl=300)  # Cache 5 min
                    return result

            # 3. Reachability check
            if 'reachability' in checks:
                reachability_result = self.reachability_validator.validate(url)
                result['checks']['reachability'] = reachability_result

                if not reachability_result['passed']:
                    result['valid'] = False
                    self._cache_result(url, result, ttl=60)  # Cache 1 min
                    return result

            # 4. Security check
            if 'security' in checks:
                security_result = self.security_validator.validate(url)
                result['checks']['security'] = security_result

                if not security_result['passed']:
                    result['valid'] = False
                    self._cache_result(url, result, ttl=86400)  # Cache 24 hours
                    return result

            # All checks passed
            self._cache_result(url, result, ttl=3600)  # Cache 1 hour
            return result

        def _cache_result(self, url: str, result: dict, ttl: int):
            """Cache validation result"""
            cache_key = f"url_validation:{url}"
            self.cache.setex(cache_key, ttl, result)
    ```

=== "⚡ Step 4: Scale & Optimize"

    ## Performance Optimization

    ### 1. Caching Strategy

    **Different TTLs based on result:**
    - Valid URLs: 1 hour
    - Invalid syntax: 1 hour (unlikely to change)
    - DNS failures: 5 minutes (might be temporary)
    - HTTP errors: 1 minute (might be temporary)
    - Blacklisted: 24 hours (rarely change)

    ### 2. Async Validation

    For batch requests, validate URLs in parallel:

    ```python
    import asyncio

    async def validate_batch(self, urls: list) -> list:
        """Validate multiple URLs in parallel"""
        tasks = [self.validate_url_async(url) for url in urls]
        return await asyncio.gather(*tasks)
    ```

    ### 3. Rate Limiting

    Prevent abuse by limiting validations per client:
    - 100 validations per minute per IP
    - 1000 validations per hour per API key

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Multi-layer validation** - Syntax → DNS → HTTP → Security
    2. **Fail fast** - Stop at first failure
    3. **Caching with TTL** - Different TTLs for different results
    4. **HTTP HEAD vs GET** - HEAD is faster, sufficient for reachability
    5. **Rate limiting** - Prevent abuse

    ## Interview Tips

    ✅ **Discuss validation layers** - Syntax, DNS, reachability, security
    ✅ **Explain caching strategy** - Different TTLs based on result type
    ✅ **Consider performance** - Parallel validation, HEAD vs GET
    ✅ **Address security** - Blacklist checks, rate limiting

---

**Difficulty:** 🟢 Easy | **Interview Time:** 30-40 minutes | **Companies:** Security companies, browsers, URL shorteners

---

*This problem demonstrates API design, validation logic, caching, and security considerations.*
