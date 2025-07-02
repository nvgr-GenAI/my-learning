# Fault Tolerance Patterns 🛡️

Build systems that gracefully handle failures and continue operating when components fail.

## 🎯 Overview

Fault tolerance is the ability of a system to continue operating correctly even when some of its components fail. This is crucial for building reliable distributed systems.

## 🔧 Core Patterns

### 1. Circuit Breaker Pattern

Prevents cascading failures by temporarily disabling failing services.

```python
import time
import threading
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60, success_threshold=3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self):
        return (time.time() - self.last_failure_time) > self.recovery_timeout
    
    def _on_success(self):
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
```

### 2. Bulkhead Pattern

Isolate critical resources to prevent failure cascades.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

class BulkheadExecutor:
    def __init__(self, pool_configs: Dict[str, int]):
        """
        Initialize bulkhead with separate thread pools for different operations
        
        Args:
            pool_configs: Dict mapping operation names to max worker counts
        """
        self.executors = {
            name: ThreadPoolExecutor(max_workers=max_workers)
            for name, max_workers in pool_configs.items()
        }
    
    async def execute(self, operation_name: str, func, *args, **kwargs):
        """Execute function in the appropriate bulkhead"""
        if operation_name not in self.executors:
            raise ValueError(f"Unknown operation: {operation_name}")
        
        executor = self.executors[operation_name]
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(executor, func, *args, **kwargs)
    
    def shutdown(self):
        """Shutdown all executors"""
        for executor in self.executors.values():
            executor.shutdown(wait=True)

# Usage example
bulkhead = BulkheadExecutor({
    'database': 10,    # Database operations get 10 threads
    'external_api': 5, # External API calls get 5 threads
    'file_io': 3,      # File I/O operations get 3 threads
    'critical': 2      # Critical operations get dedicated threads
})

async def example_usage():
    # These operations are isolated from each other
    result1 = await bulkhead.execute('database', database_query)
    result2 = await bulkhead.execute('external_api', api_call)
    result3 = await bulkhead.execute('file_io', read_file)
```

### 3. Retry Pattern with Backoff

Handle transient failures with intelligent retry strategies.

```python
import asyncio
import random
import time
from typing import Callable, Any, Optional

class RetryStrategy:
    def __init__(self, max_attempts=3, base_delay=1.0, max_delay=60.0, 
                 backoff_factor=2.0, jitter=True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        if attempt <= 0:
            return 0
        
        # Exponential backoff
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await self._execute_with_timeout(func, *args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts:
                    break
                
                delay = self.get_delay(attempt)
                print(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    async def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function, handling both sync and async functions"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

# Usage
retry_strategy = RetryStrategy(max_attempts=5, base_delay=1.0, backoff_factor=2.0)

async def unreliable_operation():
    if random.random() < 0.7:  # 70% chance of failure
        raise Exception("Service temporarily unavailable")
    return "Success!"

async def example():
    try:
        result = await retry_strategy.execute(unreliable_operation)
        print(f"Operation succeeded: {result}")
    except Exception as e:
        print(f"Operation failed after all retries: {e}")
```

### 4. Timeout Pattern

Prevent operations from hanging indefinitely.

```python
import asyncio
import signal
from contextlib import asynccontextmanager
from typing import Optional

class TimeoutManager:
    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout
    
    @asynccontextmanager
    async def timeout(self, timeout_seconds: Optional[float] = None):
        """Context manager for timeout operations"""
        timeout_seconds = timeout_seconds or self.default_timeout
        
        try:
            async with asyncio.timeout(timeout_seconds):
                yield
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_seconds}s")
    
    async def execute_with_timeout(self, func, *args, timeout_seconds=None, **kwargs):
        """Execute function with timeout"""
        async with self.timeout(timeout_seconds):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

# Usage
timeout_manager = TimeoutManager(default_timeout=10.0)

async def example_with_timeout():
    try:
        async with timeout_manager.timeout(5.0):
            result = await some_long_running_operation()
            return result
    except TimeoutError as e:
        print(f"Operation timed out: {e}")
        return None
```

### 5. Graceful Degradation

Maintain core functionality when dependencies fail.

```python
import asyncio
from typing import Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class ServiceLevel:
    FULL = "full"
    DEGRADED = "degraded"
    MINIMAL = "minimal"

class GracefulDegradationManager:
    def __init__(self):
        self.service_health = {}
        self.fallback_strategies = {}
    
    def register_service(self, name: str, health_check: Callable, 
                        fallback_strategy: Optional[Callable] = None):
        """Register a service with its health check and fallback strategy"""
        self.service_health[name] = {
            'health_check': health_check,
            'is_healthy': True,
            'last_check': 0
        }
        if fallback_strategy:
            self.fallback_strategies[name] = fallback_strategy
    
    async def call_service(self, service_name: str, operation: Callable, 
                          *args, **kwargs) -> Any:
        """Call service with graceful degradation"""
        if await self._is_service_healthy(service_name):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                # Mark service as unhealthy
                self.service_health[service_name]['is_healthy'] = False
                print(f"Service {service_name} failed: {e}")
        
        # Use fallback strategy if available
        if service_name in self.fallback_strategies:
            print(f"Using fallback for {service_name}")
            return await self.fallback_strategies[service_name](*args, **kwargs)
        
        # Return degraded response
        return self._get_degraded_response(service_name)
    
    async def _is_service_healthy(self, service_name: str) -> bool:
        """Check if service is healthy"""
        service = self.service_health.get(service_name)
        if not service:
            return False
        
        # Periodic health check
        current_time = time.time()
        if current_time - service['last_check'] > 30:  # Check every 30 seconds
            try:
                await service['health_check']()
                service['is_healthy'] = True
            except Exception:
                service['is_healthy'] = False
            finally:
                service['last_check'] = current_time
        
        return service['is_healthy']
    
    def _get_degraded_response(self, service_name: str):
        """Return appropriate degraded response"""
        degraded_responses = {
            'recommendation_service': {'recommendations': [], 'source': 'cache'},
            'user_service': {'user': {'name': 'Guest'}, 'source': 'default'},
            'analytics_service': None  # Analytics can be completely disabled
        }
        return degraded_responses.get(service_name, {'error': 'Service unavailable'})

# Usage example
degradation_manager = GracefulDegradationManager()

async def recommendation_health_check():
    # Simulate health check
    async with aiohttp.ClientSession() as session:
        async with session.get('http://recommendation-service/health') as response:
            if response.status != 200:
                raise Exception("Health check failed")

async def get_cached_recommendations():
    # Fallback to cached recommendations
    return {'recommendations': ['Popular Item 1', 'Popular Item 2'], 'source': 'cache'}

# Register services
degradation_manager.register_service(
    'recommendation_service', 
    recommendation_health_check,
    get_cached_recommendations
)
```

## 🔧 Implementation Strategies

### Combining Patterns

```python
class FaultTolerantService:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(failure_threshold=3)
        self.retry_strategy = RetryStrategy(max_attempts=3)
        self.timeout_manager = TimeoutManager(default_timeout=10.0)
    
    async def resilient_call(self, service_func, *args, **kwargs):
        """Combine multiple fault tolerance patterns"""
        
        async def protected_call():
            async with self.timeout_manager.timeout(5.0):
                return self.circuit_breaker.call(service_func, *args, **kwargs)
        
        return await self.retry_strategy.execute(protected_call)

# Usage
service = FaultTolerantService()
result = await service.resilient_call(external_api_call, param1, param2)
```

## 🎯 Best Practices

### Design Principles

1. **Fail Fast**: Detect failures quickly and avoid cascading effects
2. **Fail Safe**: Ensure system remains in a safe state when failures occur
3. **Fail Gracefully**: Provide degraded functionality rather than complete failure
4. **Isolate Failures**: Prevent failures from spreading across system boundaries

### Implementation Guidelines

- **Set Appropriate Timeouts**: Balance responsiveness with reliability
- **Use Exponential Backoff**: Prevent overwhelming struggling services
- **Monitor Circuit Breaker States**: Track open/closed states for insights
- **Test Failure Scenarios**: Regularly test fault tolerance mechanisms
- **Document Degradation Behavior**: Clearly define what happens when services fail

### Common Pitfalls

- **Insufficient Timeout Values**: Too short timeouts cause false failures
- **Retry Storms**: Overwhelming failed services with retries
- **Ignoring Circuit Breaker States**: Not monitoring breaker health
- **Poor Fallback Strategies**: Inadequate degraded functionality
- **Tight Coupling**: Dependencies that prevent effective isolation

## 🔗 Related Patterns

- [Circuit Breaker](circuit-breaker.md)
- [Retry Strategies](retry-strategies.md)
- [Graceful Degradation](graceful-degradation.md)
- [Monitoring & Alerting](../performance/monitoring-strategies.md)

## 📚 Further Reading

- [Release It!](https://pragprog.com/titles/mnee2/release-it-second-edition/)
- [Building Secure & Reliable Systems](https://www.oreilly.com/library/view/building-secure-and/9781492083115/)
- [Microservices Patterns](https://www.manning.com/books/microservices-patterns)
