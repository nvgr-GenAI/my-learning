# Load Balancing Strategies ⚖️

Master the art of distributing traffic across multiple servers efficiently. Learn different algorithms, implementation patterns, and best practices for load balancing.

## 🎯 What is Load Balancing?

Load balancing distributes incoming network traffic across multiple servers to ensure no single server bears too much demand. This improves application responsiveness and availability.

## 🏗️ Load Balancing Types

=== "🌐 Layer 4 (Transport)"

    **Distribute based on IP and port**
    
    - **Protocol**: TCP/UDP level routing
    - **Speed**: Very fast, minimal processing
    - **Visibility**: Cannot see application data
    - **Use Cases**: High-performance applications, TCP proxying
    
    **Best For**: Raw performance, simple routing needs

=== "🔍 Layer 7 (Application)"

    **Distribute based on application data**
    
    - **Protocol**: HTTP/HTTPS content inspection
    - **Intelligence**: URL-based routing, header inspection
    - **Features**: SSL termination, content modification
    - **Use Cases**: Web applications, API gateways
    
    **Best For**: Complex routing, content-aware decisions

=== "🌍 Global Load Balancing"

    **Distribute across geographic regions**
    
    - **DNS-based**: Route to closest data center
    - **Anycast**: Same IP, multiple locations
    - **Latency-based**: Route to fastest endpoint
    - **Use Cases**: Global applications, disaster recovery
    
    **Best For**: Global scale, disaster recovery

## 🔄 Load Balancing Algorithms

=== "🔄 Round Robin"

    **Simple rotation through servers**
    
    ```python
    class RoundRobinBalancer:
        def __init__(self, servers):
            self.servers = servers
            self.current = 0
        
        def get_server(self):
            server = self.servers[self.current]
            self.current = (self.current + 1) % len(self.servers)
            return server
    ```
    
    **Pros**: Simple, equal distribution
    **Cons**: Doesn't consider server capacity

=== "⚖️ Weighted Round Robin"

    **Distribute based on server capacity**
    
    ```python
    class WeightedRoundRobinBalancer:
        def __init__(self, server_weights):
            self.servers = list(server_weights.keys())
            self.weights = list(server_weights.values())
            self.current_weights = [0] * len(self.servers)
        
        def get_server(self):
            total = sum(self.weights)
            
            # Increase current weights
            for i in range(len(self.current_weights)):
                self.current_weights[i] += self.weights[i]
            
            # Find server with highest current weight
            selected = self.current_weights.index(max(self.current_weights))
            self.current_weights[selected] -= total
            
            return self.servers[selected]
    ```
    
    **Pros**: Considers server capacity
    **Cons**: Static weights, not dynamic

=== "🔗 Least Connections"

    **Route to server with fewest active connections**
    
    ```python
    class LeastConnectionsBalancer:
        def __init__(self, servers):
            self.servers = servers
            self.connections = {server: 0 for server in servers}
        
        def get_server(self):
            return min(self.servers, key=lambda s: self.connections[s])
        
        def track_connection(self, server, increment=True):
            if increment:
                self.connections[server] += 1
            else:
                self.connections[server] = max(0, self.connections[server] - 1)
    ```
    
    **Pros**: Dynamic load awareness
    **Cons**: Requires connection tracking

=== "⚡ Least Response Time"

    **Route to fastest responding server**
    
    ```python
    import time
    from collections import defaultdict
    
    class LeastResponseTimeBalancer:
        def __init__(self, servers):
            self.servers = servers
            self.response_times = defaultdict(list)
            self.connections = {server: 0 for server in servers}
        
        def get_server(self):
            # Calculate average response times
            avg_times = {}
            for server in self.servers:
                times = self.response_times[server][-10:]  # Last 10 requests
                if times:
                    avg_times[server] = sum(times) / len(times)
                else:
                    avg_times[server] = 0
            
            # Combine response time and connection count
            scores = {}
            for server in self.servers:
                scores[server] = (avg_times[server] * self.connections[server])
            
            return min(self.servers, key=lambda s: scores[s])
        
        def record_response(self, server, response_time):
            self.response_times[server].append(response_time)
            # Keep only recent measurements
            if len(self.response_times[server]) > 100:
                self.response_times[server] = self.response_times[server][-50:]
    ```
    
    **Pros**: Performance-aware routing
    **Cons**: Complex implementation, monitoring overhead

## 🎯 Advanced Load Balancing

### Consistent Hashing

```python
import hashlib
import bisect

class ConsistentHashBalancer:
    def __init__(self, servers, replicas=150):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        
        for server in servers:
            self.add_server(server)
    
    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_server(self, server):
        for i in range(self.replicas):
            key = self._hash(f"{server}:{i}")
            self.ring[key] = server
            bisect.insort(self.sorted_keys, key)
    
    def remove_server(self, server):
        for i in range(self.replicas):
            key = self._hash(f"{server}:{i}")
            del self.ring[key]
            self.sorted_keys.remove(key)
    
    def get_server(self, key):
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        idx = bisect.bisect_right(self.sorted_keys, hash_key)
        
        if idx == len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]
```

### Session Affinity (Sticky Sessions)

```python
import hashlib

class StickySessionBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.session_map = {}
    
    def get_server(self, session_id):
        if session_id in self.session_map:
            server = self.session_map[session_id]
            # Verify server is still available
            if server in self.servers:
                return server
            else:
                # Server unavailable, remove mapping
                del self.session_map[session_id]
        
        # New session or server unavailable
        server = self._hash_to_server(session_id)
        self.session_map[session_id] = server
        return server
    
    def _hash_to_server(self, session_id):
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        return self.servers[hash_value % len(self.servers)]
    
    def remove_server(self, server):
        if server in self.servers:
            self.servers.remove(server)
            # Reassign sessions from removed server
            sessions_to_reassign = [
                sid for sid, srv in self.session_map.items() 
                if srv == server
            ]
            for session_id in sessions_to_reassign:
                self.session_map[session_id] = self._hash_to_server(session_id)
```

## 🏥 Health Checking

### Active Health Checks

```python
import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ServerHealth:
    is_healthy: bool
    last_check: float
    response_time: float
    consecutive_failures: int = 0

class HealthChecker:
    def __init__(self, 
                 check_interval: int = 30,
                 timeout: int = 5,
                 failure_threshold: int = 3):
        self.check_interval = check_interval
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self.server_health: Dict[str, ServerHealth] = {}
    
    async def start_monitoring(self, servers: List[str]):
        """Start health monitoring for servers"""
        while True:
            tasks = [self._check_server(server) for server in servers]
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(self.check_interval)
    
    async def _check_server(self, server: str):
        """Check health of a single server"""
        health_url = f"http://{server}/health"
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    health_url, 
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        self.server_health[server] = ServerHealth(
                            is_healthy=True,
                            last_check=time.time(),
                            response_time=response_time,
                            consecutive_failures=0
                        )
                    else:
                        self._mark_unhealthy(server, response_time)
                        
        except Exception:
            self._mark_unhealthy(server, time.time() - start_time)
    
    def _mark_unhealthy(self, server: str, response_time: float):
        """Mark server as unhealthy"""
        current_health = self.server_health.get(server, ServerHealth(True, 0, 0))
        current_health.consecutive_failures += 1
        current_health.last_check = time.time()
        current_health.response_time = response_time
        
        if current_health.consecutive_failures >= self.failure_threshold:
            current_health.is_healthy = False
        
        self.server_health[server] = current_health
    
    def get_healthy_servers(self, servers: List[str]) -> List[str]:
        """Get list of healthy servers"""
        return [
            server for server in servers
            if self.server_health.get(server, ServerHealth(True, 0, 0)).is_healthy
        ]
```

### Passive Health Checks

```python
import time
from collections import defaultdict

class PassiveHealthChecker:
    def __init__(self, 
                 failure_threshold: int = 5,
                 success_threshold: int = 2,
                 timeout_window: int = 60):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_window = timeout_window
        
        self.failure_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        self.unhealthy_servers = set()
        self.last_failure_time = defaultdict(float)
    
    def record_success(self, server: str):
        """Record successful request to server"""
        if server in self.unhealthy_servers:
            self.success_counts[server] += 1
            
            if self.success_counts[server] >= self.success_threshold:
                self.unhealthy_servers.remove(server)
                self.failure_counts[server] = 0
                self.success_counts[server] = 0
        else:
            # Reset failure count on success
            self.failure_counts[server] = 0
    
    def record_failure(self, server: str):
        """Record failed request to server"""
        self.failure_counts[server] += 1
        self.last_failure_time[server] = time.time()
        
        if self.failure_counts[server] >= self.failure_threshold:
            self.unhealthy_servers.add(server)
            self.success_counts[server] = 0
    
    def is_healthy(self, server: str) -> bool:
        """Check if server is healthy"""
        if server not in self.unhealthy_servers:
            return True
        
        # Check if enough time has passed to retry
        last_failure = self.last_failure_time.get(server, 0)
        if time.time() - last_failure > self.timeout_window:
            # Reset for retry
            self.success_counts[server] = 0
            return True
        
        return False
```

## 🛠️ Load Balancer Implementation

### Software Load Balancer

```python
import asyncio
import aiohttp
from aiohttp import web
import json
import time

class SoftwareLoadBalancer:
    def __init__(self, port=8080):
        self.port = port
        self.app = web.Application()
        self.servers = []
        self.balancer = None
        self.health_checker = HealthChecker()
        
        # Setup routes
        self.app.router.add_route('*', '/{path:.*}', self.handle_request)
        self.app.router.add_get('/lb/health', self.health_status)
        self.app.router.add_post('/lb/servers', self.add_server)
        self.app.router.add_delete('/lb/servers/{server}', self.remove_server)
    
    def set_balancing_algorithm(self, algorithm: str):
        """Set load balancing algorithm"""
        if algorithm == 'round_robin':
            self.balancer = RoundRobinBalancer(self.servers)
        elif algorithm == 'least_connections':
            self.balancer = LeastConnectionsBalancer(self.servers)
        elif algorithm == 'consistent_hash':
            self.balancer = ConsistentHashBalancer(self.servers)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    async def handle_request(self, request):
        """Handle incoming request and forward to backend"""
        healthy_servers = self.health_checker.get_healthy_servers(self.servers)
        
        if not healthy_servers:
            return web.Response(status=503, text="No healthy servers available")
        
        # Select server using balancing algorithm
        server = self.balancer.get_server()
        if server not in healthy_servers:
            server = healthy_servers[0]  # Fallback to first healthy server
        
        # Forward request
        try:
            start_time = time.time()
            
            # Track connection for least connections algorithm
            if hasattr(self.balancer, 'track_connection'):
                self.balancer.track_connection(server, increment=True)
            
            # Build target URL
            target_url = f"http://{server}{request.path_qs}"
            
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=request.headers,
                    data=await request.read()
                ) as response:
                    body = await response.read()
                    
                    # Record response time for algorithms that need it
                    response_time = time.time() - start_time
                    if hasattr(self.balancer, 'record_response'):
                        self.balancer.record_response(server, response_time)
                    
                    # Track successful request
                    if hasattr(self, 'passive_health_checker'):
                        self.passive_health_checker.record_success(server)
                    
                    return web.Response(
                        body=body,
                        status=response.status,
                        headers=response.headers
                    )
                    
        except Exception as e:
            # Track failed request
            if hasattr(self, 'passive_health_checker'):
                self.passive_health_checker.record_failure(server)
            
            return web.Response(status=502, text=f"Backend error: {str(e)}")
        
        finally:
            # Untrack connection
            if hasattr(self.balancer, 'track_connection'):
                self.balancer.track_connection(server, increment=False)
    
    async def health_status(self, request):
        """Return health status of all servers"""
        status = {
            'load_balancer': 'healthy',
            'servers': {}
        }
        
        for server in self.servers:
            health = self.health_checker.server_health.get(server)
            if health:
                status['servers'][server] = {
                    'healthy': health.is_healthy,
                    'response_time': health.response_time,
                    'last_check': health.last_check
                }
            else:
                status['servers'][server] = {'healthy': 'unknown'}
        
        return web.json_response(status)
    
    async def add_server(self, request):
        """Add server to pool"""
        data = await request.json()
        server = data.get('server')
        
        if server and server not in self.servers:
            self.servers.append(server)
            if hasattr(self.balancer, 'add_server'):
                self.balancer.add_server(server)
            
            return web.json_response({'status': 'added', 'server': server})
        
        return web.json_response({'error': 'Invalid server'}, status=400)
    
    async def remove_server(self, request):
        """Remove server from pool"""
        server = request.match_info['server']
        
        if server in self.servers:
            self.servers.remove(server)
            if hasattr(self.balancer, 'remove_server'):
                self.balancer.remove_server(server)
            
            return web.json_response({'status': 'removed', 'server': server})
        
        return web.json_response({'error': 'Server not found'}, status=404)
    
    async def start(self):
        """Start the load balancer"""
        # Start health monitoring
        asyncio.create_task(self.health_checker.start_monitoring(self.servers))
        
        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        print(f"Load balancer started on port {self.port}")
        return runner

# Usage example
async def main():
    lb = SoftwareLoadBalancer(port=8080)
    lb.servers = ['localhost:3001', 'localhost:3002', 'localhost:3003']
    lb.set_balancing_algorithm('round_robin')
    
    runner = await lb.start()
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
```

## 🌐 Popular Load Balancer Technologies

### Nginx Configuration

```nginx
upstream backend {
    # Load balancing method
    least_conn;  # or ip_hash, hash, random
    
    # Backend servers
    server backend1.example.com:8080 weight=3 max_fails=3 fail_timeout=30s;
    server backend2.example.com:8080 weight=2 max_fails=3 fail_timeout=30s;
    server backend3.example.com:8080 weight=1 max_fails=3 fail_timeout=30s;
    
    # Health checks
    keepalive 32;
}

server {
    listen 80;
    server_name example.com;
    
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

### HAProxy Configuration

```
# HAProxy configuration
global
    daemon
    maxconn 4096
    
defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httpchk GET /health
    
frontend web_frontend
    bind *:80
    default_backend web_servers
    
backend web_servers
    balance roundrobin
    option httpchk GET /health HTTP/1.1\r\nHost:\ example.com
    
    server web1 10.0.0.10:8080 check inter 5s fall 3 rise 2
    server web2 10.0.0.11:8080 check inter 5s fall 3 rise 2
    server web3 10.0.0.12:8080 check inter 5s fall 3 rise 2
    
# Stats page
listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
```

## 📊 Monitoring and Metrics

### Key Metrics to Track

```python
class LoadBalancerMetrics:
    def __init__(self):
        self.request_count = 0
        self.response_times = []
        self.error_count = 0
        self.server_requests = defaultdict(int)
        self.server_errors = defaultdict(int)
    
    def record_request(self, server: str, response_time: float, success: bool):
        """Record request metrics"""
        self.request_count += 1
        self.response_times.append(response_time)
        self.server_requests[server] += 1
        
        if not success:
            self.error_count += 1
            self.server_errors[server] += 1
        
        # Keep only recent response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-500:]
    
    def get_metrics(self):
        """Get current metrics"""
        if not self.response_times:
            return {}
        
        sorted_times = sorted(self.response_times)
        n = len(sorted_times)
        
        return {
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': (self.error_count / self.request_count) * 100,
            'avg_response_time': sum(self.response_times) / n,
            'p50_response_time': sorted_times[int(n * 0.5)],
            'p95_response_time': sorted_times[int(n * 0.95)],
            'p99_response_time': sorted_times[int(n * 0.99)],
            'server_distribution': dict(self.server_requests),
            'server_errors': dict(self.server_errors)
        }
```

## ✅ Best Practices

### 1. Algorithm Selection

- **Round Robin**: Equal servers, simple workloads
- **Weighted**: Different server capacities
- **Least Connections**: Long-lived connections
- **Consistent Hashing**: Cache-friendly routing

### 2. Health Checking

- **Active**: Regular health endpoint checks
- **Passive**: Monitor request success/failure
- **Circuit Breaking**: Temporarily remove failing servers
- **Graceful Degradation**: Partial functionality during failures

### 3. Session Management

- **Stateless Design**: Preferred approach
- **Session Stores**: External session storage
- **Sticky Sessions**: When stateless isn't possible
- **Session Replication**: Backup session data

## 🚀 Next Steps

- **[Auto-scaling](auto-scaling.md)** - Dynamic capacity management
- **[Caching](../caching/index.md)** - Reduce backend load
- **[Database Scaling](database-scaling.md)** - Scale data layer
- **[Performance Monitoring](../performance/monitoring.md)** - Track system health

---

**Balance the load, balance the power! ⚖️💪**
