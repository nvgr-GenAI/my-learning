# Consensus Algorithms & Distributed Coordination 🤝

Understanding how distributed systems reach agreement is crucial for building reliable, consistent systems. This comprehensive guide covers consensus algorithms, coordination patterns, and distributed system synchronization.

## 🎯 The Consensus Problem

### **What is Consensus?**

Consensus is the fundamental problem of getting multiple distributed nodes to agree on a single value, even when some nodes may fail or the network may be unreliable.

> **Real-World Analogy**: Think of distributed consensus like a group of friends deciding on a restaurant for dinner when they can only communicate by text messages that might be delayed or lost.

**Key Requirements**:

1. **Agreement**: All non-faulty nodes must agree on the same value
2. **Validity**: The agreed value must be proposed by some node  
3. **Termination**: All non-faulty nodes must eventually decide

### **Challenges in Distributed Systems**

| Challenge | Description | Example |
|-----------|-------------|---------|
| **Network Partitions** | Nodes can't communicate with each other | Network split-brain |
| **Node Failures** | Individual nodes may crash or stop responding | Server hardware failure |
| **Message Delays** | Messages may be delayed or arrive out of order | Network congestion |
| **Byzantine Failures** | Nodes may behave maliciously or unpredictably | Compromised nodes |

## 🗳️ Consensus Algorithms

### **1. Raft Algorithm**

Raft is designed to be easy to understand and implement. It decomposes consensus into leader election, log replication, and safety.

**Key Components**:
- **Leader Election**: One node becomes leader for a term
- **Log Replication**: Leader replicates commands to followers
- **Safety**: Ensures committed entries are never lost

**Implementation**:

```python
import asyncio
import random
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

@dataclass
class LogEntry:
    term: int
    index: int
    command: Any
    committed: bool = False

class RaftNode:
    def __init__(self, node_id: str, peer_ids: List[str]):
        self.node_id = node_id
        self.peer_ids = peer_ids
        self.state = NodeState.FOLLOWER
        
        # Persistent state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        
        # Volatile state
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Timing
        self.last_heartbeat = time.time()
        self.election_timeout = random.uniform(0.5, 1.0)
        self.heartbeat_interval = 0.1
        
        self.running = False
        self.votes_received = set()
    
    async def start(self):
        """Start the Raft node"""
        self.running = True
        
        # Start main loop
        await asyncio.gather(
            self._main_loop(),
            self._apply_committed_entries()
        )
    
    async def _main_loop(self):
        """Main state machine loop"""
        while self.running:
            if self.state == NodeState.FOLLOWER:
                await self._follower_loop()
            elif self.state == NodeState.CANDIDATE:
                await self._candidate_loop()
            elif self.state == NodeState.LEADER:
                await self._leader_loop()
    
    async def _follower_loop(self):
        """Follower state behavior"""
        while self.state == NodeState.FOLLOWER and self.running:
            # Check for election timeout
            if time.time() - self.last_heartbeat > self.election_timeout:
                print(f"Node {self.node_id}: Election timeout, becoming candidate")
                self.state = NodeState.CANDIDATE
                break
            
            await asyncio.sleep(0.05)
    
    async def _candidate_loop(self):
        """Candidate state behavior"""
        # Start new election
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self.last_heartbeat = time.time()
        
        print(f"Node {self.node_id}: Starting election for term {self.current_term}")
        
        # Send vote requests to all peers
        vote_tasks = []
        for peer_id in self.peer_ids:
            task = asyncio.create_task(self._request_vote(peer_id))
            vote_tasks.append(task)
        
        # Wait for votes or timeout
        timeout_task = asyncio.create_task(asyncio.sleep(self.election_timeout))
        done, pending = await asyncio.wait(
            vote_tasks + [timeout_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
        
        # Check if we got majority
        if len(self.votes_received) > len(self.peer_ids) / 2:
            print(f"Node {self.node_id}: Won election with {len(self.votes_received)} votes")
            self.state = NodeState.LEADER
            self._initialize_leader_state()
        else:
            print(f"Node {self.node_id}: Lost election, becoming follower")
            self.state = NodeState.FOLLOWER
    
    async def _leader_loop(self):
        """Leader state behavior"""
        while self.state == NodeState.LEADER and self.running:
            # Send heartbeats to all followers
            heartbeat_tasks = []
            for peer_id in self.peer_ids:
                task = asyncio.create_task(self._send_heartbeat(peer_id))
                heartbeat_tasks.append(task)
            
            if heartbeat_tasks:
                await asyncio.gather(*heartbeat_tasks, return_exceptions=True)
            
            await asyncio.sleep(self.heartbeat_interval)
    
    def _initialize_leader_state(self):
        """Initialize leader-specific state"""
        last_log_index = len(self.log)
        for peer_id in self.peer_ids:
            self.next_index[peer_id] = last_log_index + 1
            self.match_index[peer_id] = 0
    
    async def _request_vote(self, peer_id: str) -> bool:
        """Request vote from peer"""
        try:
            # In real implementation, this would be a network call
            # For simulation, we'll assume some votes are granted
            await asyncio.sleep(random.uniform(0.01, 0.05))
            
            # Simulate vote response
            if random.random() > 0.3:  # 70% chance of granting vote
                self.votes_received.add(peer_id)
                return True
            return False
        except Exception:
            return False
    
    async def _send_heartbeat(self, peer_id: str):
        """Send heartbeat to follower"""
        try:
            prev_log_index = self.next_index[peer_id] - 1
            prev_log_term = self.log[prev_log_index].term if prev_log_index > 0 else 0
            
            # In real implementation, this would be AppendEntries RPC
            await asyncio.sleep(random.uniform(0.001, 0.01))
            
            # Simulate successful heartbeat
            if random.random() > 0.1:  # 90% success rate
                self.match_index[peer_id] = len(self.log)
        except Exception:
            pass
    
    async def append_entry(self, command: Any) -> bool:
        """Append new entry to log (only for leader)"""
        if self.state != NodeState.LEADER:
            return False
        
        # Create new log entry
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log) + 1,
            command=command
        )
        
        self.log.append(entry)
        
        # Replicate to followers
        replication_tasks = []
        for peer_id in self.peer_ids:
            task = asyncio.create_task(self._replicate_to_peer(peer_id, entry))
            replication_tasks.append(task)
        
        if replication_tasks:
            results = await asyncio.gather(*replication_tasks, return_exceptions=True)
            
            # Check if majority replicated
            successful_replications = sum(1 for result in results if result is True)
            if successful_replications >= len(self.peer_ids) / 2:
                entry.committed = True
                self.commit_index = entry.index
                return True
        
        return False
    
    async def _replicate_to_peer(self, peer_id: str, entry: LogEntry) -> bool:
        """Replicate entry to specific peer"""
        try:
            # In real implementation, this would be AppendEntries RPC
            await asyncio.sleep(random.uniform(0.01, 0.05))
            
            # Simulate replication success
            if random.random() > 0.2:  # 80% success rate
                self.match_index[peer_id] = entry.index
                return True
            return False
        except Exception:
            return False
    
    async def _apply_committed_entries(self):
        """Apply committed entries to state machine"""
        while self.running:
            if self.last_applied < self.commit_index:
                for i in range(self.last_applied + 1, self.commit_index + 1):
                    if i <= len(self.log):
                        entry = self.log[i - 1]
                        if entry.committed:
                            # Apply to state machine
                            print(f"Node {self.node_id}: Applying command {entry.command}")
                            self.last_applied = i
            
            await asyncio.sleep(0.1)
    
    def stop(self):
        """Stop the node"""
        self.running = False

# Example usage
async def raft_example():
    # Create 5-node Raft cluster
    node_ids = ['node1', 'node2', 'node3', 'node4', 'node5']
    nodes = {}
    
    for node_id in node_ids:
        other_nodes = [id for id in node_ids if id != node_id]
        nodes[node_id] = RaftNode(node_id, other_nodes)
    
    # Start all nodes
    node_tasks = []
    for node in nodes.values():
        task = asyncio.create_task(node.start())
        node_tasks.append(task)
    
    # Let the cluster stabilize
    await asyncio.sleep(2)
    
    # Find leader and send some commands
    leader = None
    for node in nodes.values():
        if node.state == NodeState.LEADER:
            leader = node
            break
    
    if leader:
        print(f"Leader is {leader.node_id}")
        
        # Send some commands
        for i in range(5):
            command = f"command_{i}"
            success = await leader.append_entry(command)
            print(f"Command {command} committed: {success}")
            await asyncio.sleep(0.5)
    
    # Stop all nodes
    for node in nodes.values():
        node.stop()
    
    # Cancel node tasks
    for task in node_tasks:
        task.cancel()
```

### **2. Byzantine Fault Tolerance (BFT)**

BFT algorithms handle malicious or arbitrary failures, where nodes may behave unpredictably.

**PBFT (Practical Byzantine Fault Tolerance)**:

```python
import hashlib
import json
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class Message:
    type: str
    view: int
    sequence: int
    digest: str
    node_id: str
    timestamp: float

class PBFTNode:
    def __init__(self, node_id: str, total_nodes: int, f: int):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.f = f  # Maximum number of faulty nodes
        self.view = 0
        self.sequence = 0
        
        # Message logs
        self.prepare_messages: Dict[str, List[Message]] = {}
        self.commit_messages: Dict[str, List[Message]] = {}
        
        # State
        self.requests_log: List[Dict] = []
        self.committed_requests: Set[str] = set()
        
        self.is_primary = node_id == 'node_0'
    
    def create_digest(self, request: Dict) -> str:
        """Create digest for request"""
        return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    
    async def handle_request(self, request: Dict) -> bool:
        """Handle client request (primary only)"""
        if not self.is_primary:
            return False
        
        digest = self.create_digest(request)
        
        # Skip if already processed
        if digest in self.committed_requests:
            return True
        
        # Add to log
        self.requests_log.append(request)
        
        # Send PRE-PREPARE to all replicas
        pre_prepare_msg = Message(
            type='PRE-PREPARE',
            view=self.view,
            sequence=self.sequence,
            digest=digest,
            node_id=self.node_id,
            timestamp=time.time()
        )
        
        # Broadcast to all replicas
        await self._broadcast_message(pre_prepare_msg)
        
        self.sequence += 1
        return True
    
    async def handle_pre_prepare(self, message: Message, request: Dict) -> bool:
        """Handle PRE-PREPARE message"""
        if message.node_id == self.node_id:
            return False
        
        # Validate message
        if not self._validate_pre_prepare(message, request):
            return False
        
        # Send PREPARE to all replicas
        prepare_msg = Message(
            type='PREPARE',
            view=message.view,
            sequence=message.sequence,
            digest=message.digest,
            node_id=self.node_id,
            timestamp=time.time()
        )
        
        await self._broadcast_message(prepare_msg)
        return True
    
    async def handle_prepare(self, message: Message) -> bool:
        """Handle PREPARE message"""
        if message.digest not in self.prepare_messages:
            self.prepare_messages[message.digest] = []
        
        # Avoid duplicate messages
        if any(msg.node_id == message.node_id for msg in self.prepare_messages[message.digest]):
            return False
        
        self.prepare_messages[message.digest].append(message)
        
        # Check if we have enough PREPARE messages (2f)
        if len(self.prepare_messages[message.digest]) >= 2 * self.f:
            # Send COMMIT
            commit_msg = Message(
                type='COMMIT',
                view=message.view,
                sequence=message.sequence,
                digest=message.digest,
                node_id=self.node_id,
                timestamp=time.time()
            )
            
            await self._broadcast_message(commit_msg)
            return True
        
        return False
    
    async def handle_commit(self, message: Message) -> bool:
        """Handle COMMIT message"""
        if message.digest not in self.commit_messages:
            self.commit_messages[message.digest] = []
        
        # Avoid duplicate messages
        if any(msg.node_id == message.node_id for msg in self.commit_messages[message.digest]):
            return False
        
        self.commit_messages[message.digest].append(message)
        
        # Check if we have enough COMMIT messages (2f+1)
        if len(self.commit_messages[message.digest]) >= 2 * self.f + 1:
            # Execute request
            await self._execute_request(message.digest)
            return True
        
        return False
    
    def _validate_pre_prepare(self, message: Message, request: Dict) -> bool:
        """Validate PRE-PREPARE message"""
        # Check if digest matches request
        expected_digest = self.create_digest(request)
        if message.digest != expected_digest:
            return False
        
        # Check sequence number
        if message.sequence != self.sequence:
            return False
        
        # Check view
        if message.view != self.view:
            return False
        
        return True
    
    async def _execute_request(self, digest: str):
        """Execute committed request"""
        if digest in self.committed_requests:
            return
        
        self.committed_requests.add(digest)
        print(f"Node {self.node_id}: Committed request with digest {digest[:8]}...")
    
    async def _broadcast_message(self, message: Message):
        """Broadcast message to all nodes"""
        # In real implementation, this would send over network
        print(f"Node {self.node_id}: Broadcasting {message.type} for sequence {message.sequence}")
```

### **3. Distributed Locks**

Distributed locks ensure mutual exclusion across multiple nodes.

**Redis-based Distributed Lock**:

```python
import redis
import uuid
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

class DistributedLock:
    def __init__(self, redis_client: redis.Redis, key: str, timeout: int = 10):
        self.redis = redis_client
        self.key = f"lock:{key}"
        self.timeout = timeout
        self.identifier = str(uuid.uuid4())
        self.acquired = False
    
    async def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire distributed lock"""
        end_time = time.time() + (timeout or self.timeout)
        
        while time.time() < end_time:
            # Try to acquire lock
            if self.redis.set(self.key, self.identifier, nx=True, ex=self.timeout):
                self.acquired = True
                return True
            
            if not blocking:
                return False
            
            # Wait before retry
            await asyncio.sleep(0.001)
        
        return False
    
    async def release(self) -> bool:
        """Release distributed lock"""
        if not self.acquired:
            return False
        
        # Use Lua script to atomically check and release
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        
        result = self.redis.eval(script, 1, self.key, self.identifier)
        self.acquired = False
        return result == 1
    
    async def extend(self, additional_time: int = 10) -> bool:
        """Extend lock timeout"""
        if not self.acquired:
            return False
        
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('expire', KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        
        result = self.redis.eval(script, 1, self.key, self.identifier, additional_time)
        return result == 1
    
    @asynccontextmanager
    async def context(self, timeout: Optional[float] = None):
        """Context manager for lock"""
        try:
            acquired = await self.acquire(timeout=timeout)
            if not acquired:
                raise TimeoutError("Could not acquire lock")
            yield
        finally:
            await self.release()

# Usage example
async def distributed_lock_example():
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    async def worker(worker_id: str):
        lock = DistributedLock(redis_client, "critical_resource", timeout=5)
        
        async with lock.context(timeout=2):
            print(f"Worker {worker_id} acquired lock")
            # Simulate work
            await asyncio.sleep(1)
            print(f"Worker {worker_id} finished work")
    
    # Start multiple workers
    workers = [worker(f"worker_{i}") for i in range(5)]
    await asyncio.gather(*workers)
```

## 🎯 Leader Election Patterns

### **Zookeeper-style Leader Election**

```python
import asyncio
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

@dataclass
class ZNode:
    path: str
    data: str
    ephemeral: bool = False
    sequence: bool = False
    created_time: float = None
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = time.time()

class MockZooKeeper:
    """Mock ZooKeeper for demonstration"""
    
    def __init__(self):
        self.nodes: Dict[str, ZNode] = {}
        self.watchers: Dict[str, Set[asyncio.Future]] = {}
        self.sequence_counter = 0
    
    async def create(self, path: str, data: str, ephemeral: bool = False, sequence: bool = False) -> str:
        """Create znode"""
        final_path = path
        
        if sequence:
            self.sequence_counter += 1
            final_path = f"{path}{self.sequence_counter:010d}"
        
        node = ZNode(path=final_path, data=data, ephemeral=ephemeral, sequence=sequence)
        self.nodes[final_path] = node
        
        # Notify watchers
        await self._notify_watchers(path)
        
        return final_path
    
    async def get_children(self, path: str) -> List[str]:
        """Get children of znode"""
        children = []
        for node_path in self.nodes:
            if node_path.startswith(path + '/') and node_path != path:
                relative_path = node_path[len(path) + 1:]
                if '/' not in relative_path:
                    children.append(relative_path)
        
        return sorted(children)
    
    async def delete(self, path: str):
        """Delete znode"""
        if path in self.nodes:
            del self.nodes[path]
            await self._notify_watchers(path)
    
    async def exists(self, path: str) -> bool:
        """Check if znode exists"""
        return path in self.nodes
    
    async def watch(self, path: str) -> asyncio.Future:
        """Watch for changes to znode"""
        if path not in self.watchers:
            self.watchers[path] = set()
        
        future = asyncio.Future()
        self.watchers[path].add(future)
        
        return future
    
    async def _notify_watchers(self, path: str):
        """Notify watchers about changes"""
        if path in self.watchers:
            for future in self.watchers[path]:
                if not future.done():
                    future.set_result(True)
            self.watchers[path].clear()

class LeaderElection:
    def __init__(self, zk: MockZooKeeper, election_path: str, node_id: str):
        self.zk = zk
        self.election_path = election_path
        self.node_id = node_id
        self.znode_path: Optional[str] = None
        self.is_leader = False
        self.leader_callbacks: List[callable] = []
        self.follower_callbacks: List[callable] = []
    
    async def start_election(self):
        """Start leader election process"""
        # Create ephemeral sequential znode
        self.znode_path = await self.zk.create(
            f"{self.election_path}/candidate_",
            self.node_id,
            ephemeral=True,
            sequence=True
        )
        
        print(f"Node {self.node_id}: Created election znode {self.znode_path}")
        
        # Start monitoring
        await self._check_leadership()
    
    async def _check_leadership(self):
        """Check if this node is the leader"""
        children = await self.zk.get_children(self.election_path)
        
        if not children:
            return
        
        # Sort children to find the leader
        sorted_children = sorted(children)
        leader_path = f"{self.election_path}/{sorted_children[0]}"
        
        # Check if we are the leader
        if self.znode_path == leader_path:
            if not self.is_leader:
                self.is_leader = True
                print(f"Node {self.node_id}: Became leader")
                
                # Notify callbacks
                for callback in self.leader_callbacks:
                    await callback()
        else:
            if self.is_leader:
                self.is_leader = False
                print(f"Node {self.node_id}: Lost leadership")
                
                # Notify callbacks
                for callback in self.follower_callbacks:
                    await callback()
            
            # Watch the predecessor
            my_index = sorted_children.index(self.znode_path.split('/')[-1])
            if my_index > 0:
                predecessor = sorted_children[my_index - 1]
                predecessor_path = f"{self.election_path}/{predecessor}"
                
                # Watch for predecessor deletion
                watch_future = await self.zk.watch(predecessor_path)
                
                async def wait_for_predecessor():
                    await watch_future
                    # Predecessor is gone, re-check leadership
                    await self._check_leadership()
                
                asyncio.create_task(wait_for_predecessor())
    
    def on_leader(self, callback: callable):
        """Register callback for becoming leader"""
        self.leader_callbacks.append(callback)
    
    def on_follower(self, callback: callable):
        """Register callback for becoming follower"""
        self.follower_callbacks.append(callback)
    
    async def stop(self):
        """Stop participating in election"""
        if self.znode_path:
            await self.zk.delete(self.znode_path)
        
        self.is_leader = False

# Usage example
async def leader_election_example():
    zk = MockZooKeeper()
    election_path = "/election"
    
    # Create multiple nodes
    nodes = []
    for i in range(5):
        node_id = f"node_{i}"
        election = LeaderElection(zk, election_path, node_id)
        
        async def leader_callback():
            print(f"Node {node_id}: I am now the leader!")
        
        async def follower_callback():
            print(f"Node {node_id}: I am now a follower!")
        
        election.on_leader(leader_callback)
        election.on_follower(follower_callback)
        
        nodes.append(election)
    
    # Start all elections
    for node in nodes:
        await node.start_election()
    
    # Let them run for a bit
    await asyncio.sleep(2)
    
    # Simulate leader failure
    leader_node = None
    for node in nodes:
        if node.is_leader:
            leader_node = node
            break
    
    if leader_node:
        print(f"Stopping leader node {leader_node.node_id}")
        await leader_node.stop()
    
    # Let new leader emerge
    await asyncio.sleep(1)
    
    # Stop all nodes
    for node in nodes:
        await node.stop()
```

## 📊 Distributed Coordination Services

### **Service Discovery Pattern**

```python
import asyncio
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import json

@dataclass
class ServiceInstance:
    service_name: str
    instance_id: str
    address: str
    port: int
    metadata: Dict = None
    health_check_url: Optional[str] = None
    last_heartbeat: float = None
    status: str = "healthy"
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_heartbeat is None:
            self.last_heartbeat = time.time()

class ServiceRegistry:
    def __init__(self, heartbeat_timeout: float = 30):
        self.services: Dict[str, Dict[str, ServiceInstance]] = {}
        self.watchers: Dict[str, Set[asyncio.Future]] = {}
        self.heartbeat_timeout = heartbeat_timeout
        self.running = False
    
    async def register(self, instance: ServiceInstance) -> bool:
        """Register service instance"""
        if instance.service_name not in self.services:
            self.services[instance.service_name] = {}
        
        self.services[instance.service_name][instance.instance_id] = instance
        
        print(f"Registered service: {instance.service_name}:{instance.instance_id} at {instance.address}:{instance.port}")
        
        # Notify watchers
        await self._notify_watchers(instance.service_name)
        
        return True
    
    async def deregister(self, service_name: str, instance_id: str) -> bool:
        """Deregister service instance"""
        if service_name in self.services and instance_id in self.services[service_name]:
            del self.services[service_name][instance_id]
            
            # Clean up empty service
            if not self.services[service_name]:
                del self.services[service_name]
            
            print(f"Deregistered service: {service_name}:{instance_id}")
            
            # Notify watchers
            await self._notify_watchers(service_name)
            
            return True
        
        return False
    
    async def discover(self, service_name: str) -> List[ServiceInstance]:
        """Discover healthy instances of a service"""
        if service_name not in self.services:
            return []
        
        healthy_instances = []
        for instance in self.services[service_name].values():
            if instance.status == "healthy":
                healthy_instances.append(instance)
        
        return healthy_instances
    
    async def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """Update heartbeat for service instance"""
        if service_name in self.services and instance_id in self.services[service_name]:
            instance = self.services[service_name][instance_id]
            instance.last_heartbeat = time.time()
            instance.status = "healthy"
            return True
        
        return False
    
    async def watch(self, service_name: str) -> asyncio.Future:
        """Watch for changes to service instances"""
        if service_name not in self.watchers:
            self.watchers[service_name] = set()
        
        future = asyncio.Future()
        self.watchers[service_name].add(future)
        
        return future
    
    async def start_health_checker(self):
        """Start background health checker"""
        self.running = True
        
        while self.running:
            await self._check_health()
            await asyncio.sleep(5)
    
    async def _check_health(self):
        """Check health of all registered instances"""
        current_time = time.time()
        
        for service_name in list(self.services.keys()):
            for instance_id in list(self.services[service_name].keys()):
                instance = self.services[service_name][instance_id]
                
                # Check if instance is stale
                if current_time - instance.last_heartbeat > self.heartbeat_timeout:
                    if instance.status == "healthy":
                        instance.status = "unhealthy"
                        print(f"Marked instance unhealthy: {service_name}:{instance_id}")
                        
                        # Notify watchers
                        await self._notify_watchers(service_name)
                    
                    # Remove if stale for too long
                    if current_time - instance.last_heartbeat > self.heartbeat_timeout * 2:
                        await self.deregister(service_name, instance_id)
    
    async def _notify_watchers(self, service_name: str):
        """Notify watchers about service changes"""
        if service_name in self.watchers:
            for future in self.watchers[service_name]:
                if not future.done():
                    future.set_result(True)
            self.watchers[service_name].clear()
    
    def stop(self):
        """Stop the registry"""
        self.running = False

class ServiceClient:
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
    
    async def call_service(self, service_name: str, endpoint: str, data: Dict = None) -> Dict:
        """Make request to service using load balancing"""
        instances = await self.registry.discover(service_name)
        
        if not instances:
            raise Exception(f"No healthy instances found for service: {service_name}")
        
        # Simple round-robin load balancing
        import random
        instance = random.choice(instances)
        
        try:
            # Simulate API call
            await asyncio.sleep(0.1)
            
            # Simulate occasional failures
            if random.random() < 0.1:
                raise Exception("Service call failed")
            
            return {
                "status": "success",
                "data": f"Response from {instance.instance_id}",
                "endpoint": endpoint
            }
        
        except Exception as e:
            # Could implement retry logic here
            raise Exception(f"Failed to call {service_name}: {e}")

# Usage example
async def service_discovery_example():
    registry = ServiceRegistry(heartbeat_timeout=10)
    
    # Start health checker
    health_checker_task = asyncio.create_task(registry.start_health_checker())
    
    # Register multiple service instances
    instances = [
        ServiceInstance("user-service", "user-1", "localhost", 8001),
        ServiceInstance("user-service", "user-2", "localhost", 8002),
        ServiceInstance("order-service", "order-1", "localhost", 8003),
        ServiceInstance("order-service", "order-2", "localhost", 8004),
    ]
    
    for instance in instances:
        await registry.register(instance)
    
    # Simulate heartbeats
    async def send_heartbeats():
        while True:
            for instance in instances:
                await registry.heartbeat(instance.service_name, instance.instance_id)
            await asyncio.sleep(5)
    
    heartbeat_task = asyncio.create_task(send_heartbeats())
    
    # Create client and make some requests
    client = ServiceClient(registry)
    
    for i in range(10):
        try:
            response = await client.call_service("user-service", f"/users/{i}")
            print(f"Request {i}: {response}")
        except Exception as e:
            print(f"Request {i} failed: {e}")
        
        await asyncio.sleep(0.5)
    
    # Stop one instance
    await registry.deregister("user-service", "user-1")
    
    # Make more requests
    for i in range(5):
        try:
            response = await client.call_service("user-service", f"/users/{i}")
            print(f"After deregister {i}: {response}")
        except Exception as e:
            print(f"After deregister {i} failed: {e}")
        
        await asyncio.sleep(0.5)
    
    # Cleanup
    registry.stop()
    heartbeat_task.cancel()
    health_checker_task.cancel()
```

## 🎯 Best Practices

### **Consensus Algorithm Selection**

| Algorithm | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **Raft** | Simple distributed systems | Easy to understand, proven | Leader bottleneck |
| **PBFT** | Byzantine fault tolerance | Handles malicious nodes | High message complexity |
| **Paxos** | Strong consistency | Theoretical foundation | Complex to implement |

### **Implementation Guidelines**

1. **Network Partitions**: Handle split-brain scenarios
2. **Timeouts**: Use appropriate timeout values
3. **Retries**: Implement exponential backoff
4. **Monitoring**: Track consensus participation
5. **Testing**: Use chaos engineering

### **Performance Considerations**

1. **Batch Operations**: Group operations for efficiency
2. **Asynchronous Processing**: Avoid blocking operations
3. **Network Optimization**: Minimize round trips
4. **State Compression**: Use snapshots for large states

---

*"In distributed systems, consensus is not about getting everyone to agree on everything, but about getting everyone to agree on the things that matter most."*
