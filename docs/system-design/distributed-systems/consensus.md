# Consensus Algorithms

**Reaching agreement in distributed systems** | 🤝 Agreement | 🔄 Replication | 💪 Fault Tolerance

---

## Overview

Consensus algorithms allow multiple nodes in a distributed system to agree on a single value, even when some nodes fail or the network is unreliable.

**The Problem:** How do distributed nodes agree on state when they can't all communicate reliably?

---

## Why Consensus Matters

=== "Use Cases"
    **Critical distributed system problems:**

    | Problem | Solution | Example |
    |---------|----------|---------|
    | **Leader Election** | Choose one node as leader | Elect primary database |
    | **Distributed Locks** | Coordinate access to resource | Prevent double-booking |
    | **Configuration Management** | Agree on system config | Service discovery (Consul) |
    | **Database Replication** | Keep replicas consistent | MySQL with Galera |
    | **Distributed Transactions** | Commit or abort together | 2PC in databases |

=== "The Challenge"
    **What makes consensus hard:**

    ```
    Perfect Network (doesn't exist):
    Node A ──────────▶ Node B
              1ms
              100% reliable

    Real Network (what we have):
    Node A ─ ─ ─ ─ ─ ▶ Node B
              100ms
              Messages lost
              Messages reordered
              Nodes crash
              Network partitions
    ```

    **CAP Theorem:**
    - **C**onsistency: All nodes see same data
    - **A**vailability: Every request gets response
    - **P**artition tolerance: Works despite network splits

    **Pick 2 out of 3** (must always have P in distributed systems)

---

## Paxos

=== "Overview"
    **The original consensus algorithm (1989)**

    **Key Idea:** Use majority voting with proposals and promises

    **Roles:**
    - **Proposer:** Proposes values
    - **Acceptor:** Votes on proposals
    - **Learner:** Learns chosen value

    ```
    Phase 1 (Prepare):
    Proposer → All Acceptors: "Can I propose value X?"
    Acceptors → Proposer: "Yes" or "No, someone else already proposed"

    Phase 2 (Accept):
    Proposer → Majority: "Please accept value X"
    Acceptors → Proposer: "Accepted!"

    Result: Value X is chosen (permanent)
    ```

=== "How It Works"
    **Two-phase commit process:**

    ```
    Scenario: 5 nodes need to agree on a value

    Node 1 (Proposer):
      Prepare(proposal #10, value="foo")
      ↓
    Nodes 2, 3, 4, 5 (Acceptors):
      "OK, I promise not to accept proposals < #10"
      (3 out of 4 responded - majority!)
      ↓
    Node 1:
      Accept(proposal #10, value="foo")
      ↓
    Nodes 2, 3, 4:
      "Accepted!"
      (3 out of 4 accepted - value chosen!)
    ```

    **Guarantees:**
    - ✅ Only one value chosen
    - ✅ Survives minority node failures
    - ✅ Progress with majority

=== "Example"
    **Real-world scenario:**

    ```python
    class PaxosNode:
        def __init__(self, node_id):
            self.node_id = node_id
            self.promised_proposal = None
            self.accepted_proposal = None
            self.accepted_value = None

        def prepare(self, proposal_number):
            """Phase 1: Prepare request"""
            if self.promised_proposal is None or \
               proposal_number > self.promised_proposal:
                self.promised_proposal = proposal_number
                return {
                    'promise': True,
                    'accepted_proposal': self.accepted_proposal,
                    'accepted_value': self.accepted_value
                }
            return {'promise': False}

        def accept(self, proposal_number, value):
            """Phase 2: Accept request"""
            if proposal_number >= self.promised_proposal:
                self.accepted_proposal = proposal_number
                self.accepted_value = value
                return {'accepted': True}
            return {'accepted': False}

    # Usage
    nodes = [PaxosNode(i) for i in range(5)]

    # Phase 1: Prepare
    proposal_num = 10
    promises = []
    for node in nodes:
        response = node.prepare(proposal_num)
        if response['promise']:
            promises.append(response)

    # Need majority (3 out of 5)
    if len(promises) >= 3:
        # Phase 2: Accept
        value = "leader_node_3"
        accepts = []
        for node in nodes:
            response = node.accept(proposal_num, value)
            if response['accepted']:
                accepts.append(response)

        if len(accepts) >= 3:
            print(f"Consensus reached: {value}")
    ```

=== "Challenges"
    **Why Paxos is hard:**

    - ❌ Complex to understand and implement
    - ❌ Doesn't handle membership changes well
    - ❌ Livelock possible (competing proposers)
    - ❌ Performance issues under contention

    **Quote from Leslie Lamport (creator):**
    > "The Paxos algorithm, when presented in plain English, is very simple."

    **Reality:** Most engineers disagree! 😅

---

## Raft

=== "Overview"
    **Designed for understandability (2014)**

    **Key Innovation:** Decompose consensus into:
    1. **Leader Election:** Choose one leader
    2. **Log Replication:** Leader replicates log to followers
    3. **Safety:** Ensure chosen values never change

    ```
    Normal Operation:
    
    Leader (Node 1):
      Receives client requests
      Appends to local log
      Replicates to followers
      Commits when majority confirm
    
    Followers (Nodes 2, 3, 4, 5):
      Accept log entries from leader
      Respond with confirmation
      Apply committed entries
    ```

=== "Leader Election"
    **How leaders are elected:**

    ```
    State Machine (each node):
    
    Follower ────timeout────▶ Candidate ────wins election────▶ Leader
       ▲                           │                              │
       │                           │ loses/new term               │
       └───────────────────────────┴──────────────────────────────┘
    
    Process:
    1. Follower times out (150-300ms, no heartbeat from leader)
    2. Becomes Candidate, increments term, votes for self
    3. Requests votes from other nodes
    4. If gets majority: becomes Leader
    5. If another leader elected: becomes Follower
    6. If timeout: starts new election
    ```

    **Election Example:**
    ```
    Initial: All followers
    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
    │  F  │ │  F  │ │  F  │ │  F  │ │  F  │
    └─────┘ └─────┘ └─────┘ └─────┘ └─────┘

    Node 3 times out:
    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
    │  F  │ │  F  │ │  C  │ │  F  │ │  F  │
    └─────┘ └─────┘ └──┬──┘ └─────┘ └─────┘
                       │
                  Vote for me!
                       │
    ┌─────────────┬────┴────┬─────────────┐
    ▼             ▼          ▼             ▼
    Yes          Yes        No            Yes
    
    3 votes (majority) → Node 3 becomes Leader
    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
    │  F  │ │  F  │ │  L  │ │  F  │ │  F  │
    └─────┘ └─────┘ └─────┘ └─────┘ └─────┘
    ```

=== "Log Replication"
    **Replicating state changes:**

    ```
    Client Request: SET x = 5
    
    Leader Log:
    [SET x=5] ← New entry (uncommitted)
        ↓
    Send to Followers:
    AppendEntries(entry: "SET x=5", index: 10)
        ↓
    Followers append to log:
    Follower 1: [... SET x=5] ✅
    Follower 2: [... SET x=5] ✅
    Follower 3: [... SET x=5] ✅
        ↓
    Majority confirmed (3/5)
        ↓
    Leader commits entry:
    [SET x=5] ✓ Committed
        ↓
    Leader applies to state machine:
    x = 5 (visible to clients)
        ↓
    Next heartbeat tells followers to commit
    ```

=== "Implementation"
    **Raft implementation example:**

    ```python
    class RaftNode:
        def __init__(self, node_id, peers):
            self.node_id = node_id
            self.peers = peers
            self.state = 'follower'  # follower, candidate, or leader
            self.current_term = 0
            self.voted_for = None
            self.log = []
            self.commit_index = 0
            
        def start_election(self):
            """Start election when timeout occurs"""
            self.state = 'candidate'
            self.current_term += 1
            self.voted_for = self.node_id
            votes_received = 1  # Vote for self
            
            # Request votes from peers
            for peer in self.peers:
                response = peer.request_vote(
                    term=self.current_term,
                    candidate_id=self.node_id,
                    last_log_index=len(self.log) - 1,
                    last_log_term=self.log[-1]['term'] if self.log else 0
                )
                if response['vote_granted']:
                    votes_received += 1
            
            # Check if won election
            if votes_received > len(self.peers) // 2:
                self.become_leader()
        
        def become_leader(self):
            """Transition to leader state"""
            self.state = 'leader'
            print(f"Node {self.node_id} became leader for term {self.current_term}")
            
            # Start sending heartbeats
            self.send_heartbeats()
        
        def send_heartbeats(self):
            """Send periodic heartbeats to maintain leadership"""
            for peer in self.peers:
                peer.append_entries(
                    term=self.current_term,
                    leader_id=self.node_id,
                    entries=[]  # Empty for heartbeat
                )
        
        def replicate_log(self, entry):
            """Replicate log entry to followers"""
            # Append to leader's log
            entry['term'] = self.current_term
            self.log.append(entry)
            
            # Replicate to followers
            acks = 1  # Leader counts as ack
            for peer in self.peers:
                response = peer.append_entries(
                    term=self.current_term,
                    leader_id=self.node_id,
                    entries=[entry]
                )
                if response['success']:
                    acks += 1
            
            # Commit if majority acknowledged
            if acks > len(self.peers) // 2:
                self.commit_index = len(self.log) - 1
                return True
            return False
    ```

=== "Advantages"
    **Why Raft is popular:**

    - ✅ Easier to understand than Paxos
    - ✅ Clear leader makes operations simpler
    - ✅ Handles membership changes cleanly
    - ✅ Strong guarantees (linearizability)
    - ✅ Many production implementations

    **Used by:**
    - **etcd:** Kubernetes configuration
    - **Consul:** Service discovery
    - **CockroachDB:** Distributed SQL
    - **TiDB:** Distributed database

---

## 2PC (Two-Phase Commit)

=== "Overview"
    **Atomic commit protocol for distributed transactions**

    **Goal:** Either all nodes commit or all abort

    ```
    Coordinator:
      Prepare Phase → Ask all: "Can you commit?"
      Commit Phase → Tell all: "Commit" or "Abort"
    
    Participants:
      Prepare → Vote Yes/No
      Commit → Execute commit or abort
    ```

=== "How It Works"
    **Two phases:**

    ```
    Example: Transfer $100 from Account A to Account B
             (A on Server 1, B on Server 2)

    Phase 1: Prepare
    Coordinator: "Can you transfer?"
    ├─▶ Server 1: Deduct $100 from A → "YES, ready"
    └─▶ Server 2: Add $100 to B → "YES, ready"

    Both said YES → Proceed to Phase 2

    Phase 2: Commit
    Coordinator: "COMMIT"
    ├─▶ Server 1: Commit transaction ✅
    └─▶ Server 2: Commit transaction ✅

    Result: Transfer complete!

    ---

    Alternative: One says NO

    Phase 1: Prepare
    Coordinator: "Can you transfer?"
    ├─▶ Server 1: Deduct $100 from A → "YES, ready"
    └─▶ Server 2: Add $100 to B → "NO, insufficient space"

    One said NO → Abort

    Phase 2: Abort
    Coordinator: "ABORT"
    ├─▶ Server 1: Rollback transaction ✅
    └─▶ Server 2: Nothing to rollback ✅

    Result: Transfer canceled
    ```

=== "Problems"
    **Why 2PC is problematic:**

    **Blocking Problem:**
    ```
    Phase 1: All participants vote YES
    Phase 2: Coordinator crashes before sending COMMIT
    
    Result: Participants stuck waiting!
           - Can't commit (didn't get command)
           - Can't abort (voted YES, might commit)
           - Locks held indefinitely ❌
    ```

    **Performance Issues:**
    - Multiple round trips (high latency)
    - Blocking nature (locks held during protocol)
    - Single point of failure (coordinator)

    **Why it's still used:**
    - Simple to understand and implement
    - Works well in reliable networks
    - Good for small number of participants

---

## 3PC (Three-Phase Commit)

=== "Overview"
    **Non-blocking variant of 2PC**

    **Phases:**
    1. **CanCommit:** Ask if ready
    2. **PreCommit:** Tell to prepare
    3. **DoCommit:** Tell to commit

    **Advantage:** Can make progress even if coordinator fails

=== "How It Works"
    ```
    Phase 1: CanCommit
    Coordinator → Participants: "Can you commit?"
    Participants → Coordinator: "YES" or "NO"

    Phase 2: PreCommit
    Coordinator → Participants: "Prepare to commit"
    Participants: Lock resources, write to log
    Participants → Coordinator: "Ready"

    Phase 3: DoCommit
    Coordinator → Participants: "Commit!"
    Participants: Commit and release locks
    ```

    **Key difference:** PreCommit phase allows timeout-based recovery

=== "Limitations"
    - ❌ Still has edge cases in network partitions
    - ❌ More complex than 2PC
    - ❌ Higher latency (three phases vs two)
    - ✅ Rarely used in practice (Raft/Paxos preferred)

---

## Comparison Table

| Algorithm | Year | Complexity | Fault Tolerance | Performance | Use Case |
|-----------|------|------------|----------------|-------------|----------|
| **Paxos** | 1989 | Very High | Excellent | Medium | Theoretical foundation |
| **Raft** | 2014 | Medium | Excellent | Good | Production systems |
| **2PC** | 1970s | Low | Poor (blocking) | Poor | Small, reliable networks |
| **3PC** | 1980s | Medium | Better | Poor | Rarely used |

---

## Real-World Usage

=== "Raft Deployments"
    **etcd (Kubernetes):**
    ```bash
    # etcd cluster (Raft for consensus)
    etcd --name node1 \
         --initial-cluster node1=http://10.0.0.1:2380,node2=http://10.0.0.2:2380 \
         --initial-cluster-state new
    
    # Kubernetes uses etcd for:
    # - Pod configurations
    # - Service discovery
    # - Cluster state
    ```

    **Consul (Service Discovery):**
    ```bash
    # Consul servers use Raft
    consul agent -server -bootstrap-expect=3 \
         -data-dir=/tmp/consul
    
    # Leader election automatic
    # Configuration changes replicated
    ```

=== "Paxos Deployments"
    **Google Chubby:**
    - Lock service for Google infrastructure
    - Used by BigTable, GFS
    - Multi-Paxos variant

    **Apache Cassandra:**
    - Lightweight transactions (LWT) use Paxos
    - Opt-in for critical operations
    - Most operations use eventual consistency

=== "2PC Deployments"
    **Traditional Databases:**
    ```sql
    -- PostgreSQL distributed transaction
    BEGIN;
    -- Operations on local DB
    PREPARE TRANSACTION 'xact_1';

    -- On remote DB
    BEGIN;
    -- Operations on remote DB
    PREPARE TRANSACTION 'xact_1';

    -- Commit both
    COMMIT PREPARED 'xact_1';
    ```

    **Used by:** MySQL, PostgreSQL, Oracle (XA transactions)

---

## Interview Talking Points

**Q: Explain the difference between Paxos and Raft.**

✅ **Strong Answer:**
> "Both Paxos and Raft solve consensus, but Raft was explicitly designed for understandability. Paxos is more general but harder to implement correctly - it doesn't prescribe a specific leader, leading to complex coordinator election logic. Raft simplifies this by always having a single leader who handles all client requests and log replication. This makes Raft easier to reason about and implement, which is why it's more popular in production systems like etcd and Consul. However, Paxos variants like Multi-Paxos can achieve similar performance."

**Q: Why is 2PC blocking and how does Raft avoid this?**

✅ **Strong Answer:**
> "2PC is blocking because if the coordinator crashes after participants vote YES but before sending the commit decision, participants are stuck - they can't commit or abort without knowing the coordinator's decision, so they hold locks indefinitely. Raft avoids this through its log replication approach. Raft commits entries once a majority has them in their logs, so even if the leader fails, a new leader with the committed entries can be elected and operations continue. The key difference is Raft's leader election mechanism allows the system to recover automatically, while 2PC requires manual intervention or coordinator recovery."

**Q: When would you use eventual consistency instead of consensus?**

✅ **Strong Answer:**
> "I'd use eventual consistency for non-critical operations where the cost of consensus (latency, complexity) isn't worth immediate consistency. For example, social media 'likes' counts don't need consensus - it's okay if different users see slightly different numbers for a few seconds. Similarly, analytics dashboards can tolerate stale data. However, I'd use consensus for critical operations like financial transactions, inventory management, or leader election where split-brain scenarios could cause serious issues. The trade-off is: consensus provides strong guarantees but adds latency and complexity."

---

## Related Topics

- [Distributed Systems](index.md) - Overview of distributed challenges
- [Consistent Hashing](consistent-hashing.md) - Distributed data partitioning
- [Database Replication](../data/databases/replication.md) - Data consistency
- [CAP Theorem](../fundamentals/cap-theorem.md) - Consistency trade-offs

---

**Consensus is hard, but necessary for building reliable distributed systems! 🤝**
