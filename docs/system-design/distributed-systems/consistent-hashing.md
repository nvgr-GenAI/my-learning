# Consistent Hashing

When you distribute data across multiple servers, you need a way to decide which server holds which key. The naive approach -- `hash(key) % N` where N is the number of servers -- works fine until you add or remove a server. At that point, nearly every key maps to a different server, causing massive cache misses and potential system overload. Consistent hashing solves this problem by ensuring that when the server count changes, only a minimal fraction of keys need to move.

=== "The Problem"

    ## Why Modulo Hashing Fails

    Simple modulo hashing assigns each key to a server using `hash(key) % N`. With three servers, the mapping is straightforward:

    ```
    hash("user:101") % 3 = 0  --> Server A
    hash("user:202") % 3 = 1  --> Server B
    hash("user:303") % 3 = 2  --> Server C
    hash("user:404") % 3 = 1  --> Server B
    hash("user:505") % 3 = 0  --> Server A
    hash("user:606") % 3 = 2  --> Server C
    ```

    Now suppose traffic grows and you add a fourth server. The divisor changes from 3 to 4:

    ```
    hash("user:101") % 4 = 1  --> Server B  (was A)  MOVED
    hash("user:202") % 4 = 2  --> Server C  (was B)  MOVED
    hash("user:303") % 4 = 3  --> Server D  (was C)  MOVED
    hash("user:404") % 4 = 0  --> Server A  (was B)  MOVED
    hash("user:505") % 4 = 1  --> Server B  (was A)  MOVED
    hash("user:606") % 4 = 2  --> Server C  (was C)  stayed
    ```

    Five out of six keys moved to a different server. In the general case, going from N to N+1 servers remaps roughly (N-1)/N of all keys. Here is how this scales:

    ```
    Servers: 3 --> 4    Keys remapped: ~75%
    Servers: 4 --> 5    Keys remapped: ~80%
    Servers: 9 --> 10   Keys remapped: ~90%
    Servers: 99 --> 100 Keys remapped: ~99%
    ```

    The larger your cluster, the worse this gets. For a caching layer holding millions of entries, this means millions of simultaneous cache misses. Every request falls through to the database at once, a phenomenon known as a thundering herd, which can take down the entire system.

    ```
    Before (3 servers):          After adding Server D (4 servers):

    Server A: [101, 505]         Server A: [404]
    Server B: [202, 404]         Server B: [101, 505]
    Server C: [303, 606]         Server C: [202, 606]
                                 Server D: [303]

    Result: 5 of 6 keys moved --> mass cache invalidation
    ```

    Consider the real impact: if you have a Memcached cluster with 10 servers caching 50 million keys, and you need to add an 11th server, modulo hashing would invalidate approximately 45 million keys simultaneously. Every one of those keys would trigger a cache miss and a database query. If your database normally handles 10,000 queries per second, suddenly receiving 45 million queries in a short window would almost certainly cause a cascading failure.

    What distributed systems need is an approach where adding one server out of N causes only about 1/N of keys to move, not (N-1)/N.

=== "How It Works"

    ## The Hash Ring

    Consistent hashing, first described by Karger et al. in 1997, arranges the entire hash space as a circle (a "ring") from 0 to some maximum value, typically 2^32 - 1. Both servers and keys are hashed onto positions on this ring. Each key is assigned to the first server found by walking clockwise from the key's position.

    ```
                      0
                      |
              K1(50)  |  Node A (100)
                   \  |  /
                    \ | /
      360 ----------[RING]---------- 180
                    / | \
                   /  |  \
            K3(350)   |   Node B (200)
                      |
                 Node C (300)
                      |
                  K2(250)
    ```

    In this arrangement, key K1 at position 50 walks clockwise and finds Node A at position 100. Key K2 at position 250 walks clockwise and finds Node C at position 300. Key K3 at position 350 wraps around past 0 and finds Node A at position 100.

    The lookup algorithm in pseudocode is minimal:

    ```
    def get_node(key):
        position = hash(key) % RING_SIZE
        # Walk clockwise to find first node
        return first node with position >= key position
               (wrap around to lowest node if past the end)
    ```

    The critical insight emerges when a node is added or removed. Suppose you add Node D at position 150:

    ```
                      0
                      |
              K1(50)  |  Node A (100)
                   \  |  /
                    \ | /           Node D (150) <-- new
      360 ----------[RING]---------- 180
                    / | \
                   /  |  \
            K3(350)   |   Node B (200)
                      |
                 Node C (300)
                      |
                  K2(250)
    ```

    Only keys that fall between Node A (100) and Node D (150) are affected -- they now map to Node D instead of Node B. Keys K1 and K3 still map to Node A, and K2 still maps to Node C. No disruption for the vast majority of keys.

    When a node is removed, the effect is equally contained. If Node B fails, its keys (those between positions 150 and 200) simply shift to the next clockwise node, Node C. Every other key on the ring remains where it was.

    ```
    Adding Node D:                   Removing Node B:

    Before:                          Before:
      Keys 101-200 --> Node B          Keys 201-300 --> Node C

    After:                           After:
      Keys 101-150 --> Node D          Keys 151-300 --> Node C
      Keys 151-200 --> Node B          (Node B's keys absorbed by Node C)
      All other keys: unchanged        All other keys: unchanged
    ```

    The mathematical result is that when you add or remove one node from a ring with N nodes, only K/N keys need to move on average, where K is the total number of keys. This is the theoretical minimum -- you cannot do better while maintaining a balanced distribution.

=== "Virtual Nodes"

    ## Solving Uneven Distribution

    With only a handful of physical nodes on the ring, the distribution of hash space is often lopsided. Three nodes placed on a ring do not divide it into three equal arcs -- depending on where the hash function places them, one node might own 60% of the space while another owns only 15%. This problem gets worse with fewer nodes and does not reliably improve as nodes are added and removed over time.

    The solution is virtual nodes (vnodes). Instead of placing each physical server at a single point on the ring, you create multiple virtual copies of it at different positions:

    ```
    Physical Node A maps to:        Physical Node B maps to:
        A-vn1   (hash: 42)              B-vn1   (hash: 15)
        A-vn2   (hash: 198)             B-vn2   (hash: 120)
        A-vn3   (hash: 301)             B-vn3   (hash: 267)
        A-vn4   (hash: 455)             B-vn4   (hash: 389)
        ...                              ...
        A-vn200 (hash: 877)             B-vn200 (hash: 940)
    ```

    With 200 virtual nodes per physical server, the hash space becomes finely divided. Each physical node ends up owning many small arcs scattered around the ring rather than one potentially large arc. By the law of large numbers, the total space owned converges toward an even split.

    ```
    3 physical nodes, 1 point each (uneven):

    |=======A=======|====B====|===========A===========|==C==|
    Node A: ~55%     Node B: ~25%                Node C: ~20%


    3 physical nodes, 200 virtual nodes each (balanced):

    |A|B|C|A|B|C|A|B|C|A|B|C|A|B|C|A|B|C|A|B|C|A|B|C|A|B|C|
    Node A: ~33%    Node B: ~34%    Node C: ~33%
    ```

    The effect on load balance is dramatic. With 10 physical nodes and no virtual nodes, the most-loaded node typically handles 2-3x the average load. With 100 virtual nodes per physical node, the imbalance drops to roughly 10%. With 200 virtual nodes, it drops below 5%.

    The trade-off is memory and lookup time. Each virtual node requires an entry in the ring data structure (typically a sorted array or balanced tree). With 200 virtual nodes and 100 physical servers, you maintain 20,000 entries. This is easily manageable in memory, and binary search over a sorted ring gives O(log V) lookup time where V is the total number of virtual nodes. Going to thousands of virtual nodes per server provides diminishing returns on balance while increasing both memory usage and lookup cost.

    Netflix uses approximately 200 virtual nodes per physical server in their EVCache infrastructure, which provides a good balance between distribution uniformity and memory overhead. Most production systems settle on between 100 and 300 virtual nodes.

    **Weighted nodes.** When servers have different capacities (for example, one machine has 64 GB of RAM and another has 128 GB), you can assign more virtual nodes to the larger machine. A server with twice the capacity gets twice the virtual nodes and consequently handles roughly twice the key space. This is simpler and more flexible than trying to position a single node at a "better" point on the ring.

    ```
    Node A (64 GB):  100 virtual nodes  -->  ~25% of keys
    Node B (128 GB): 200 virtual nodes  -->  ~50% of keys
    Node C (64 GB):  100 virtual nodes  -->  ~25% of keys
    ```

    **Replication on the ring.** To replicate data for fault tolerance, you store each key on N successive distinct physical nodes walking clockwise from the key's position. The word "distinct" is critical: if the next three virtual nodes clockwise happen to be A-vn3, A-vn7, and B-vn2, you skip the second virtual node belonging to A and continue walking until you find N different physical servers. This ensures that replicas are spread across separate machines, so a single hardware failure does not lose all copies.

    ```
    Key X at position 150:
      Walk clockwise: A-vn3(160), A-vn7(175), B-vn2(190), C-vn5(210)
      With replication factor 3:
        Replica 1: Node A (first distinct physical node)
        Replica 2: Node B (skip A-vn7, next distinct node)
        Replica 3: Node C (next distinct node)
    ```

=== "Real-World Usage"

    ## Production Systems

    Consistent hashing is not a theoretical curiosity -- it underpins some of the largest distributed systems ever built.

    **Amazon DynamoDB** is perhaps the most influential example. The 2007 Dynamo paper described how Amazon uses consistent hashing to partition data across storage nodes. Each key is assigned to a coordinator node via the ring, with data replicated to N-1 successor nodes. DynamoDB handles millions of requests per second across Amazon's infrastructure, and consistent hashing allows nodes to be added during peak traffic (like Prime Day) without reshuffling the entire dataset.

    **Apache Cassandra** adopted the Dynamo design and uses a token ring where each node owns a range of tokens. Cassandra's virtual nodes (called vnodes, 256 per node by default) spread data evenly across the cluster. When a new node joins a Cassandra cluster, it takes over portions of token ranges from existing nodes, automatically rebalancing with minimal data movement. Clusters in production at Apple, Netflix, and Instagram handle petabytes of data using this approach.

    **Akamai CDN** has a direct historical connection to consistent hashing. The original 1997 paper by Karger, Lehman, Leighton, Panigrahy, Levine, and Lewin was motivated by the problem of distributing web content across cache servers. Akamai, co-founded by several of the paper's authors, uses consistent hashing to route user requests to the nearest and most appropriate cache server across more than 300,000 servers worldwide.

    **Discord** uses consistent hashing to route messages across its infrastructure. With millions of concurrent guilds (servers) and channels, consistent hashing determines which backend process handles which channel. When Discord scales up its backend fleet, only a fraction of channels are reassigned to new processes.

    | System | Use Case | Virtual Nodes | Scale |
    |--------|----------|---------------|-------|
    | DynamoDB | Key-value partition assignment | Configurable | Millions of req/sec |
    | Cassandra | Token ring data distribution | 256 per node (default) | Petabytes at Apple, Netflix |
    | Akamai CDN | Cache server selection | Implementation-specific | 300,000+ servers globally |
    | Discord | Message/channel routing | Implementation-specific | Millions of concurrent channels |
    | Memcached clients | Cache node selection | 100-200 typical | Widely used across industry |

    ## Alternative Algorithms

    While ring-based consistent hashing is the most widely deployed approach, several alternatives exist for specialized scenarios.

    **Jump consistent hash**, published by Lamping and Veach at Google in 2014, uses a clever pseudorandom algorithm that requires O(ln N) time and O(1) memory -- no ring data structure at all. It produces near-perfect load balance and is remarkably simple (the entire algorithm fits in about 10 lines of code). However, it only works when nodes are numbered sequentially from 0 to N-1, and it does not support arbitrary node removal. This makes it ideal for static or append-only server sets like sharded databases where servers are rarely decommissioned from the middle of the range.

    **Rendezvous hashing** (also called highest random weight, or HRW hashing) computes a hash for every combination of key and node, then picks the node with the highest hash value. This requires O(N) time per lookup since it must evaluate all nodes, but it has several attractive properties: there is no ring or data structure to maintain, it handles node additions and removals gracefully with minimal key movement, and the algorithm is trivially simple to implement. It is commonly used in systems where the node set is small (tens of nodes, not thousands) but changes frequently.

    **Maglev hashing**, developed by Google for their network load balancer, uses a precomputed lookup table that provides O(1) lookup time with excellent distribution. The table construction is more expensive (requires iterating through all nodes), but once built, every lookup is a single array access. Google uses Maglev to distribute packets across backend servers handling billions of requests per day. Its primary limitation is that the lookup table must be rebuilt whenever the node set changes, though this is fast enough in practice.

    | Algorithm | Lookup Time | Memory | Best For |
    |-----------|-------------|--------|----------|
    | Ring-based consistent hash | O(log N) | O(N * vnodes) | General distributed systems |
    | Jump consistent hash | O(ln N) | O(1) | Static/append-only server sets |
    | Rendezvous hashing | O(N) | O(1) | Small, dynamic node sets |
    | Maglev hashing | O(1) | O(lookup table) | High-throughput load balancing |

---

## Key Takeaways

1. Naive modulo hashing (`hash % N`) remaps nearly all keys when N changes, making it unsuitable for distributed systems that need to scale dynamically.

2. Consistent hashing uses a ring structure so that adding or removing a node moves only K/N keys on average -- the theoretical minimum disruption.

3. Virtual nodes (100-300 per physical server) are essential in practice to ensure even distribution of keys across physical machines. Without them, load imbalance of 2-3x is common.

4. The Dynamo paper (2007) popularized consistent hashing for production key-value stores, and its design directly influenced Cassandra, Riak, and Voldemort.

5. For specialized cases, consider alternatives: jump consistent hash for static server sets (O(1) memory), rendezvous hashing for small dynamic sets, or Maglev hashing for high-throughput load balancing.

6. When implementing replication on a consistent hash ring, walk clockwise and choose N successive distinct physical nodes to avoid placing replicas on virtual nodes that belong to the same machine.

---

## Related Topics

- [Database Sharding](../data/databases/sharding.md) -- partitioning strategies that often use consistent hashing underneath
- [Caching Strategies](../data/caching/strategies.md) -- distributed cache placement and eviction policies
- [Consensus](consensus.md) -- agreement protocols for distributed systems where consistent hashing determines data placement
