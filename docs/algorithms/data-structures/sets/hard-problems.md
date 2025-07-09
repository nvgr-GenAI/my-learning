# Sets: Hard Problems

## 🚀 Advanced Set Challenges

Master complex set operations and algorithms that combine multiple data structures to solve the most challenging problems.

=== "� Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Maximum XOR of Two Numbers | Trie + Bit Manipulation | Hard | O(n) | O(n) |
    | 2 | Word Break II | Dynamic Programming + Set | Hard | O(n³) | O(n) |
    | 3 | Word Search II | Trie + Backtracking | Hard | O(m×n×4^l) | O(k) |
    | 4 | Sliding Window Maximum | Monotonic Queue | Hard | O(n) | O(k) |
    | 5 | Distinct Subsequences | Dynamic Programming | Hard | O(m×n) | O(m×n) |
    | 6 | Minimum Window Substring | Sliding Window | Hard | O(n) | O(k) |
    | 7 | Longest Consecutive Sequence | Set Operations | Hard | O(n) | O(n) |
    | 8 | Insert Delete GetRandom O(1) - Duplicates allowed | Set + HashMap | Hard | O(1) | O(n) |
    | 9 | Substring with Concatenation of All Words | Sliding Window | Hard | O(n×m) | O(m) |
    | 10 | Word Ladder | BFS + Set | Hard | O(n×m²) | O(n×m) |
    | 11 | Alien Dictionary | Topological Sort | Hard | O(C) | O(1) |
    | 12 | N-Queens | Backtracking + Set | Hard | O(n!) | O(n) |
    | 13 | Serialize and Deserialize Binary Tree | Tree Serialization | Hard | O(n) | O(n) |
    | 14 | Redundant Connection II | Union-Find | Hard | O(n) | O(n) |
    | 15 | Stream of Characters | Trie | Hard | O(q×m) | O(w) |

=== "🎯 Interview Tips"

    **📝 Key Advanced Set Patterns:**
    
    - **Set + Other Data Structures**: Combining sets with tries, heaps, etc.
    - **Backtracking with Sets**: Using sets to track visited states
    - **Dynamic Sets**: Maintaining sets that change over time
    - **Bit Manipulation with Sets**: Using bits to represent set membership
    - **Custom Set Operations**: Implementing specialized set operations
    
    **⚡ Problem-Solving Strategies:**
    
    - Break down the problem into subproblems that can utilize set operations
    - Use sets to eliminate duplicates in recursive algorithms
    - Consider bitsets for memory-efficient set representation
    - Apply multiple passes: first gather information in a set, then process it
    - When dealing with character problems, use sets for quick membership tests
    
    **🚫 Common Pitfalls:**
    
    - Inefficient repeated conversions between sets and other data structures
    - Not considering the hash function performance for large sets
    - Overlooking the possibility of using bitsets for better memory usage
    - Using sets when ordered operations are required
    - Forgetting that set operations have their own complexity (union, intersection)

=== "📚 Study Plan"

    **Week 1: Set Foundations for Hard Problems (Problems 1-5)**
    - Apply sets in bit manipulation problems
    - Combine sets with tries for efficient lookups
    - Practice set-based dynamic programming
    
    **Week 2: Advanced Techniques (Problems 6-10)**
    - Master sliding window with set constraints
    - Apply set operations in graph problems
    - Practice word and string problems with sets
    
    **Week 3: Complex Applications (Problems 11-15)**
    - Solve system design problems using sets
    - Implement custom set operations
    - Combine sets with advanced algorithms

=== "Maximum XOR of Two Numbers"

    **Problem Statement:**
    Given an integer array `nums`, return the maximum result of `nums[i] XOR nums[j]`, where `0 ≤ i ≤ j < n`.

    **Example:**
    ```text
    Input: nums = [3,10,5,25,2,8]
    Output: 28
    Explanation: The maximum result is 5 XOR 25 = 28.
    ```

    **Solution:**
    ```python
    def findMaximumXOR(nums):
        """
        Bit manipulation with trie approach.
        
        Time: O(n) - we process each number once, with constant bit operations
        Space: O(n) - for storing the trie
        """
        # Edge case
        if not nums:
            return 0
            
        # Implement trie node for bit representation
        class TrieNode:
            def __init__(self):
                self.children = {}
                
        # Build the trie
        root = TrieNode()
        for num in nums:
            node = root
            # Process each bit from most significant to least
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                if bit not in node.children:
                    node.children[bit] = TrieNode()
                node = node.children[bit]
                
        # Find maximum XOR for each number
        max_xor = 0
        for num in nums:
            node = root
            current_xor = 0
            
            # Try to find complementary path for maximum XOR
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                # Optimal bit to maximize XOR is the opposite
                toggle_bit = 1 - bit
                
                # If complement exists, go that way
                if toggle_bit in node.children:
                    current_xor |= (1 << i)
                    node = node.children[toggle_bit]
                else:
                    # Otherwise, take what's available
                    node = node.children[bit]
                    
            # Update maximum XOR found
            max_xor = max(max_xor, current_xor)
            
        return max_xor
    ```

    **Key Insights:**
    - XOR properties are leveraged to maximize the result
    - Trie data structure efficiently finds the best complement for each number
    - Processing bits from most significant to least significant maximizes value
    - For each bit, we try to go the opposite path in trie to maximize XOR
    - Time complexity is O(n) because we only examine each number once (32 bits max)
            else:
                node = node.children[bit]
        
        max_xor = max(max_xor, current_xor)
    
    return max_xor

# Alternative approach using prefix
def find_maximum_xor_prefix(nums):
    """
    Find maximum XOR using prefix approach
    Time: O(32n), Space: O(32n)
    """
    max_xor = 0
    mask = 0
    
    for i in range(31, -1, -1):
        mask |= (1 << i)
        prefixes = {num & mask for num in nums}
        
        temp = max_xor | (1 << i)
        
        # Check if we can achieve this max_xor
        for prefix in prefixes:
            if temp ^ prefix in prefixes:
                max_xor = temp
                break
    
    return max_xor

# Test
nums = [3, 10, 5, 25, 2, 8]
print(find_maximum_xor(nums))  # 28 (25 ^ 3)
print(find_maximum_xor_prefix(nums))  # 28
```

### Problem 2: Design Twitter

**Problem**: Design a simplified Twitter with core functionality.

```python
import heapq
from collections import defaultdict, deque

class Twitter:
    def __init__(self):
        self.tweets = defaultdict(deque)  # userId -> deque of (tweetId, timestamp)
        self.following = defaultdict(set)  # userId -> set of followeeIds
        self.timestamp = 0
    
    def post_tweet(self, user_id, tweet_id):
        """Post a new tweet"""
        self.tweets[user_id].appendleft((tweet_id, self.timestamp))
        self.timestamp += 1
        
        # Keep only last 10 tweets per user for efficiency
        if len(self.tweets[user_id]) > 10:
            self.tweets[user_id].pop()
    
    def get_news_feed(self, user_id):
        """Get 10 most recent tweets from user and followees"""
        # Use min heap to get top 10 most recent tweets
        heap = []
        
        # Add user's own tweets
        for tweet_id, timestamp in self.tweets[user_id]:
            heapq.heappush(heap, (-timestamp, tweet_id))
        
        # Add followees' tweets
        for followee_id in self.following[user_id]:
            for tweet_id, timestamp in self.tweets[followee_id]:
                heapq.heappush(heap, (-timestamp, tweet_id))
        
        # Get top 10 most recent
        result = []
        for _ in range(min(10, len(heap))):
            if heap:
                _, tweet_id = heapq.heappop(heap)
                result.append(tweet_id)
        
        return result
    
    def follow(self, follower_id, followee_id):
        """Follow a user"""
        if follower_id != followee_id:
            self.following[follower_id].add(followee_id)
    
    def unfollow(self, follower_id, followee_id):
        """Unfollow a user"""
        self.following[follower_id].discard(followee_id)

# Test
twitter = Twitter()
twitter.post_tweet(1, 5)
twitter.get_news_feed(1)    # [5]
twitter.follow(1, 2)
twitter.post_tweet(2, 6)
twitter.get_news_feed(1)    # [6, 5]
twitter.unfollow(1, 2)
twitter.get_news_feed(1)    # [5]
```

### Problem 3: First Missing Positive

**Problem**: Find the first missing positive integer in O(n) time and O(1) space.

```python
def first_missing_positive(nums):
    """
    Find first missing positive using array as hash set
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    
    # Step 1: Replace non-positive numbers with n+1
    for i in range(n):
        if nums[i] <= 0:
            nums[i] = n + 1
    
    # Step 2: Use indices as hash keys
    for i in range(n):
        num = abs(nums[i])
        if num <= n:
            nums[num - 1] = -abs(nums[num - 1])
    
    # Step 3: Find first positive index
    for i in range(n):
        if nums[i] > 0:
            return i + 1
    
    return n + 1

def first_missing_positive_swap(nums):
    """
    Alternative approach using swapping
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    
    # Place each number at its correct position
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    
    # Find first missing positive
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    
    return n + 1

# Test
nums1 = [1, 2, 0]
print(first_missing_positive(nums1))  # 3

nums2 = [3, 4, -1, 1]
print(first_missing_positive_swap(nums2))  # 2

nums3 = [7, 8, 9, 11, 12]
print(first_missing_positive(nums3))  # 1
```

## 🚀 Advanced Techniques

### Bloom Filters for Approximate Membership

```python
import hashlib
import math

class BloomFilter:
    def __init__(self, capacity, error_rate):
        """
        Probabilistic data structure for membership testing
        """
        self.capacity = capacity
        self.error_rate = error_rate
        
        # Calculate optimal parameters
        self.bit_array_size = int(-(capacity * math.log(error_rate)) / (math.log(2) ** 2))
        self.hash_count = int((self.bit_array_size / capacity) * math.log(2))
        
        self.bit_array = [0] * self.bit_array_size
    
    def _hash(self, item, seed):
        """Generate hash with different seeds"""
        hasher = hashlib.md5()
        hasher.update(f"{item}{seed}".encode('utf-8'))
        return int(hasher.hexdigest(), 16) % self.bit_array_size
    
    def add(self, item):
        """Add item to filter"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = 1
    
    def might_contain(self, item):
        """Check if item might be in set (no false negatives)"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False
        return True
    
    def false_positive_rate(self):
        """Calculate current false positive rate"""
        bits_set = sum(self.bit_array)
        return (1 - math.exp(-self.hash_count * bits_set / self.bit_array_size)) ** self.hash_count

# Usage example
bf = BloomFilter(1000, 0.01)
bf.add("user123")
bf.add("user456")

print(bf.might_contain("user123"))  # True
print(bf.might_contain("user999"))  # False (probably)
print(f"False positive rate: {bf.false_positive_rate():.4f}")
```

### Union-Find for Set Operations

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.components = n
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        self.size[px] += self.size[py]
        
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x, y):
        """Check if two elements are in same set"""
        return self.find(x) == self.find(y)
    
    def component_size(self, x):
        """Get size of component containing x"""
        return self.size[self.find(x)]

def find_similar_groups(strs):
    """
    Group similar strings using Union-Find
    Time: O(n² × m), Space: O(n)
    """
    def are_similar(s1, s2):
        """Check if strings are similar (differ by at most 2 positions)"""
        diff_count = sum(c1 != c2 for c1, c2 in zip(s1, s2))
        return diff_count <= 2
    
    n = len(strs)
    uf = UnionFind(n)
    
    # Union similar strings
    for i in range(n):
        for j in range(i + 1, n):
            if are_similar(strs[i], strs[j]):
                uf.union(i, j)
    
    return uf.components

# Test
strs = ["tars", "rats", "arts", "star"]
print(find_similar_groups(strs))  # 2 groups
```

### Advanced XOR Techniques

```python
def maximum_xor_with_k_changes(nums, k):
    """
    Maximum XOR after at most k bit flips
    Time: O(n × 32), Space: O(1)
    """
    max_xor = 0
    
    # Try each possible result bit by bit
    for bit in range(31, -1, -1):
        # Count how many numbers have this bit set
        count_set = sum((num >> bit) & 1 for num in nums)
        count_unset = len(nums) - count_set
        
        # Determine if we should set this bit in result
        if k >= count_unset:
            # We can flip all unset bits to set
            max_xor |= (1 << bit)
            k -= count_unset
            # Update nums to reflect the flips
            nums = [num | (1 << bit) if not ((num >> bit) & 1) else num for num in nums]
        else:
            # We can flip some unset bits
            flips_needed = min(k, count_unset)
            if flips_needed > count_set:
                max_xor |= (1 << bit)
            k -= flips_needed
    
    return max_xor

def xor_queries_subarray(arr, queries):
    """
    Answer XOR queries on subarrays efficiently
    Time: O(n + q), Space: O(n)
    """
    # Build prefix XOR array
    prefix_xor = [0]
    for num in arr:
        prefix_xor.append(prefix_xor[-1] ^ num)
    
    result = []
    for left, right in queries:
        # XOR of subarray [left, right] = prefix_xor[right+1] ^ prefix_xor[left]
        result.append(prefix_xor[right + 1] ^ prefix_xor[left])
    
    return result

# Test
nums = [5, 6, 7, 8]
k = 3
print(maximum_xor_with_k_changes(nums, k))

arr = [1, 3, 4, 8]
queries = [[0, 1], [1, 2], [0, 3], [3, 3]]
print(xor_queries_subarray(arr, queries))  # [2, 7, 14, 8]
```

## 🎯 Problem Patterns

### Pattern 1: Set as State Space

- **Use Case**: When you need to track visited states or configurations
- **Examples**: Word ladder, minimum genetic mutation
- **Key**: Represent states as hashable objects

### Pattern 2: Set Operations for Relationships

- **Use Case**: Finding common elements, differences, unions
- **Examples**: Friend recommendations, data deduplication
- **Key**: Leverage built-in set operations efficiently

### Pattern 3: Bit Manipulation for Small Universes

- **Use Case**: When dealing with small fixed-size sets
- **Examples**: Subset generation, state compression in DP
- **Key**: Use bitwise operations for speed

### Pattern 4: Probabilistic Sets

- **Use Case**: When approximate membership is acceptable
- **Examples**: Web crawling, caching systems
- **Key**: Trade accuracy for space/time efficiency

## 📊 Complexity Analysis

| **Problem Type** | **Time Complexity** | **Space Complexity** | **Notes** |
|------------------|-------------------|---------------------|-----------|
| **Maximum XOR** | O(32n) | O(32n) | Using Trie |
| **Design Twitter** | O(k log k) | O(n × k) | k = feed size |
| **First Missing Positive** | O(n) | O(1) | In-place hash |
| **Union-Find** | O(α(n)) | O(n) | Amortized |
| **Bloom Filter** | O(k) | O(m) | k = hash functions |

## 🏆 Interview Tips

1. **Clarify constraints**: Universe size affects approach choice
2. **Consider trade-offs**: Space vs time vs accuracy
3. **Think about edge cases**: Empty sets, single elements
4. **Optimize for the common case**: What operations are most frequent?
5. **Use appropriate data structures**: Hash set vs tree set vs bit set

## 🔗 Related Topics

- **Graph Algorithms**: Connected components, spanning trees
- **String Algorithms**: Pattern matching, text processing
- **Dynamic Programming**: State compression, memoization
- **Probability**: Bloom filters, hash functions
- **System Design**: Distributed sets, consistent hashing
