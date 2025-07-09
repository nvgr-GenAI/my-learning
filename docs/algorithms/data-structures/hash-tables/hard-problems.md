# Hash Tables: Hard Problems

## 🚀 Advanced Hash Table Challenges

Master complex hash table applications that combine multiple data structures and algorithmic patterns for solving the most challenging problems.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | LFU Cache | Custom Data Structure | Hard | O(1) | O(n) |
    | 2 | Minimum Window Substring | Sliding Window | Hard | O(n) | O(k) |
    | 3 | Substring with Concatenation of All Words | Sliding Window | Hard | O(n×m) | O(m) |
    | 4 | Word Ladder | BFS + Hash Set | Hard | O(n×m²) | O(n×m) |
    | 5 | All O`one Data Structure | Custom Data Structure | Hard | O(1) | O(n) |
    | 6 | Longest Duplicate Substring | Binary Search + Rabin-Karp | Hard | O(n log n) | O(n) |
    | 7 | Palindrome Pairs | Trie/HashMap | Hard | O(n×k²) | O(n×k) |
    | 8 | Number of Valid Words for Each Puzzle | Bitmask + HashMap | Hard | O(n×k + m×2^p) | O(n) |
    | 9 | Maximum Frequency Stack | Multi-level HashMap | Hard | O(1) | O(n) |
    | 10 | First Missing Positive | Array Hashing | Hard | O(n) | O(1) |
    | 11 | Subarrays with K Different Integers | Sliding Window | Hard | O(n) | O(k) |
    | 12 | Design Search Autocomplete System | Trie + HashMap | Hard | O(p+q) | O(n) |
    | 13 | Design In-Memory File System | Trie + HashMap | Hard | O(l) | O(n) |
    | 14 | Random Pick with Blacklist | HashMap + Math | Hard | O(b) | O(b) |
    | 15 | Longest Consecutive Sequence | Set Operations | Hard | O(n) | O(n) |

=== "🎯 Interview Tips"

    **📝 Key Advanced Hash Table Patterns:**
    
    - **Multi-layer Hash Maps**: Nesting hash maps for complex relationships
    - **Hash + Heap Combinations**: For prioritized access with fast lookups
    - **Custom Hash Functions**: For specialized equality comparisons
    - **Distributed Hash Tables**: For system design questions
    - **Probabilistic Data Structures**: Bloom filters, Count-Min sketch
    
    **⚡ Problem-Solving Strategies:**
    
    - Design custom hash keys for complex objects or patterns
    - Combine hash tables with other data structures (trees, heaps, graphs)
    - Use memoization with hash tables for dynamic programming optimization
    - Apply sliding window with counters for substring problems
    - Leverage bit manipulation with hash tables for space optimization
    
    **🚫 Common Pitfalls:**
    
    - Creating overly complex hash functions that slow down operations
    - Excessive memory use in multi-map solutions
    - Not accounting for collisions in custom implementations
    - Using mutable objects as hash keys
    - Inefficient handling of resizing operations in custom implementations

=== "📚 Study Plan"

    **Week 1: Hash Table Design Challenges (Problems 1-5)**
    - Focus on custom data structure design
    - Master frequency tracking and efficient updates
    - Practice cache implementation patterns
    
    **Week 2: String & Substring Problems (Problems 6-10)**
    - Learn advanced sliding window techniques
    - Implement string matching with hash tables
    - Practice substring and subsequence problems
    
    **Week 3: System Design with Hash Tables (Problems 11-15)**
    - Build complex system components
    - Optimize for memory and access patterns
    - Combine with other data structures for efficient solutions

=== "LFU Cache"

    **Problem Statement:**
    Design and implement a data structure for a Least Frequently Used (LFU) cache.
    
    Implement the `LFUCache` class:
    - `LFUCache(int capacity)` - Initialize the object with the capacity of the data structure.
    - `int get(int key)` - Return the value of the key if it exists, otherwise return -1.
    - `void put(int key, int value)` - Update the value of the key if present, or insert the key if not present. When the cache reaches its capacity, invalidate the least frequently used item before inserting a new item. If there is a tie (i.e., two or more keys with the same frequency), the least recently used key would be invalidated.

    **Example:**
    ```text
    Input:
    ["LFUCache", "put", "put", "get", "put", "get", "get", "put", "get", "get", "get"]
    [[2], [1, 1], [2, 2], [1], [3, 3], [2], [3], [4, 4], [1], [3], [4]]
    Output:
    [null, null, null, 1, null, -1, 3, null, -1, 3, 4]
    ```

    **Solution:**
    ```python
    class LFUCache:
        """
        Implementation using hash maps and frequency counting.
        
        Time: O(1) for all operations
        Space: O(capacity) for storing key-value pairs and frequency info
        """
        def __init__(self, capacity: int):
            self.capacity = capacity
            self.min_freq = 0
            self.key_to_val = {}  # Maps key to value
            self.key_to_freq = {}  # Maps key to frequency
            self.freq_to_keys = defaultdict(OrderedDict)  # Maps frequency to ordered keys
            
        def get(self, key: int) -> int:
            if key not in self.key_to_val:
                return -1
                
            # Get current frequency and value
            freq = self.key_to_freq[key]
            val = self.key_to_val[key]
            
            # Remove key from current frequency list
            self.freq_to_keys[freq].pop(key)
            
            # Update min_freq if needed
            if len(self.freq_to_keys[self.min_freq]) == 0 and self.min_freq == freq:
                self.min_freq += 1
                
            # Increment frequency and update mappings
            self.key_to_freq[key] = freq + 1
            self.freq_to_keys[freq + 1][key] = None  # OrderedDict as queue, value doesn't matter
            
            return val
            
        def put(self, key: int, value: int) -> None:
            if self.capacity == 0:
                return
                
            # If key exists, update value and increment frequency
            if key in self.key_to_val:
                self.key_to_val[key] = value
                self.get(key)  # Use get to increment frequency
                return
                
            # If at capacity, remove least frequent item
            if len(self.key_to_val) >= self.capacity:
                # Get least frequently used key
                lfu_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
                del self.key_to_val[lfu_key]
                del self.key_to_freq[lfu_key]
                
            # Add new key-value pair
            self.key_to_val[key] = value
            self.key_to_freq[key] = 1
            self.freq_to_keys[1][key] = None
            self.min_freq = 1  # New item is least frequent
    ```

    **Key Insights:**
    - Use three hash maps to maintain all required information:
      1. key → value mapping
      2. key → frequency mapping
      3. frequency → ordered set of keys mapping
    - OrderedDict preserves insertion order for LRU tie-breaking
    - Track minimum frequency for O(1) eviction of least frequent item
    - Use get operation logic to update frequency in the put method
    - Handle edge cases like capacity of zero or existing keys
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> node mapping
        
        # Create dummy head and tail nodes
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove existing node from linked list"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node):
        """Move node to head (mark as recently used)"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Remove last node (least recently used)"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key):
        node = self.cache.get(key)
        
        if not node:
            return -1
        
        # Move to head (mark as recently used)
        self._move_to_head(node)
        return node.value
    
    def put(self, key, value):
        node = self.cache.get(key)
        
        if not node:
            # New key-value pair
            new_node = self.Node(key, value)
            
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            # Add new node
            self.cache[key] = new_node
            self._add_node(new_node)
        else:
            # Update existing key
            node.value = value
            self._move_to_head(node)

# Alternative implementation using OrderedDict
from collections import OrderedDict

class LRUCacheOrderedDict:
    """
    Simpler implementation using OrderedDict
    (Less educational but more concise)
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key):
        if key not in self.cache:
            return -1
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used (first item)
            self.cache.popitem(last=False)
        
        self.cache[key] = value

# Test
lru = LRUCache(2)
lru.put(1, 1)
lru.put(2, 2)
print(lru.get(1))    # 1
lru.put(3, 3)        # evicts key 2
print(lru.get(2))    # -1
```

### Advanced LRU Variations

```python
class LFUCache:
    """
    Least Frequently Used Cache
    More complex: track both frequency and recency
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.key_to_val = {}      # key -> value
        self.key_to_freq = {}     # key -> frequency
        self.freq_to_keys = {}    # frequency -> set of keys
        self.min_freq = 0
    
    def _update_freq(self, key):
        """Update frequency of key"""
        freq = self.key_to_freq[key]
        
        # Remove from old frequency bucket
        self.freq_to_keys[freq].remove(key)
        if freq == self.min_freq and not self.freq_to_keys[freq]:
            self.min_freq += 1
        
        # Add to new frequency bucket
        new_freq = freq + 1
        self.key_to_freq[key] = new_freq
        if new_freq not in self.freq_to_keys:
            self.freq_to_keys[new_freq] = set()
        self.freq_to_keys[new_freq].add(key)
    
    def get(self, key):
        if key not in self.key_to_val:
            return -1
        
        self._update_freq(key)
        return self.key_to_val[key]
    
    def put(self, key, value):
        if self.capacity <= 0:
            return
        
        if key in self.key_to_val:
            # Update existing
            self.key_to_val[key] = value
            self._update_freq(key)
            return
        
        # Add new key
        if len(self.key_to_val) >= self.capacity:
            # Remove least frequent key
            remove_key = self.freq_to_keys[self.min_freq].pop()
            del self.key_to_val[remove_key]
            del self.key_to_freq[remove_key]
        
        # Insert new key
        self.key_to_val[key] = value
        self.key_to_freq[key] = 1
        if 1 not in self.freq_to_keys:
            self.freq_to_keys[1] = set()
        self.freq_to_keys[1].add(key)
        self.min_freq = 1
```

---

## Problem 2: Design Twitter

**Difficulty**: 🔴 Hard  
**Pattern**: Multiple Hash Maps + Heap  
**Time**: O(log N) for posting/following, O(K log N) for timeline  
**Space**: O(N + M) where N=users, M=tweets

### Problem Statement

Design a simplified Twitter with these operations:
- `postTweet(userId, tweetId)` - User posts a tweet
- `getNewsFeed(userId)` - Get 10 most recent tweets from user and followees
- `follow(followerId, followeeId)` - User follows another
- `unfollow(followerId, followeeId)` - User unfollows another

### Solution

```python
import heapq
from collections import defaultdict

class Twitter:
    """
    Twitter system design using multiple hash maps
    """
    
    def __init__(self):
        self.timestamp = 0
        self.tweets = defaultdict(list)  # userId -> [(timestamp, tweetId), ...]
        self.following = defaultdict(set)  # userId -> set of followees
    
    def postTweet(self, userId, tweetId):
        """Post a tweet"""
        self.tweets[userId].append((self.timestamp, tweetId))
        self.timestamp += 1
    
    def getNewsFeed(self, userId):
        """
        Get 10 most recent tweets from user and followees
        Use min-heap to efficiently merge sorted lists
        """
        # Get all relevant users (self + followees)
        users = set(self.following[userId])
        users.add(userId)
        
        # Use heap to merge recent tweets
        heap = []
        
        for user in users:
            if user in self.tweets and self.tweets[user]:
                # Add most recent tweet from each user
                timestamp, tweetId = self.tweets[user][-1]
                # Use negative timestamp for max-heap behavior
                heapq.heappush(heap, (-timestamp, tweetId, user, len(self.tweets[user]) - 1))
        
        result = []
        while heap and len(result) < 10:
            neg_timestamp, tweetId, user, index = heapq.heappop(heap)
            result.append(tweetId)
            
            # Add next tweet from same user if available
            if index > 0:
                timestamp, tweetId = self.tweets[user][index - 1]
                heapq.heappush(heap, (-timestamp, tweetId, user, index - 1))
        
        return result
    
    def follow(self, followerId, followeeId):
        """Follow another user"""
        if followerId != followeeId:
            self.following[followerId].add(followeeId)
    
    def unfollow(self, followerId, followeeId):
        """Unfollow another user"""
        self.following[followerId].discard(followeeId)

# Alternative implementation with better data structure
class TwitterOptimized:
    """
    Optimized version using deque for tweets and better heap management
    """
    
    class Tweet:
        def __init__(self, tweetId, timestamp):
            self.tweetId = tweetId
            self.timestamp = timestamp
            self.next = None
    
    class User:
        def __init__(self, userId):
            self.userId = userId
            self.followed = set()
            self.tweetHead = None  # Most recent tweet
            self.follow(userId)  # Follow self
        
        def follow(self, userId):
            self.followed.add(userId)
        
        def unfollow(self, userId):
            self.followed.discard(userId)
        
        def post(self, tweetId, timestamp):
            tweet = TwitterOptimized.Tweet(tweetId, timestamp)
            tweet.next = self.tweetHead
            self.tweetHead = tweet
    
    def __init__(self):
        self.timestamp = 0
        self.users = {}
    
    def _getUser(self, userId):
        if userId not in self.users:
            self.users[userId] = self.User(userId)
        return self.users[userId]
    
    def postTweet(self, userId, tweetId):
        user = self._getUser(userId)
        user.post(tweetId, self.timestamp)
        self.timestamp += 1
    
    def getNewsFeed(self, userId):
        user = self._getUser(userId)
        
        # Use heap to merge tweet streams
        heap = []
        for followeeId in user.followed:
            followee = self._getUser(followeeId)
            if followee.tweetHead:
                heapq.heappush(heap, (-followee.tweetHead.timestamp, 
                                     followee.tweetHead))
        
        result = []
        while heap and len(result) < 10:
            _, tweet = heapq.heappop(heap)
            result.append(tweet.tweetId)
            
            if tweet.next:
                heapq.heappush(heap, (-tweet.next.timestamp, tweet.next))
        
        return result
    
    def follow(self, followerId, followeeId):
        follower = self._getUser(followerId)
        follower.follow(followeeId)
    
    def unfollow(self, followerId, followeeId):
        follower = self._getUser(followerId)
        follower.unfollow(followeeId)

# Test
twitter = Twitter()
twitter.postTweet(1, 5)
print(twitter.getNewsFeed(1))  # [5]
twitter.follow(1, 2)
twitter.postTweet(2, 6)
print(twitter.getNewsFeed(1))  # [6, 5]
```

---

## Problem 3: Alien Dictionary

**Difficulty**: 🔴 Hard  
**Pattern**: Topological Sort + Hash Map  
**Time**: O(C) where C is total content of words, **Space**: O(1) - limited alphabet

### Problem Statement

Given a list of words from an alien language's dictionary (sorted lexicographically), find the order of characters in the alien alphabet.

```python
Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"
```

### Solution

```python
from collections import defaultdict, deque

def alien_order(words):
    """
    Extract character order using topological sort
    
    Steps:
    1. Build graph of character dependencies
    2. Calculate in-degrees for each character
    3. Use topological sort (Kahn's algorithm)
    """
    
    # Initialize graph and in-degree count
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    
    # Initialize all characters with in-degree 0
    for word in words:
        for char in word:
            in_degree[char] = 0
    
    # Build graph by comparing adjacent words
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        
        # Find first different character
        min_len = min(len(word1), len(word2))
        found_diff = False
        
        for j in range(min_len):
            if word1[j] != word2[j]:
                # word1[j] comes before word2[j]
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    in_degree[word2[j]] += 1
                found_diff = True
                break
        
        # Invalid case: word1 is prefix of word2 but comes after
        if not found_diff and len(word1) > len(word2):
            return ""
    
    # Topological sort using Kahn's algorithm
    queue = deque([char for char in in_degree if in_degree[char] == 0])
    result = []
    
    while queue:
        char = queue.popleft()
        result.append(char)
        
        # Reduce in-degree of neighbors
        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if all characters are processed (no cycle)
    if len(result) != len(in_degree):
        return ""  # Cycle detected
    
    return ''.join(result)

def alien_order_dfs(words):
    """
    Alternative DFS-based approach
    """
    graph = defaultdict(set)
    all_chars = set()
    
    # Build graph
    for word in words:
        all_chars.update(word)
    
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))
        
        found_diff = False
        for j in range(min_len):
            if word1[j] != word2[j]:
                graph[word1[j]].add(word2[j])
                found_diff = True
                break
        
        if not found_diff and len(word1) > len(word2):
            return ""
    
    # DFS with cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {char: WHITE for char in all_chars}
    result = []
    
    def dfs(node):
        if color[node] == GRAY:  # Cycle detected
            return False
        if color[node] == BLACK:  # Already processed
            return True
        
        color[node] = GRAY
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        
        color[node] = BLACK
        result.append(node)
        return True
    
    # Process all characters
    for char in all_chars:
        if color[char] == WHITE:
            if not dfs(char):
                return ""
    
    return ''.join(reversed(result))

# Test
words = ["wrt", "wrf", "er", "ett", "rftt"]
print(alien_order(words))  # "wertf"
```

---

## Problem 4: Substring with Concatenation of All Words

**Difficulty**: 🔴 Hard  
**Pattern**: Sliding Window + Hash Map  
**Time**: O(N × M × K), **Space**: O(M × K)

### Problem Statement

Given string `s` and array `words` of same-length strings, find all starting indices where `s` contains concatenation of all words exactly once.

```python
Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
Explanation: "barfoo" at index 0, "foobar" at index 9
```

### Solution

```python
def find_substring(s, words):
    """
    Sliding window approach with hash map frequency matching
    """
    if not s or not words:
        return []
    
    word_len = len(words[0])
    total_len = len(words) * word_len
    word_count = {}
    
    # Count frequency of each word
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    result = []
    
    # Try each possible starting position within first word_len positions
    for i in range(word_len):
        left = i
        right = i
        current_count = {}
        matched_words = 0
        
        while right + word_len <= len(s):
            # Get word at right pointer
            word = s[right:right + word_len]
            right += word_len
            
            if word in word_count:
                current_count[word] = current_count.get(word, 0) + 1
                matched_words += 1
                
                # If word frequency exceeds required, slide left window
                while current_count[word] > word_count[word]:
                    left_word = s[left:left + word_len]
                    current_count[left_word] -= 1
                    matched_words -= 1
                    left += word_len
                
                # Check if we have a valid window
                if matched_words == len(words):
                    result.append(left)
            else:
                # Reset window if word not in dictionary
                current_count.clear()
                matched_words = 0
                left = right
    
    return result

def find_substring_brute_force(s, words):
    """
    Brute force approach for comparison
    """
    if not s or not words:
        return []
    
    word_len = len(words[0])
    total_len = len(words) * word_len
    word_count = {}
    
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    result = []
    
    for i in range(len(s) - total_len + 1):
        seen = {}
        j = 0
        
        while j < len(words):
            word = s[i + j * word_len:i + (j + 1) * word_len]
            
            if word not in word_count:
                break
            
            seen[word] = seen.get(word, 0) + 1
            if seen[word] > word_count[word]:
                break
            
            j += 1
        
        if j == len(words):
            result.append(i)
    
    return result

# Test
s = "barfoothefoobarman"
words = ["foo", "bar"]
print(find_substring(s, words))  # [0, 9]
```

---

## Problem 5: Insert Delete GetRandom O(1)

**Difficulty**: 🔴 Hard  
**Pattern**: Hash Map + Dynamic Array  
**Time**: O(1) for all operations, **Space**: O(N)

### Problem Statement

Implement `RandomizedSet` class:
- `insert(val)` - Insert value, return true if not present
- `remove(val)` - Remove value, return true if present  
- `getRandom()` - Return random element with equal probability

All operations must be O(1) average time complexity.

### Solution

```python
import random

class RandomizedSet:
    """
    Combine hash map and dynamic array for O(1) operations
    
    Key insight:
    - Array enables O(1) random access
    - Hash map enables O(1) insert/remove by tracking indices
    - Swap-with-last technique for O(1) removal
    """
    
    def __init__(self):
        self.values = []  # Store actual values
        self.indices = {}  # value -> index mapping
    
    def insert(self, val):
        """Insert value if not present"""
        if val in self.indices:
            return False
        
        # Add to end of array and update index map
        self.indices[val] = len(self.values)
        self.values.append(val)
        return True
    
    def remove(self, val):
        """Remove value if present"""
        if val not in self.indices:
            return False
        
        # Swap with last element to avoid shifting
        index = self.indices[val]
        last_val = self.values[-1]
        
        # Move last element to position of removed element
        self.values[index] = last_val
        self.indices[last_val] = index
        
        # Remove last element and its index mapping
        self.values.pop()
        del self.indices[val]
        
        return True
    
    def getRandom(self):
        """Return random element"""
        return random.choice(self.values)

# Enhanced version with duplicates allowed
class RandomizedCollection:
    """
    Allow duplicate values
    """
    
    def __init__(self):
        self.values = []
        self.indices = defaultdict(set)  # value -> set of indices
    
    def insert(self, val):
        """Insert value (duplicates allowed)"""
        self.indices[val].add(len(self.values))
        self.values.append(val)
        return len(self.indices[val]) == 1  # True if first occurrence
    
    def remove(self, val):
        """Remove one occurrence of value"""
        if not self.indices[val]:
            return False
        
        # Get an arbitrary index to remove
        remove_index = self.indices[val].pop()
        last_val = self.values[-1]
        
        # Swap with last element
        self.values[remove_index] = last_val
        
        # Update indices for the moved element
        self.indices[last_val].add(remove_index)
        self.indices[last_val].discard(len(self.values) - 1)
        
        self.values.pop()
        return True
    
    def getRandom(self):
        """Return random element"""
        return random.choice(self.values)

# Test
randomized_set = RandomizedSet()
print(randomized_set.insert(1))    # True
print(randomized_set.remove(2))    # False
print(randomized_set.insert(2))    # True
print(randomized_set.getRandom())  # 1 or 2
print(randomized_set.remove(1))    # True
print(randomized_set.insert(2))    # False
print(randomized_set.getRandom())  # 2
```

---

## 🎯 Advanced Concepts Summary

### System Design Patterns

1. **Multi-Data Structure**: Combine hash maps with other structures (lists, heaps, trees)
2. **Cache Design**: LRU, LFU with O(1) operations using clever data structure combinations
3. **Real-time Systems**: Twitter-like feeds using heaps for efficient merging
4. **Graph Problems**: Topological sort with hash-based adjacency representation

### Advanced Techniques

- **Swap-with-last**: O(1) removal from arrays while maintaining hash map consistency
- **Multiple Hash Maps**: Track different aspects (frequency, position, relationships)
- **Heap + Hash Map**: Efficient priority operations with fast lookups
- **Sliding Window + Hash Map**: Complex pattern matching with frequency constraints

### Performance Optimization

- Choose right collision resolution strategy
- Consider memory vs. time tradeoffs
- Use appropriate hash functions for data distribution
- Implement proper resizing strategies

---

## 🚀 Next Steps

Congratulations on mastering hash tables! These problems represent some of the most challenging applications. Consider exploring:

1. **Consistent Hashing** for distributed systems
2. **Bloom Filters** for probabilistic membership testing
3. **Skip Lists** as hash table alternatives
4. **Concurrent Hash Maps** for multi-threaded applications

Hash tables are fundamental to many advanced algorithms and system designs. The patterns you've learned here will serve you well in technical interviews and real-world problem solving!

---

*"Hash tables are like Swiss Army knives of data structures - versatile, efficient, and essential for any programmer's toolkit."*
