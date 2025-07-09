# Hash Tables: Medium Problems

## 🚀 Intermediate Hash Table Challenges

Master intermediate hash table techniques for solving complex problems involving frequency counting, custom key design, and advanced lookup patterns.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Group Anagrams | Grouping with Hash Keys | Medium | O(n×m log m) | O(n×m) |
    | 2 | LRU Cache | Custom Data Structure | Medium | O(1) | O(n) |
    | 3 | Subarray Sum Equals K | Prefix Sum | Medium | O(n) | O(n) |
    | 4 | Longest Substring Without Repeating Characters | Sliding Window | Medium | O(n) | O(min(n,m)) |
    | 5 | Top K Frequent Elements | Bucket Sort | Medium | O(n) | O(n) |
    | 6 | Design HashMap | Direct Addressing | Medium | O(1) | O(n) |
    | 7 | Valid Sudoku | Matrix Validation | Medium | O(1) | O(1) |
    | 8 | Copy List with Random Pointer | HashMap | Medium | O(n) | O(n) |
    | 9 | Longest Consecutive Sequence | Set Operations | Medium | O(n) | O(n) |
    | 10 | Find All Anagrams in a String | Sliding Window | Medium | O(n) | O(1) |
    | 11 | 4Sum II | Hash Counting | Medium | O(n²) | O(n²) |
    | 12 | Continuous Subarray Sum | Prefix Sum with Modulo | Medium | O(n) | O(k) |
    | 13 | Design Twitter | Custom Data Structure | Medium | O(n log k) | O(n) |
    | 14 | Time Based Key-Value Store | Binary Search with HashMap | Medium | O(log n) | O(n) |
    | 15 | Longest Substring with At Most K Distinct Characters | Sliding Window | Medium | O(n) | O(k) |

=== "🎯 Interview Tips"

    **📝 Key Hash Table Patterns:**
    
    - **Frequency Counting**: Counting elements, characters, or patterns
    - **Two-Sum Pattern**: Use complement lookup for efficient pair finding
    - **Custom Hash Keys**: Creating unique hash keys for complex objects
    - **Prefix Sum**: Track cumulative sums for subarray calculations
    - **Sliding Window**: Combine with hash table for efficient window tracking
    
    **⚡ Problem-Solving Strategies:**
    
    - Identify what makes a good hash key for the problem
    - Use `defaultdict` or `Counter` for frequency problems
    - Consider using multiple hash maps for complex relationships
    - For optimization problems, store computed results in a hash table
    - Combine hash tables with other data structures like heaps or arrays
    
    **🚫 Common Pitfalls:**
    
    - Not handling collisions properly in custom implementations
    - Forgetting to check if a key exists before accessing
    - Creating unnecessarily complex hash functions
    - Not considering string vs object equality for custom objects
    - Using mutable objects as hash keys

=== "📚 Study Plan"

    **Week 1: Basic Hash Table Applications (Problems 1-5)**
    - Master grouping and counting techniques
    - Practice frequency-based problems
    - Learn custom hash key design
    
    **Week 2: Advanced Implementations (Problems 6-10)**
    - Implement custom hash structures
    - Combine with other data structures
    - Practice sliding window with hash tables
    
    **Week 3: Complex Applications (Problems 11-15)**
    - Multi-step hash table approaches
    - Time-based and dynamic hash tables
    - Performance optimization techniques

=== "Group Anagrams"

    **Problem Statement:**
    Given an array of strings `strs`, group the anagrams together. An anagram is a word formed by rearranging the letters of another, using all the original letters exactly once.

    **Example:**
    ```text
    Input: strs = ["eat","tea","tan","ate","nat","bat"]
    Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
    ```

    **Solution:**
    ```python
    def groupAnagrams(strs):
        """
        Use sorted string as hash key.
        
        Time: O(n×m log m) where n=number of strings, m=max string length
        Space: O(n×m) for the dictionary and result
        """
        from collections import defaultdict
        
        groups = defaultdict(list)
        
        for s in strs:
            # Sorted string as unique key for anagrams
            key = ''.join(sorted(s))
            groups[key].append(s)
        
        return list(groups.values())
    ```

    **Key Insights:**
    - The sorted version of an anagram creates a unique identifier
    - Using a hash map allows O(1) lookup for each group
    - defaultdict avoids key existence checking
    - For very large inputs, can use character count as key instead of sorting

=== "LRU Cache"

    **Problem Statement:**
    Design and implement a data structure for Least Recently Used (LRU) cache. It should support get and put operations with O(1) time complexity.
    
    - get(key): Return the value of the key if it exists, otherwise return -1.
    - put(key, value): Update or insert the value if the key exists. When the cache reaches capacity, invalidate the least recently used key.

    **Example:**
    ```text
    LRUCache cache = new LRUCache(2);  // capacity = 2
    cache.put(1, 1);
    cache.put(2, 2);
    cache.get(1);       // returns 1
    cache.put(3, 3);    // evicts key 2
    cache.get(2);       // returns -1 (not found)
    cache.put(4, 4);    // evicts key 1
    cache.get(1);       // returns -1 (not found)
    cache.get(3);       // returns 3
    cache.get(4);       // returns 4
    ```

    **Solution:**
    ```python
    class LRUCache:
        """
        Hash map + Doubly linked list implementation.
        
        Time: O(1) for both get and put
        Space: O(capacity) for storing at most capacity entries
        """
        class Node:
            def __init__(self, key=0, value=0):
                self.key = key
                self.value = value
                self.prev = None
                self.next = None
                
        def __init__(self, capacity):
            self.capacity = capacity
            self.cache = {}  # map key to node
            
            # Initialize doubly linked list with dummy head and tail
            self.head = self.Node()  # Most recently used
            self.tail = self.Node()  # Least recently used
            self.head.next = self.tail
            self.tail.prev = self.head
            
        def _remove_node(self, node):
            # Remove node from list
            p, n = node.prev, node.next
            p.next, n.prev = n, p
            
        def _add_node(self, node):
            # Always add to head (most recently used)
            node.prev = self.head
            node.next = self.head.next
            self.head.next.prev = node
            self.head.next = node
            
        def get(self, key):
            if key in self.cache:
                # Update usage by moving to front
                node = self.cache[key]
                self._remove_node(node)
                self._add_node(node)
                return node.value
            return -1
            
        def put(self, key, value):
            # Remove old key if present
            if key in self.cache:
                self._remove_node(self.cache[key])
                
            # Create new node and add to head
            node = self.Node(key, value)
            self._add_node(node)
            self.cache[key] = node
            
            # Evict if over capacity
            if len(self.cache) > self.capacity:
                lru = self.tail.prev
                self._remove_node(lru)
                del self.cache[lru.key]
    ```

    **Key Insights:**
    - Combining hash map and doubly-linked list gives O(1) operations
    - Hash map enables O(1) lookups by key
    - Doubly-linked list tracks usage order and enables O(1) removals
    - Dummy head/tail nodes simplify edge cases
    - Move-to-front strategy maintains recency order

=== "Subarray Sum Equals K"

    **Problem Statement:**
    Given an array of integers `nums` and an integer `k`, return the total number of continuous subarrays whose sum equals to `k`.

    **Example:**
    ```text
    Input: nums = [1,1,1], k = 2
    Output: 2
    Explanation: [1,1] and [1,1] are two continuous subarrays with sum 2.
    ```

    **Solution:**
    ```python
    def subarraySum(nums, k):
        """
        Use a hashmap to track prefix sums.
        
        Time: O(n) - single pass through the array
        Space: O(n) - store prefix sum counts
        """
        count = 0
        curr_sum = 0
        # Maps prefix sum to number of occurrences
        prefix_sum = {0: 1}  # Initialize with 0 sum occurring once
        
        for num in nums:
            # Update running sum
            curr_sum += num
            
            # Check if we've seen curr_sum - k before
            # If so, it means subarray(s) with sum k exist
            if curr_sum - k in prefix_sum:
                count += prefix_sum[curr_sum - k]
            
            # Update prefix sum frequency
            prefix_sum[curr_sum] = prefix_sum.get(curr_sum, 0) + 1
        
        return count
    ```

    **Key Insights:**
    - Use prefix sums to convert to a "two-sum" like problem
    - Hash map stores the frequency of each prefix sum
    - For each position, check if (currSum - k) exists in our prefix map
    - This indicates a subarray ending at current position with sum k
    - Track frequencies instead of just presence to handle multiple matches

=== "Longest Substring Without Repeating Characters"

    **Problem Statement:**
    Given a string `s`, find the length of the longest substring without repeating characters.

    **Example:**
    ```text
    Input: s = "abcabcbb"
    Output: 3
    Explanation: The answer is "abc", with the length of 3.
    ```

    **Solution:**
    ```python
    def lengthOfLongestSubstring(s):
        """
        Sliding window with hash set.
        
        Time: O(n) - each character is processed at most twice
        Space: O(min(m,n)) where m is the size of the character set
        """
        char_set = set()  # Track characters in current window
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            # If duplicate found, shrink window from left
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
                
            # Add current character and update max length
            char_set.add(s[right])
            max_length = max(max_length, right - left + 1)
        
        return max_length
    ```

    **Key Insights:**
    - Sliding window with two pointers controls the substring range
    - Hash set provides O(1) lookups for character existence
    - Expanding window adds characters, contracting removes them
    - Only need to contract until the duplicate is removed
    - For large character sets, a dictionary mapping chars to indices is more efficient

=== "Top K Frequent Elements"

    **Problem Statement:**
    Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in any order.

    **Example:**
    ```text
    Input: nums = [1,1,1,2,2,3], k = 2
    Output: [1,2]
    Explanation: Elements 1 and 2 appear most frequently.
    ```

    **Solution:**
    ```python
    def topKFrequent(nums, k):
        """
        Frequency counting with bucket sort.
        
        Time: O(n) - linear time counting and retrieval
        Space: O(n) - for the frequency map and buckets
        """
        # Count frequencies
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
            
        # Create frequency buckets
        # bucket[i] contains elements that appear i times
        bucket = [[] for _ in range(len(nums) + 1)]
        
        # Place elements in buckets by frequency
        for num, freq in count.items():
            bucket[freq].append(num)
            
        # Collect top k elements from highest frequencies
        result = []
        for i in range(len(bucket) - 1, 0, -1):
            result.extend(bucket[i])
            if len(result) >= k:
                return result[:k]
                
        return result  # Should not reach here if k is valid
    ```

    **Key Insights:**
    - Hash map counts frequencies in linear time
    - Bucket sort efficiently finds top k without sorting entire array
    - Buckets are indexed by frequency, making retrieval O(n)
    - This is more efficient than heap-based approach (O(n log k))
    - Works well when the range of frequencies is limited
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Create frequency signature
        count = [0] * 26
        for char in s:
            count[ord(char) - ord('a')] += 1
        
        # Convert to tuple (hashable)
        key = tuple(count)
        groups[key].append(s)
    
    return list(groups.values())

# Test
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_anagrams(strs))
```

### Analysis

- **Method 1**: Simple but O(M log M) per string for sorting
- **Method 2**: O(M) per string but uses more space for frequency array
- Hash map groups strings with identical "signatures"

---

## Problem 2: Top K Frequent Elements

**Difficulty**: 🟡 Medium  
**Pattern**: Frequency + Priority  
**Time**: O(N log K), **Space**: O(N)

### Problem Description

Given an integer array `nums` and integer `k`, return the `k` most frequent elements.

```python
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

### Solution Approaches

```python
def top_k_frequent(nums, k):
    """
    Method 1: Hash map + Min heap
    """
    from collections import Counter
    import heapq
    
    # Count frequencies
    freq = Counter(nums)
    
    # Use min heap to keep top k elements
    heap = []
    for num, count in freq.items():
        heapq.heappush(heap, (count, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for count, num in heap]

def top_k_frequent_v2(nums, k):
    """
    Method 2: Bucket sort approach
    """
    from collections import Counter
    
    freq = Counter(nums)
    
    # Create buckets for each frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    
    for num, count in freq.items():
        buckets[count].append(num)
    
    # Collect top k from highest frequency buckets
    result = []
    for i in range(len(buckets) - 1, -1, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    
    return result

def top_k_frequent_v3(nums, k):
    """
    Method 3: Using built-in most_common
    """
    from collections import Counter
    
    freq = Counter(nums)
    return [num for num, count in freq.most_common(k)]

# Test
nums = [1, 1, 1, 2, 2, 3]
k = 2
print(top_k_frequent(nums, k))  # [1, 2]
```

### Key Insights

- **Heap approach**: Best for streaming data, O(N log K)
- **Bucket sort**: O(N) time, leverages frequency bounds
- **Built-in**: Simplest but less educational

---

## Problem 3: Subarray Sum Equals K

**Difficulty**: 🟡 Medium  
**Pattern**: Prefix Sum + Hash Map  
**Time**: O(N), **Space**: O(N)

### Problem Description

Given an array of integers `nums` and integer `k`, return the total number of continuous subarrays whose sum equals `k`.

```python
Input: nums = [1,1,1], k = 2
Output: 2  # [1,1] appears twice
```

### Solution

```python
def subarray_sum(nums, k):
    """
    Use prefix sum with hash map
    Key insight: sum[i:j] = prefix[j] - prefix[i-1]
    So we need: prefix[j] - prefix[i-1] = k
    Which means: prefix[i-1] = prefix[j] - k
    """
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}  # Handle subarrays starting from index 0
    
    for num in nums:
        prefix_sum += num
        
        # Check if (prefix_sum - k) exists
        if (prefix_sum - k) in sum_freq:
            count += sum_freq[prefix_sum - k]
        
        # Add current prefix sum to frequency map
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1
    
    return count

def subarray_sum_detailed(nums, k):
    """
    Version with explanation and subarray tracking
    """
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}
    subarrays = []  # For demonstration
    
    for i, num in enumerate(nums):
        prefix_sum += num
        
        # If we've seen (prefix_sum - k) before,
        # there are subarrays ending at current position with sum k
        target = prefix_sum - k
        if target in sum_freq:
            count += sum_freq[target]
            # Could track actual subarrays here if needed
        
        # Record current prefix sum
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1
    
    return count

# Test
nums = [1, 1, 1]
k = 2
print(subarray_sum(nums, k))  # 2
```

### Key Insights

- Prefix sum transforms subarray sum into difference problem
- Hash map stores frequency of prefix sums seen so far
- Initialize with {0: 1} to handle subarrays from start

---

## Problem 4: Longest Substring Without Repeating Characters

**Difficulty**: 🟡 Medium  
**Pattern**: Sliding Window + Hash Map  
**Time**: O(N), **Space**: O(min(M,N))

### Problem Description

Given string `s`, find length of longest substring without repeating characters.

```python
Input: s = "abcabcbb"
Output: 3  # "abc"
```

### Solution

```python
def length_of_longest_substring(s):
    """
    Sliding window with hash map to track character positions
    """
    char_index = {}
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        # If character seen before and within current window
        if char in char_index and char_index[char] >= start:
            # Move start to after the duplicate
            start = char_index[char] + 1
        
        # Update character's latest position
        char_index[char] = end
        
        # Update max length
        max_length = max(max_length, end - start + 1)
    
    return max_length

def length_of_longest_substring_v2(s):
    """
    Alternative: using set for current window characters
    """
    char_set = set()
    max_length = 0
    left = 0
    
    for right in range(len(s)):
        # Shrink window until no duplicates
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        # Add current character
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Test
s = "abcabcbb"
print(length_of_longest_substring(s))  # 3
```

### Key Insights

- Sliding window maintains valid substring
- Hash map tracks character positions for efficient start adjustment
- Set version is simpler but potentially less efficient

---

## Problem 5: Design HashMap

**Difficulty**: 🟡 Medium  
**Pattern**: Data Structure Design  
**Time**: O(1) average, **Space**: O(N)

### Problem Description

Design a HashMap without using built-in hash table libraries.

### Solution

```python
class MyHashMap:
    """
    Custom HashMap implementation using separate chaining
    """
    
    def __init__(self):
        self.size = 1000
        self.table = [[] for _ in range(self.size)]
    
    def _hash(self, key):
        """Simple hash function"""
        return key % self.size
    
    def put(self, key, value):
        """Insert or update key-value pair"""
        index = self._hash(key)
        bucket = self.table[index]
        
        # Check if key exists, update if found
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # Add new key-value pair
        bucket.append((key, value))
    
    def get(self, key):
        """Get value by key, return -1 if not found"""
        index = self._hash(key)
        bucket = self.table[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        return -1
    
    def remove(self, key):
        """Remove key-value pair"""
        index = self._hash(key)
        bucket = self.table[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return

class MyHashMapLinearProbing:
    """
    Alternative implementation using open addressing
    """
    
    def __init__(self):
        self.size = 1000
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.deleted = [False] * self.size
    
    def _hash(self, key):
        return key % self.size
    
    def _find_slot(self, key):
        """Find slot for key using linear probing"""
        index = self._hash(key)
        
        while (self.keys[index] is not None and 
               self.keys[index] != key and 
               not self.deleted[index]):
            index = (index + 1) % self.size
        
        return index
    
    def put(self, key, value):
        index = self._find_slot(key)
        self.keys[index] = key
        self.values[index] = value
        self.deleted[index] = False
    
    def get(self, key):
        index = self._find_slot(key)
        
        if (self.keys[index] == key and 
            not self.deleted[index]):
            return self.values[index]
        
        return -1
    
    def remove(self, key):
        index = self._find_slot(key)
        
        if (self.keys[index] == key and 
            not self.deleted[index]):
            self.deleted[index] = True

# Test
hashmap = MyHashMap()
hashmap.put(1, 1)
hashmap.put(2, 2)
print(hashmap.get(1))    # 1
print(hashmap.get(3))    # -1
hashmap.put(2, 1)        # update
print(hashmap.get(2))    # 1
hashmap.remove(2)
print(hashmap.get(2))    # -1
```

### Design Considerations

- **Collision Resolution**: Chaining vs. open addressing
- **Hash Function**: Simple modulo vs. more sophisticated
- **Load Factor**: When to resize for performance
- **Deletion**: Mark as deleted vs. actual removal

---

## Problem 6: Four Sum II

**Difficulty**: 🟡 Medium  
**Pattern**: Hash Map Optimization  
**Time**: O(N²), **Space**: O(N²)

### Problem Description

Given four integer arrays, return number of tuples `(i, j, k, l)` such that `nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0`.

### Solution

```python
def four_sum_count(nums1, nums2, nums3, nums4):
    """
    Split into two parts: (A+B) and (C+D)
    Use hash map to store frequencies of A+B sums
    """
    from collections import defaultdict
    
    # Count all possible sums from first two arrays
    sum_freq = defaultdict(int)
    for a in nums1:
        for b in nums2:
            sum_freq[a + b] += 1
    
    count = 0
    # Check if -(C+D) exists in sum_freq
    for c in nums3:
        for d in nums4:
            target = -(c + d)
            count += sum_freq[target]
    
    return count

# Test
nums1 = [1, 2]
nums2 = [-2, -1]
nums3 = [-1, 2]
nums4 = [0, 2]
print(four_sum_count(nums1, nums2, nums3, nums4))  # 2
```

### Key Insights

- Split O(N⁴) problem into two O(N²) problems
- Hash map eliminates need for nested loops in second phase
- Space-time tradeoff: use O(N²) space to save time

---

## 🎯 Practice Summary

### Advanced Patterns

1. **Grouping**: Use computed keys (sorted, frequency signature)
2. **Prefix Sum**: Transform subarray problems into hash lookups
3. **Sliding Window**: Hash map tracks window state efficiently
4. **Design Problems**: Implement core data structure operations
5. **Problem Splitting**: Break complex constraints into manageable parts

### Key Techniques

- Frequency signatures for anagram grouping
- Prefix sum with hash map for subarray problems
- Sliding window with position tracking
- Collision resolution strategies in design problems

### Next Steps

Ready for the ultimate challenge? Try **[Hard Hash Table Problems](hard-problems.md)** featuring:

- LRU Cache implementation
- Advanced system design problems
- Complex multi-data structure solutions

---

*These medium problems showcase the true power of hash tables in algorithm optimization!*
