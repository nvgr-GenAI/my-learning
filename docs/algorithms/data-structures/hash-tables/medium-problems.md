# Hash Tables - Medium Problems

## 🎯 Learning Objectives

Master intermediate hash table techniques:

- Advanced frequency counting and grouping
- Sliding window with hash maps  
- Custom hash map design
- Complex key-value relationships

---

## Problem 1: Group Anagrams

**Difficulty**: 🟡 Medium  
**Pattern**: Grouping with Hash Keys  
**Time**: O(N × M log M), **Space**: O(N × M)

### Problem Description

Given an array of strings `strs`, group the anagrams together.

```python
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

### Solution Approaches

```python
def group_anagrams(strs):
    """
    Method 1: Use sorted string as hash key
    """
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Sorted string as unique key for anagrams
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())

def group_anagrams_v2(strs):
    """
    Method 2: Use character frequency as hash key
    """
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
