# Advanced Search Algorithms

This section covers advanced searching techniques beyond basic binary search, including specialized algorithms for complex data structures and optimization problems.

## Exponential Search

Exponential search is useful when the search space is unbounded or when the target is closer to the beginning of the array.

**Time Complexity:** O(log n)  
**Space Complexity:** O(1)  
**Use Case:** Unbounded arrays, when target is near the beginning

```python
def exponential_search(arr, target):
    """
    Exponential search: find range then apply binary search
    """
    if arr[0] == target:
        return 0
    
    # Find range for binary search
    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2
    
    # Apply binary search in found range
    return binary_search_range(arr, target, i // 2, min(i, len(arr) - 1))

def binary_search_range(arr, target, left, right):
    """Binary search in given range"""
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

## Jump Search

Jump search works by jumping ahead by fixed steps, then performing linear search in the identified block.

**Time Complexity:** O(√n)  
**Space Complexity:** O(1)  
**Optimal Jump Size:** √n

```python
import math

def jump_search(arr, target):
    """
    Jump search with optimal jump size of sqrt(n)
    """
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0
    
    # Jump to find the block containing target
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    # Linear search in identified block
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return -1
    
    if arr[prev] == target:
        return prev
    
    return -1
```

## Interpolation Search

Interpolation search improves upon binary search for uniformly distributed data by estimating the target's position.

**Time Complexity:** O(log log n) for uniform data, O(n) worst case  
**Space Complexity:** O(1)  
**Best For:** Uniformly distributed sorted arrays

```python
def interpolation_search(arr, target):
    """
    Interpolation search: estimate position based on value distribution
    """
    left, right = 0, len(arr) - 1
    
    while left <= right and target >= arr[left] and target <= arr[right]:
        if left == right:
            return left if arr[left] == target else -1
        
        # Estimate position using interpolation formula
        pos = left + ((target - arr[left]) * (right - left)) // (arr[right] - arr[left])
        
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    
    return -1
```

## Ternary Search

Ternary search is used to find the maximum or minimum of a unimodal function.

**Time Complexity:** O(log₃ n)  
**Space Complexity:** O(1)  
**Use Case:** Finding extrema in unimodal functions

```python
def ternary_search_max(arr, left, right, epsilon=1e-9):
    """
    Find maximum in unimodal function using ternary search
    """
    while right - left > epsilon:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        if arr[int(mid1)] < arr[int(mid2)]:
            left = mid1
        else:
            right = mid2
    
    return (left + right) / 2

def ternary_search_discrete(arr, target):
    """
    Ternary search for discrete arrays (finding target)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2
        
        if target < arr[mid1]:
            right = mid1 - 1
        elif target > arr[mid2]:
            left = mid2 + 1
        else:
            left = mid1 + 1
            right = mid2 - 1
    
    return -1
```

## Search in 2D Matrix

Various approaches for searching in 2D matrices with different properties.

### Search in Row-wise and Column-wise Sorted Matrix

```python
def search_2d_matrix_sorted(matrix, target):
    """
    Search in matrix where each row and column is sorted
    Time: O(m + n), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    
    row, col = 0, len(matrix[0]) - 1
    
    while row < len(matrix) and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    
    return False

def search_2d_matrix_binary(matrix, target):
    """
    Search in matrix where rows are sorted and first element of each row
    is greater than last element of previous row
    Time: O(log(m*n)), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        row, col = divmod(mid, n)
        
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False
```

## String Searching Algorithms

Advanced string searching techniques beyond naive approach.

### KMP (Knuth-Morris-Pratt) Algorithm

```python
def build_lps(pattern):
    """
    Build Longest Proper Prefix which is also Suffix array
    """
    lps = [0] * len(pattern)
    length = 0
    i = 1
    
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    return lps

def kmp_search(text, pattern):
    """
    KMP string searching algorithm
    Time: O(n + m), Space: O(m)
    """
    if not pattern:
        return []
    
    lps = build_lps(pattern)
    matches = []
    
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == len(pattern):
            matches.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches
```

### Rabin-Karp Algorithm

```python
def rabin_karp_search(text, pattern, prime=101):
    """
    Rabin-Karp string searching using rolling hash
    Time: O(n) average, O(nm) worst case
    """
    if len(pattern) > len(text):
        return []
    
    base = 256
    pattern_len = len(pattern)
    text_len = len(text)
    
    # Calculate hash of pattern and first window
    pattern_hash = 0
    window_hash = 0
    h = 1
    
    # Calculate h = base^(pattern_len-1) % prime
    for i in range(pattern_len - 1):
        h = (h * base) % prime
    
    # Calculate initial hashes
    for i in range(pattern_len):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % prime
        window_hash = (base * window_hash + ord(text[i])) % prime
    
    matches = []
    
    # Slide the pattern over text
    for i in range(text_len - pattern_len + 1):
        if pattern_hash == window_hash:
            # Check character by character
            if text[i:i + pattern_len] == pattern:
                matches.append(i)
        
        # Calculate next window hash
        if i < text_len - pattern_len:
            window_hash = (base * (window_hash - ord(text[i]) * h) + ord(text[i + pattern_len])) % prime
            
            # Handle negative hash
            if window_hash < 0:
                window_hash += prime
    
    return matches
```

## Specialized Search Algorithms

### Union-Find for Connectivity Queries

```python
class UnionFind:
    """
    Union-Find data structure for connectivity queries
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank"""
        root_x, root_y = self.find(x), self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x, y):
        """Check if two elements are connected"""
        return self.find(x) == self.find(y)
```

### Trie for Prefix Searches

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # Number of words passing through this node

class Trie:
    """
    Trie for efficient prefix-based searches
    """
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert word into trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True
    
    def search(self, word):
        """Search for exact word"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        """Check if any word starts with prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def count_words_with_prefix(self, prefix):
        """Count words with given prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.count
```

## Performance Comparison

| Algorithm | Time Complexity | Space | Best Use Case |
|-----------|----------------|-------|---------------|
| Binary Search | O(log n) | O(1) | Sorted arrays |
| Exponential Search | O(log n) | O(1) | Unbounded/infinite arrays |
| Jump Search | O(√n) | O(1) | Large sorted arrays |
| Interpolation Search | O(log log n) | O(1) | Uniformly distributed data |
| Ternary Search | O(log₃ n) | O(1) | Unimodal functions |
| KMP | O(n + m) | O(m) | String pattern matching |
| Rabin-Karp | O(n) avg | O(1) | Multiple pattern search |

## Advanced Applications

1. **Database Systems**: B+ trees, hash indexing
2. **Information Retrieval**: Inverted indices, search engines
3. **Computational Geometry**: Range queries, nearest neighbor
4. **Graph Algorithms**: Path finding, connectivity queries
5. **Machine Learning**: Feature selection, hyperparameter optimization

These advanced search algorithms form the foundation for many complex systems and are essential for solving optimization problems efficiently.
