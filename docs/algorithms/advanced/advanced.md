# Advanced Algorithm Techniques

This section covers sophisticated algorithmic paradigms and techniques used to solve complex computational problems efficiently. These methods are essential for competitive programming and advanced software development.

## Greedy Algorithms

Greedy algorithms make locally optimal choices at each step, hoping to find a global optimum. They work well for problems with optimal substructure and greedy choice property.

### Activity Selection Problem

```python
def activity_selection(activities):
    """
    Select maximum number of non-overlapping activities
    Activities: list of (start, end) tuples
    Time: O(n log n), Space: O(1)
    """
    # Sort by end time
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_end = activities[0][1]
    
    for start, end in activities[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end
    
    return selected

# Example usage
activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
result = activity_selection(activities)
print(f"Selected activities: {result}")
```

### Huffman Coding

```python
import heapq
from collections import defaultdict, Counter

class Node:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(text):
    """
    Build Huffman tree and encode text
    Time: O(n log k) where k is number of unique characters
    """
    if not text:
        return "", {}
    
    # Count frequencies
    freq = Counter(text)
    
    # Special case: single character
    if len(freq) == 1:
        char = list(freq.keys())[0]
        return '0' * len(text), {char: '0'}
    
    # Build min heap
    heap = [Node(char, freq) for char, freq in freq.items()]
    heapq.heapify(heap)
    
    # Build Huffman tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged = Node(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    root = heap[0]
    
    # Generate codes
    codes = {}
    
    def generate_codes(node, code=""):
        if node.char:  # Leaf node
            codes[node.char] = code if code else "0"
        else:
            generate_codes(node.left, code + "0")
            generate_codes(node.right, code + "1")
    
    generate_codes(root)
    
    # Encode text
    encoded = ''.join(codes[char] for char in text)
    
    return encoded, codes

def huffman_decoding(encoded_text, codes):
    """Decode Huffman encoded text"""
    if not encoded_text:
        return ""
    
    # Reverse the codes dictionary
    reverse_codes = {code: char for char, code in codes.items()}
    
    decoded = []
    current_code = ""
    
    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_codes:
            decoded.append(reverse_codes[current_code])
            current_code = ""
    
    return ''.join(decoded)
```

### Fractional Knapsack

```python
def fractional_knapsack(capacity, items):
    """
    Fractional knapsack problem using greedy approach
    Items: list of (value, weight) tuples
    Time: O(n log n), Space: O(1)
    """
    # Calculate value-to-weight ratio and sort
    items_with_ratio = [(value/weight, value, weight) for value, weight in items]
    items_with_ratio.sort(reverse=True)
    
    total_value = 0
    remaining_capacity = capacity
    result = []
    
    for ratio, value, weight in items_with_ratio:
        if remaining_capacity == 0:
            break
        
        if weight <= remaining_capacity:
            # Take the whole item
            total_value += value
            remaining_capacity -= weight
            result.append((value, weight, 1.0))  # fraction = 1.0
        else:
            # Take fraction of the item
            fraction = remaining_capacity / weight
            total_value += value * fraction
            result.append((value, weight, fraction))
            remaining_capacity = 0
    
    return total_value, result
```

## Divide and Conquer

Divide and conquer algorithms solve problems by breaking them into smaller subproblems, solving them recursively, and combining the results.

### Quick Select (Kth Largest Element)

```python
import random

def quick_select(arr, k):
    """
    Find kth largest element using quickselect
    Average Time: O(n), Worst: O(n²), Space: O(log n)
    """
    if not arr or k < 1 or k > len(arr):
        return None
    
    def partition(left, right, pivot_idx):
        pivot_value = arr[pivot_idx]
        # Move pivot to end
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        
        store_idx = left
        for i in range(left, right):
            if arr[i] > pivot_value:  # For kth largest
                arr[i], arr[store_idx] = arr[store_idx], arr[i]
                store_idx += 1
        
        # Move pivot to final position
        arr[store_idx], arr[right] = arr[right], arr[store_idx]
        return store_idx
    
    def select(left, right, k):
        if left == right:
            return arr[left]
        
        # Random pivot for better average performance
        pivot_idx = random.randint(left, right)
        pivot_idx = partition(left, right, pivot_idx)
        
        if k == pivot_idx + 1:
            return arr[pivot_idx]
        elif k < pivot_idx + 1:
            return select(left, pivot_idx - 1, k)
        else:
            return select(pivot_idx + 1, right, k)
    
    return select(0, len(arr) - 1, k)
```

### Maximum Subarray (Divide and Conquer)

```python
def max_subarray_dc(arr):
    """
    Find maximum subarray sum using divide and conquer
    Time: O(n log n), Space: O(log n)
    """
    def max_crossing_sum(arr, left, mid, right):
        """Find max sum crossing the midpoint"""
        left_sum = float('-inf')
        current_sum = 0
        
        for i in range(mid, left - 1, -1):
            current_sum += arr[i]
            left_sum = max(left_sum, current_sum)
        
        right_sum = float('-inf')
        current_sum = 0
        
        for i in range(mid + 1, right + 1):
            current_sum += arr[i]
            right_sum = max(right_sum, current_sum)
        
        return left_sum + right_sum
    
    def max_subarray_rec(arr, left, right):
        if left == right:
            return arr[left]
        
        mid = (left + right) // 2
        
        left_max = max_subarray_rec(arr, left, mid)
        right_max = max_subarray_rec(arr, mid + 1, right)
        cross_max = max_crossing_sum(arr, left, mid, right)
        
        return max(left_max, right_max, cross_max)
    
    if not arr:
        return 0
    
    return max_subarray_rec(arr, 0, len(arr) - 1)
```

## Backtracking

Backtracking systematically explores all possible solutions by building candidates incrementally and abandoning candidates that cannot lead to a valid solution.

### N-Queens Problem

```python
def solve_n_queens(n):
    """
    Solve N-Queens problem using backtracking
    Returns all possible solutions
    """
    def is_safe(board, row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 1:
                return False
        
        # Check upper diagonal on left side
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 1:
                return False
            i, j = i - 1, j - 1
        
        # Check upper diagonal on right side
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 1:
                return False
            i, j = i - 1, j + 1
        
        return True
    
    def solve(board, row):
        if row >= n:
            # Found a solution
            solutions.append([row[:] for row in board])
            return
        
        for col in range(n):
            if is_safe(board, row, col):
                board[row][col] = 1
                solve(board, row + 1)
                board[row][col] = 0  # Backtrack
    
    solutions = []
    board = [[0] * n for _ in range(n)]
    solve(board, 0)
    
    return solutions

def print_board(board):
    """Helper function to print the board"""
    n = len(board)
    for i in range(n):
        for j in range(n):
            print('Q' if board[i][j] == 1 else '.', end=' ')
        print()
```

### Sudoku Solver

```python
def solve_sudoku(board):
    """
    Solve 9x9 Sudoku using backtracking
    Modifies board in-place, returns True if solved
    """
    def is_valid(board, row, col, num):
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def find_empty():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return i, j
        return None
    
    def solve():
        empty = find_empty()
        if not empty:
            return True  # Solved
        
        row, col = empty
        
        for num in range(1, 10):
            if is_valid(board, row, col, num):
                board[row][col] = num
                
                if solve():
                    return True
                
                board[row][col] = 0  # Backtrack
        
        return False
    
    return solve()
```

### Generate Permutations

```python
def generate_permutations(nums):
    """
    Generate all permutations using backtracking
    Time: O(n! × n), Space: O(n)
    """
    def backtrack(current_perm):
        if len(current_perm) == len(nums):
            result.append(current_perm[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            
            used[i] = True
            current_perm.append(nums[i])
            backtrack(current_perm)
            current_perm.pop()
            used[i] = False
    
    result = []
    used = [False] * len(nums)
    backtrack([])
    
    return result

def generate_combinations(nums, k):
    """
    Generate all combinations of size k
    Time: O(C(n,k) × k), Space: O(k)
    """
    def backtrack(start, current_comb):
        if len(current_comb) == k:
            result.append(current_comb[:])
            return
        
        for i in range(start, len(nums)):
            current_comb.append(nums[i])
            backtrack(i + 1, current_comb)
            current_comb.pop()
    
    result = []
    backtrack(0, [])
    
    return result
```

## Bit Manipulation

Bit manipulation techniques can lead to very efficient solutions for certain problems.

### Common Bit Operations

```python
def bit_operations_demo():
    """Demonstrate common bit manipulation operations"""
    
    # Check if number is power of 2
    def is_power_of_2(n):
        return n > 0 and (n & (n - 1)) == 0
    
    # Count number of set bits
    def count_set_bits(n):
        count = 0
        while n:
            count += n & 1
            n >>= 1
        return count
    
    # Brian Kernighan's algorithm for counting set bits
    def count_set_bits_efficient(n):
        count = 0
        while n:
            n &= n - 1  # Clear the lowest set bit
            count += 1
        return count
    
    # Find the only non-repeated element
    def single_number(nums):
        result = 0
        for num in nums:
            result ^= num
        return result
    
    # Swap two numbers without temp variable
    def swap_without_temp(a, b):
        a ^= b
        b ^= a
        a ^= b
        return a, b
    
    # Check if bit at position i is set
    def is_bit_set(n, i):
        return (n & (1 << i)) != 0
    
    # Set bit at position i
    def set_bit(n, i):
        return n | (1 << i)
    
    # Clear bit at position i
    def clear_bit(n, i):
        return n & ~(1 << i)
    
    # Toggle bit at position i
    def toggle_bit(n, i):
        return n ^ (1 << i)
    
    return {
        'is_power_of_2': is_power_of_2,
        'count_set_bits': count_set_bits_efficient,
        'single_number': single_number,
        'swap_without_temp': swap_without_temp,
        'is_bit_set': is_bit_set,
        'set_bit': set_bit,
        'clear_bit': clear_bit,
        'toggle_bit': toggle_bit
    }
```

### Subset Generation

```python
def generate_subsets(nums):
    """
    Generate all subsets using bit manipulation
    Time: O(n × 2^n), Space: O(1) extra
    """
    n = len(nums)
    result = []
    
    # Iterate through all possible bitmasks
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    
    return result

def generate_subsets_backtrack(nums):
    """Alternative backtracking approach"""
    def backtrack(start, current_subset):
        result.append(current_subset[:])
        
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()
    
    result = []
    backtrack(0, [])
    return result
```

## Two Pointers Technique

Two pointers is a technique used to solve array and string problems efficiently.

### Two Sum in Sorted Array

```python
def two_sum_sorted(arr, target):
    """
    Find two numbers that sum to target in sorted array
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []

def three_sum(nums):
    """
    Find all unique triplets that sum to zero
    Time: O(n²), Space: O(1)
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, n - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if current_sum == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1
    
    return result
```

### Container With Most Water

```python
def max_area(height):
    """
    Find two lines that form container with most water
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        # Calculate current area
        width = right - left
        current_height = min(height[left], height[right])
        current_water = width * current_height
        max_water = max(max_water, current_water)
        
        # Move the pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water
```

## Sliding Window Technique

Sliding window is used to find subarrays or substrings that satisfy certain conditions.

### Longest Substring Without Repeating Characters

```python
def longest_unique_substring(s):
    """
    Find length of longest substring without repeating characters
    Time: O(n), Space: O(min(m,n)) where m is charset size
    """
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Shrink window until no duplicates
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

def min_window_substring(s, t):
    """
    Find minimum window substring containing all characters of t
    Time: O(|s| + |t|), Space: O(|t|)
    """
    if not s or not t:
        return ""
    
    from collections import Counter, defaultdict
    
    dict_t = Counter(t)
    required = len(dict_t)
    
    left = right = 0
    formed = 0
    
    window_counts = defaultdict(int)
    
    # ans = (window length, left, right)
    ans = float('inf'), None, None
    
    while right < len(s):
        # Add character from right to window
        char = s[right]
        window_counts[char] += 1
        
        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1
        
        # Try to shrink window from left
        while left <= right and formed == required:
            char = s[left]
            
            # Update answer if this window is smaller
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)
            
            # Remove from left of window
            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1
            
            left += 1
        
        right += 1
    
    return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]
```

## Advanced Graph Algorithms

### Topological Sort

```python
from collections import defaultdict, deque

def topological_sort_kahn(graph, num_vertices):
    """
    Kahn's algorithm for topological sorting
    Time: O(V + E), Space: O(V)
    """
    # Calculate in-degrees
    in_degree = [0] * num_vertices
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    
    # Initialize queue with vertices having 0 in-degree
    queue = deque([i for i in range(num_vertices) if in_degree[i] == 0])
    result = []
    
    while queue:
        u = queue.popleft()
        result.append(u)
        
        # Reduce in-degree of adjacent vertices
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Check for cycle
    if len(result) != num_vertices:
        return []  # Cycle detected
    
    return result

def topological_sort_dfs(graph, num_vertices):
    """
    DFS-based topological sorting
    Time: O(V + E), Space: O(V)
    """
    visited = [False] * num_vertices
    stack = []
    
    def dfs(v):
        visited[v] = True
        
        for u in graph[v]:
            if not visited[u]:
                dfs(u)
        
        stack.append(v)
    
    # Visit all vertices
    for i in range(num_vertices):
        if not visited[i]:
            dfs(i)
    
    return stack[::-1]  # Reverse to get topological order
```

## Resources

- [Competitive Programming Handbook](https://cses.fi/book/book.pdf)
- [Algorithm Design Manual](http://algorist.com/)
- [Bit Manipulation Tricks](https://graphics.stanford.edu/~seander/bithacks.html)

---

*These advanced techniques form the backbone of efficient algorithm design. Master them through practice and application to complex problems!*
