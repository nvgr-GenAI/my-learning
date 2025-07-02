# Problem-Solving Patterns & Techniques

Master these fundamental patterns to solve 80% of coding interview problems efficiently. Understanding these patterns will help you recognize similar problems and apply proven solutions quickly.

## 🎯 Core Patterns Overview

| Pattern | Use Cases | Time Complexity | Space Complexity |
|---------|-----------|-----------------|------------------|
| **Two Pointers** | Array problems, palindromes, sum problems | O(n) | O(1) |
| **Sliding Window** | Subarray/substring with constraints | O(n) | O(1) or O(k) |
| **Fast & Slow Pointers** | Cycle detection, middle elements | O(n) | O(1) |
| **Merge Intervals** | Overlapping intervals, scheduling | O(n log n) | O(n) |
| **Hash Table** | Frequency counting, lookups | O(n) | O(n) |
| **Tree Traversal** | Tree/graph problems | O(n) | O(h) |
| **Binary Search** | Sorted data, search space | O(log n) | O(1) |
| **Backtracking** | Permutations, combinations | O(b^d) | O(d) |
| **Dynamic Programming** | Optimization problems | O(n²) typical | O(n) |
| **Greedy** | Local optimization | O(n log n) | O(1) |

## 🔄 Two Pointers Technique

**Best for**: Array problems, string palindromes, sum problems, sorting-related tasks

### When to Use
- Input is sorted (or can be sorted)
- Need to find pairs that meet criteria
- Checking palindromes
- Removing duplicates

### Template
```python
def two_pointers_template(arr):
    left, right = 0, len(arr) - 1
    
    while left < right:
        # Process current pair
        if meets_condition(arr[left], arr[right]):
            # Found solution
            return result
        elif should_move_left(arr[left], arr[right]):
            left += 1
        else:
            right -= 1
    
    return default_result
```

### Classic Problems

#### Two Sum (Sorted Array)
```python
def two_sum_sorted(arr, target):
    """Find two numbers that sum to target in sorted array."""
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
```

#### Valid Palindrome
```python
def is_palindrome(s):
    """Check if string is palindrome ignoring case and non-alphanumeric."""
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True
```

#### Container With Most Water
```python
def max_area(heights):
    """Find container that holds the most water."""
    left, right = 0, len(heights) - 1
    max_water = 0
    
    while left < right:
        width = right - left
        height = min(heights[left], heights[right])
        max_water = max(max_water, width * height)
        
        # Move pointer with smaller height
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    
    return max_water
```

## 🪟 Sliding Window Pattern

**Best for**: Subarray/substring problems with size constraints, optimization problems

### When to Use
- Fixed window size problems
- Variable window size with constraints
- Maximum/minimum subarray problems
- Character frequency problems

### Fixed Window Template
```python
def fixed_window_template(arr, k):
    window_sum = sum(arr[:k])  # Initialize first window
    result = window_sum
    
    for i in range(k, len(arr)):
        # Slide window: remove left, add right
        window_sum = window_sum - arr[i - k] + arr[i]
        result = update_result(result, window_sum)
    
    return result
```

### Variable Window Template
```python
def variable_window_template(arr, condition):
    left = 0
    result = 0
    
    for right in range(len(arr)):
        # Expand window
        add_to_window(arr[right])
        
        # Contract window while condition violated
        while window_violates_condition():
            remove_from_window(arr[left])
            left += 1
        
        # Update result with current valid window
        result = update_result(result, right - left + 1)
    
    return result
```

### Classic Problems

#### Maximum Sum Subarray of Size K
```python
def max_sum_subarray(arr, k):
    """Find maximum sum of subarray of size k."""
    if len(arr) < k:
        return None
    
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

#### Longest Substring with K Distinct Characters
```python
def longest_substring_k_distinct(s, k):
    """Find longest substring with at most k distinct characters."""
    if not s or k == 0:
        return 0
    
    char_count = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Expand window
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Contract window if needed
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

#### Minimum Window Substring
```python
def min_window_substring(s, t):
    """Find minimum window substring containing all characters of t."""
    if not s or not t:
        return ""
    
    # Count characters in t
    t_count = {}
    for char in t:
        t_count[char] = t_count.get(char, 0) + 1
    
    left = 0
    min_len = float('inf')
    min_start = 0
    required = len(t_count)
    formed = 0
    window_counts = {}
    
    for right in range(len(s)):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1
        
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1
        
        # Try to contract window
        while left <= right and formed == required:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_start = left
            
            char = s[left]
            window_counts[char] -= 1
            if char in t_count and window_counts[char] < t_count[char]:
                formed -= 1
            
            left += 1
    
    return "" if min_len == float('inf') else s[min_start:min_start + min_len]
```

## 🐢🐰 Fast & Slow Pointers (Floyd's Algorithm)

**Best for**: Cycle detection, finding middle elements, palindrome linked lists

### When to Use
- Linked list cycle detection
- Finding middle of linked list
- Detecting happy numbers
- Palindrome linked lists

### Template
```python
def fast_slow_template(head):
    if not head or not head.next:
        return handle_edge_case()
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if meets_condition(slow, fast):
            return process_result(slow, fast)
    
    return default_result
```

### Classic Problems

#### Detect Cycle in Linked List
```python
def has_cycle(head):
    """Detect cycle in linked list."""
    if not head or not head.next:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False
```

#### Find Start of Cycle
```python
def find_cycle_start(head):
    """Find the start of cycle in linked list."""
    if not head or not head.next:
        return None
    
    # Phase 1: Detect cycle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle
    
    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

#### Find Middle of Linked List
```python
def find_middle(head):
    """Find middle node of linked list."""
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow
```

#### Happy Number
```python
def is_happy_number(n):
    """Check if number is happy."""
    def get_sum_of_squares(num):
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total
    
    slow = fast = n
    
    while True:
        slow = get_sum_of_squares(slow)
        fast = get_sum_of_squares(get_sum_of_squares(fast))
        
        if fast == 1:
            return True
        if slow == fast:
            return False
```

## 📅 Merge Intervals Pattern

**Best for**: Overlapping intervals, scheduling problems, calendar applications

### When to Use
- Merging overlapping intervals
- Scheduling conflicts
- Range problems
- Meeting room allocation

### Template
```python
def merge_intervals_template(intervals):
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        
        if overlaps(current, last_merged):
            last_merged[1] = merge_end(current, last_merged)
        else:
            merged.append(current)
    
    return merged
```

### Classic Problems

#### Merge Overlapping Intervals
```python
def merge_intervals(intervals):
    """Merge overlapping intervals."""
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        
        if current[0] <= last_merged[1]:  # Overlap
            last_merged[1] = max(last_merged[1], current[1])
        else:
            merged.append(current)
    
    return merged
```

#### Insert Interval
```python
def insert_interval(intervals, new_interval):
    """Insert new interval and merge if necessary."""
    result = []
    i = 0
    
    # Add intervals that end before new interval starts
    while i < len(intervals) and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals
    while i < len(intervals) and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    
    result.append(new_interval)
    
    # Add remaining intervals
    while i < len(intervals):
        result.append(intervals[i])
        i += 1
    
    return result
```

#### Meeting Rooms
```python
def can_attend_meetings(intervals):
    """Check if person can attend all meetings."""
    intervals.sort(key=lambda x: x[0])
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False
    
    return True

def min_meeting_rooms(intervals):
    """Find minimum number of meeting rooms required."""
    if not intervals:
        return 0
    
    import heapq
    
    intervals.sort(key=lambda x: x[0])
    heap = []  # Track end times
    
    for interval in intervals:
        start, end = interval
        
        # Remove meetings that have ended
        while heap and heap[0] <= start:
            heapq.heappop(heap)
        
        heapq.heappush(heap, end)
    
    return len(heap)
```

## 🔍 Binary Search Pattern

**Best for**: Sorted arrays, search space problems, optimization problems

### When to Use
- Searching in sorted data
- Finding boundaries (first/last occurrence)
- Search space reduction
- Optimization problems with monotonic property

### Template
```python
def binary_search_template(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Not found
```

### Advanced Binary Search
```python
def find_boundary(arr, target, find_first=True):
    """Find first or last occurrence of target."""
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            if find_first:
                right = mid - 1  # Continue searching left
            else:
                left = mid + 1   # Continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

## 🌳 Tree Traversal Patterns

**Best for**: Tree and graph problems, hierarchical data processing

### DFS Templates
```python
# Preorder: Root -> Left -> Right
def preorder_dfs(root):
    if not root:
        return []
    
    result = [root.val]
    result.extend(preorder_dfs(root.left))
    result.extend(preorder_dfs(root.right))
    return result

# Inorder: Left -> Root -> Right (sorted for BST)
def inorder_dfs(root):
    if not root:
        return []
    
    result = []
    result.extend(inorder_dfs(root.left))
    result.append(root.val)
    result.extend(inorder_dfs(root.right))
    return result

# Postorder: Left -> Right -> Root
def postorder_dfs(root):
    if not root:
        return []
    
    result = []
    result.extend(postorder_dfs(root.left))
    result.extend(postorder_dfs(root.right))
    result.append(root.val)
    return result
```

### BFS Template
```python
def level_order_bfs(root):
    """Level-order traversal using BFS."""
    if not root:
        return []
    
    result = []
    queue = [root]
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.pop(0)
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

## 📈 Pattern Recognition Guide

### How to Identify Patterns

| Problem Characteristics | Likely Pattern |
|------------------------|----------------|
| Sorted array, find pair/triplet | Two Pointers |
| Subarray with size constraint | Sliding Window |
| Linked list cycle/middle | Fast & Slow Pointers |
| Overlapping ranges | Merge Intervals |
| Frequency counting | Hash Table |
| Tree/graph traversal | DFS/BFS |
| Sorted search space | Binary Search |
| Generate all possibilities | Backtracking |
| Optimization with subproblems | Dynamic Programming |
| Local optimal choices | Greedy |

### Pattern Combinations

Many problems use multiple patterns:

- **Sliding Window + Hash Table**: Longest substring problems
- **Two Pointers + Sorting**: 3Sum, 4Sum problems
- **DFS + Backtracking**: Generate all paths in tree
- **Binary Search + Two Pointers**: Search in 2D matrix
- **BFS + Hash Table**: Shortest path with constraints

## 🎯 Practice Strategy

### Pattern-Based Learning

1. **Master One Pattern**: Start with Two Pointers
2. **Solve 5-10 Problems**: Use same pattern
3. **Identify Variations**: Notice pattern modifications
4. **Move to Next Pattern**: Build on previous knowledge
5. **Combine Patterns**: Solve complex problems

### Recommended Order

1. **Two Pointers** (easiest to understand)
2. **Sliding Window** (builds on two pointers)
3. **Fast & Slow Pointers** (specific application)
4. **Hash Table** (fundamental technique)
5. **Tree Traversal** (essential for trees/graphs)
6. **Binary Search** (important optimization)
7. **Merge Intervals** (scheduling problems)
8. **Backtracking** (more complex)
9. **Dynamic Programming** (most challenging)
10. **Greedy** (problem-specific)

---

Master these patterns and you'll be able to solve most coding interview problems efficiently! 🚀
