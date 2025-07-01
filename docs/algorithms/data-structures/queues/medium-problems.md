# Queues: Medium Problems

## 🎯 Intermediate Queue Challenges

These problems require combining queues with other data structures and more sophisticated algorithms.

---

## Problem 1: Sliding Window Maximum

**Difficulty:** Medium  
**Pattern:** Monotonic Deque  
**Time:** O(n) | **Space:** O(k)

### Problem Statement

You are given an array of integers `nums` and a sliding window of size `k`. Return the maximum value in each sliding window as it moves from left to right.

**Example:**

```text
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
```

### Solution

```python
from collections import deque

def max_sliding_window(nums, k):
    """
    Find maximum in each sliding window using monotonic deque.
    
    Deque stores indices in decreasing order of values.
    """
    if not nums or k == 0:
        return []
    
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove indices with smaller values (maintain decreasing order)
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add result when window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])  # Front has maximum
    
    return result

# Alternative: Brute force approach (less efficient)
def max_sliding_window_brute(nums, k):
    """Brute force: O(n*k) solution."""
    result = []
    for i in range(len(nums) - k + 1):
        window_max = max(nums[i:i+k])
        result.append(window_max)
    return result

# Test both approaches
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print(f"Optimized: {max_sliding_window(nums, k)}")
print(f"Brute force: {max_sliding_window_brute(nums, k)}")
```

### Explanation

1. **Monotonic Deque**: Maintain decreasing order of values
2. **Window Management**: Remove indices outside current window
3. **Optimization**: Front of deque always contains window maximum

---

## Problem 2: Perfect Squares

**Difficulty:** Medium  
**Pattern:** BFS with Queue  
**Time:** O(n×√n) | **Space:** O(n)

### Problem Statement

Given an integer `n`, return the least number of perfect square numbers that sum to `n`.

**Example:**

```text
Input: n = 12
Output: 3
Explanation: 12 = 4 + 4 + 4
```

### Solution

```python
from collections import deque
import math

def num_squares_bfs(n):
    """
    Find minimum squares using BFS.
    
    Each level represents using one more square.
    """
    if n <= 0:
        return 0
    
    # Generate perfect squares up to n
    squares = []
    i = 1
    while i * i <= n:
        squares.append(i * i)
        i += 1
    
    # BFS to find minimum steps
    queue = deque([n])
    visited = {n}
    level = 0
    
    while queue:
        level += 1
        for _ in range(len(queue)):
            current = queue.popleft()
            
            for square in squares:
                if square > current:
                    break
                
                next_val = current - square
                if next_val == 0:
                    return level
                
                if next_val not in visited:
                    visited.add(next_val)
                    queue.append(next_val)
    
    return level

# Alternative: Dynamic Programming approach
def num_squares_dp(n):
    """
    DP approach: O(n×√n) time, O(n) space.
    """
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    squares = [i * i for i in range(1, int(math.sqrt(n)) + 1)]
    
    for i in range(1, n + 1):
        for square in squares:
            if square > i:
                break
            dp[i] = min(dp[i], dp[i - square] + 1)
    
    return dp[n]

# Test both approaches
test_cases = [12, 13, 1, 4, 7]
for n in test_cases:
    bfs_result = num_squares_bfs(n)
    dp_result = num_squares_dp(n)
    print(f"n={n}: BFS={bfs_result}, DP={dp_result}")
```

---

## Problem 3: Open the Lock

**Difficulty:** Medium  
**Pattern:** BFS State Space Search  
**Time:** O(10^4) | **Space:** O(10^4)

### Problem Statement

You have a lock with 4 circular wheels, each with 10 slots: '0' to '9'. Starting at "0000", find the minimum number of turns to reach the target, avoiding deadends.

**Example:**

```text
Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202"
Output: 6
```

### Solution

```python
from collections import deque

def open_lock(deadends, target):
    """
    Find minimum turns using BFS on state space.
    
    Each state is a 4-digit combination.
    """
    if "0000" in deadends:
        return -1
    if target == "0000":
        return 0
    
    deadend_set = set(deadends)
    visited = {"0000"}
    queue = deque([("0000", 0)])
    
    def get_neighbors(state):
        """Generate all possible next states."""
        neighbors = []
        for i in range(4):
            digit = int(state[i])
            
            # Turn up
            new_digit = (digit + 1) % 10
            new_state = state[:i] + str(new_digit) + state[i+1:]
            neighbors.append(new_state)
            
            # Turn down  
            new_digit = (digit - 1) % 10
            new_state = state[:i] + str(new_digit) + state[i+1:]
            neighbors.append(new_state)
        
        return neighbors
    
    while queue:
        current_state, turns = queue.popleft()
        
        for neighbor in get_neighbors(current_state):
            if neighbor == target:
                return turns + 1
            
            if neighbor not in visited and neighbor not in deadend_set:
                visited.add(neighbor)
                queue.append((neighbor, turns + 1))
    
    return -1

# Optimized version with bidirectional BFS
def open_lock_bidirectional(deadends, target):
    """
    Bidirectional BFS for better performance.
    """
    if "0000" in deadends:
        return -1
    if target == "0000":
        return 0
    
    deadend_set = set(deadends)
    begin_set = {"0000"}
    end_set = {target}
    visited = set()
    
    def get_neighbors(state):
        neighbors = []
        for i in range(4):
            digit = int(state[i])
            for delta in [1, -1]:
                new_digit = (digit + delta) % 10
                new_state = state[:i] + str(new_digit) + state[i+1:]
                neighbors.append(new_state)
        return neighbors
    
    level = 0
    while begin_set and end_set:
        level += 1
        
        # Always expand the smaller set
        if len(begin_set) > len(end_set):
            begin_set, end_set = end_set, begin_set
        
        next_set = set()
        for state in begin_set:
            for neighbor in get_neighbors(state):
                if neighbor in end_set:
                    return level
                
                if neighbor not in visited and neighbor not in deadend_set:
                    visited.add(neighbor)
                    next_set.add(neighbor)
        
        begin_set = next_set
    
    return -1

# Test
deadends = ["0201", "0101", "0102", "1212", "2002"]
target = "0202"
print(f"Regular BFS: {open_lock(deadends, target)}")
print(f"Bidirectional BFS: {open_lock_bidirectional(deadends, target)}")
```

---

## Problem 4: Walls and Gates

**Difficulty:** Medium  
**Pattern:** Multi-source BFS  
**Time:** O(m×n) | **Space:** O(m×n)

### Problem Statement

Fill each empty room with the distance to its nearest gate. Given a 2D grid with:
- `-1`: Wall or obstacle
- `0`: Gate  
- `INF`: Empty room

### Solution

```python
from collections import deque

def walls_and_gates(rooms):
    """
    Fill rooms with distance to nearest gate using multi-source BFS.
    
    Start BFS from all gates simultaneously.
    """
    if not rooms or not rooms[0]:
        return
    
    rows, cols = len(rooms), len(rooms[0])
    queue = deque()
    INF = 2147483647
    
    # Find all gates and add to queue
    for i in range(rows):
        for j in range(cols):
            if rooms[i][j] == 0:
                queue.append((i, j))
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        row, col = queue.popleft()
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds and if it's an empty room farther than current
            if (0 <= new_row < rows and 
                0 <= new_col < cols and 
                rooms[new_row][new_col] == INF):
                
                rooms[new_row][new_col] = rooms[row][col] + 1
                queue.append((new_row, new_col))

# Test function
def print_grid(grid):
    """Helper to print grid nicely."""
    for row in grid:
        print([x if x != 2147483647 else 'INF' for x in row])

# Test case
INF = 2147483647
rooms = [
    [INF, -1, 0, INF],
    [INF, INF, INF, -1],
    [INF, -1, INF, -1],
    [0, -1, INF, INF]
]

print("Before:")
print_grid(rooms)

walls_and_gates(rooms)

print("\nAfter:")
print_grid(rooms)
```

---

## Problem 5: Design Hit Counter

**Difficulty:** Medium  
**Pattern:** Queue with Time Windows  
**Time:** O(1) amortized | **Space:** O(1)

### Problem Statement

Design a hit counter that counts hits in the past 5 minutes (300 seconds).

### Solution

```python
from collections import deque

class HitCounter:
    """
    Count hits in the past 300 seconds using queue.
    
    Each hit is stored with its timestamp.
    """
    
    def __init__(self):
        self.hits = deque()
    
    def hit(self, timestamp):
        """Record a hit at given timestamp."""
        self.hits.append(timestamp)
    
    def get_hits(self, timestamp):
        """Get number of hits in past 300 seconds."""
        # Remove hits older than 300 seconds
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.popleft()
        
        return len(self.hits)

# Alternative: Bucket approach for better space efficiency
class HitCounterBucket:
    """
    Use circular array with 300 buckets (one per second).
    
    More space-efficient for high-frequency hits.
    """
    
    def __init__(self):
        self.times = [0] * 300
        self.hits = [0] * 300
    
    def hit(self, timestamp):
        """Record hit at timestamp."""
        idx = timestamp % 300
        
        if self.times[idx] != timestamp:
            # New time bucket, reset count
            self.times[idx] = timestamp
            self.hits[idx] = 1
        else:
            # Same time bucket, increment count
            self.hits[idx] += 1
    
    def get_hits(self, timestamp):
        """Get hits in past 300 seconds."""
        total = 0
        for i in range(300):
            if timestamp - self.times[i] < 300:
                total += self.hits[i]
        return total

# Test both implementations
counter1 = HitCounter()
counter2 = HitCounterBucket()

test_operations = [
    ("hit", 1), ("hit", 2), ("hit", 3),
    ("get_hits", 4), ("hit", 300), ("get_hits", 300),
    ("get_hits", 301)
]

for op, timestamp in test_operations:
    if op == "hit":
        counter1.hit(timestamp)
        counter2.hit(timestamp)
        print(f"Hit at {timestamp}")
    else:
        result1 = counter1.get_hits(timestamp)
        result2 = counter2.get_hits(timestamp)
        print(f"get_hits({timestamp}): Queue={result1}, Bucket={result2}")
```

---

## Problem 6: Find Bottom Left Tree Value

**Difficulty:** Medium  
**Pattern:** Level Order Traversal  
**Time:** O(n) | **Space:** O(w)

### Problem Statement

Given the root of a binary tree, return the leftmost value in the last row of the tree.

### Solution

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def find_bottom_left_value(root):
    """
    Find bottom-left value using level-order traversal.
    
    The first node at the deepest level is the answer.
    """
    if not root:
        return None
    
    queue = deque([root])
    leftmost = root.val
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # First node of each level is leftmost
            if i == 0:
                leftmost = node.val
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return leftmost

# Alternative: Traverse right to left
def find_bottom_left_value_rtl(root):
    """
    Alternative: Process nodes right to left.
    Last processed node will be bottom-left.
    """
    if not root:
        return None
    
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        
        # Add right child first, then left
        if node.right:
            queue.append(node.right)
        if node.left:
            queue.append(node.left)
    
    return node.val  # Last processed node is bottom-left

# Create test tree:      2
#                       / \
#                      1   3
#                     /
#                    4
root = TreeNode(2)
root.left = TreeNode(1)
root.right = TreeNode(3)
root.left.left = TreeNode(4)

print(f"Bottom-left value (LTR): {find_bottom_left_value(root)}")
print(f"Bottom-left value (RTL): {find_bottom_left_value_rtl(root)}")
```

---

## 🎯 Problem-Solving Patterns

### 1. Monotonic Deque

- **Use when:** Need to maintain order while sliding window
- **Pattern:** Remove elements that can't be optimal
- **Examples:** Sliding window maximum, shortest subarray

### 2. Multi-source BFS

- **Use when:** Start from multiple points simultaneously  
- **Pattern:** Add all sources to queue initially
- **Examples:** Walls and gates, rotting oranges

### 3. State Space Search

- **Use when:** Each state leads to other states
- **Pattern:** BFS through possible states
- **Examples:** Open lock, word ladder

### 4. Level-by-Level Processing

- **Use when:** Need to process tree/graph level by level
- **Pattern:** Track level size in queue
- **Examples:** Tree level order, minimum depth

## 💡 Advanced Tips

!!! tip "Bidirectional BFS"
    When searching from start to end, try bidirectional BFS to reduce search space from O(b^d) to O(b^(d/2)).

!!! note "Space Optimization"  
    For problems with time windows, consider circular arrays instead of queues to optimize space.

!!! success "Monotonic Structures"
    Monotonic deques are powerful for sliding window problems where you need to maintain order.

## 🚀 Next Level

Ready for the ultimate challenge? Try:
- [Hard Queue Problems](hard-problems.md)
- Advanced graph algorithms using BFS
- System design with queues

---

*🎉 Excellent progress! You've conquered medium queue problems. Ready for [Hard Problems](hard-problems.md)?*
