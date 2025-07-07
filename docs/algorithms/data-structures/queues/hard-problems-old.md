# Queues: Hard Problems

## 🔥 Advanced Queue Challenges

These problems require sophisticated queue techniques, multiple data structures, and complex algorithms.

---

## Problem 1: Shortest Path in Binary Matrix

**Difficulty:** Hard  
**Pattern:** BFS with Obstacles  
**Time:** O(n²) | **Space:** O(n²)

### Problem Statement

Given an `n x n` binary matrix `grid`, return the length of the shortest clear path from top-left to bottom-right. If no path exists, return `-1`.

A clear path is from `(0,0)` to `(n-1,n-1)` such that all visited cells are 0, and you can move in 8 directions.

### Solution

```python
from collections import deque

def shortest_path_binary_matrix(grid):
    """
    Find shortest path in binary matrix using BFS.
    
    Can move in 8 directions (including diagonals).
    """
    n = len(grid)
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    if n == 1:
        return 1
    
    # 8 directions: up, down, left, right, and 4 diagonals
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    queue = deque([(0, 0, 1)])  # (row, col, path_length)
    visited = {(0, 0)}
    
    while queue:
        row, col, path_length = queue.popleft()
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds and if cell is valid
            if (0 <= new_row < n and 0 <= new_col < n and
                grid[new_row][new_col] == 0 and
                (new_row, new_col) not in visited):
                
                # Check if we reached the destination
                if new_row == n - 1 and new_col == n - 1:
                    return path_length + 1
                
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, path_length + 1))
    
    return -1

# Test with sample grids
test_grids = [
    [[0,0,0],[1,1,0],[1,1,0]],
    [[0,1],[1,0]],
    [[1,0,0],[1,1,0],[1,1,0]]
]

for i, grid in enumerate(test_grids):
    result = shortest_path_binary_matrix(grid)
    print(f"Grid {i+1}: {result}")
```

---

## Problem 2: Sliding Window Median

**Difficulty:** Hard  
**Pattern:** Two Heaps + Sliding Window  
**Time:** O(n log k) | **Space:** O(k)

### Problem Statement

Given an array `nums` and window size `k`, return the median of each sliding window.

### Solution

```python
import heapq
from collections import defaultdict

class SlidingWindowMedian:
    """
    Maintain median using two heaps with lazy deletion.
    
    Max heap (left): stores smaller half
    Min heap (right): stores larger half
    """
    
    def __init__(self):
        self.max_heap = []  # For smaller half (negated values)
        self.min_heap = []  # For larger half
        self.hash_map = defaultdict(int)  # For lazy deletion
        self.max_heap_size = 0
        self.min_heap_size = 0
    
    def median_sliding_window(self, nums, k):
        """Find median of each sliding window."""
        result = []
        
        for i in range(len(nums)):
            # Add current element
            self.add_number(nums[i])
            
            # Remove element going out of window
            if i >= k:
                self.remove_number(nums[i - k])
            
            # Calculate median when window is complete
            if i >= k - 1:
                result.append(self.get_median())
        
        return result
    
    def add_number(self, num):
        """Add number to appropriate heap."""
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
            self.max_heap_size += 1
        else:
            heapq.heappush(self.min_heap, num)
            self.min_heap_size += 1
        
        self.rebalance()
    
    def remove_number(self, num):
        """Mark number for lazy deletion."""
        self.hash_map[num] += 1
        
        if num <= -self.max_heap[0]:
            self.max_heap_size -= 1
        else:
            self.min_heap_size -= 1
        
        self.rebalance()
    
    def rebalance(self):
        """Maintain heap size balance."""
        # Max heap should have same size or one more element
        if self.max_heap_size > self.min_heap_size + 1:
            # Move from max_heap to min_heap
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
            self.max_heap_size -= 1
            self.min_heap_size += 1
        elif self.min_heap_size > self.max_heap_size:
            # Move from min_heap to max_heap
            val = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -val)
            self.min_heap_size -= 1
            self.max_heap_size += 1
        
        # Clean up lazy deletions
        self.prune_heap()
    
    def prune_heap(self):
        """Remove elements marked for deletion."""
        while self.max_heap and self.hash_map[-self.max_heap[0]] > 0:
            self.hash_map[-self.max_heap[0]] -= 1
            heapq.heappop(self.max_heap)
        
        while self.min_heap and self.hash_map[self.min_heap[0]] > 0:
            self.hash_map[self.min_heap[0]] -= 1
            heapq.heappop(self.min_heap)
    
    def get_median(self):
        """Get current median."""
        if self.max_heap_size == self.min_heap_size:
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0
        else:
            return float(-self.max_heap[0])

# Test
def median_sliding_window(nums, k):
    """Wrapper function."""
    swm = SlidingWindowMedian()
    return swm.median_sliding_window(nums, k)

nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
result = median_sliding_window(nums, k)
print(f"Sliding window medians: {result}")
```

---

## Problem 3: Serialize and Deserialize Binary Tree

**Difficulty:** Hard  
**Pattern:** BFS/DFS with Queue  
**Time:** O(n) | **Space:** O(n)

### Problem Statement

Design an algorithm to serialize and deserialize a binary tree. Serialization is converting a tree to a string, and deserialization is converting string back to tree.

### Solution

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Codec:
    """
    Serialize/deserialize binary tree using level-order traversal.
    """
    
    def serialize(self, root):
        """Encode tree to string using BFS."""
        if not root:
            return ""
        
        result = []
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            
            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append("null")
        
        return ",".join(result)
    
    def deserialize(self, data):
        """Decode string to tree using BFS."""
        if not data:
            return None
        
        values = data.split(",")
        root = TreeNode(int(values[0]))
        queue = deque([root])
        
        i = 1
        while queue and i < len(values):
            node = queue.popleft()
            
            # Process left child
            if values[i] != "null":
                node.left = TreeNode(int(values[i]))
                queue.append(node.left)
            i += 1
            
            # Process right child
            if i < len(values) and values[i] != "null":
                node.right = TreeNode(int(values[i]))
                queue.append(node.right)
            i += 1
        
        return root

# Alternative: DFS approach
class CodecDFS:
    """
    Serialize/deserialize using DFS (preorder traversal).
    """
    
    def serialize(self, root):
        """Serialize using preorder DFS."""
        def dfs(node):
            if not node:
                return "null"
            return str(node.val) + "," + dfs(node.left) + "," + dfs(node.right)
        
        return dfs(root)
    
    def deserialize(self, data):
        """Deserialize using preorder DFS."""
        def dfs():
            val = next(values)
            if val == "null":
                return None
            
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node
        
        values = iter(data.split(","))
        return dfs()

# Test both approaches
def test_codec(codec, root):
    """Test serialize/deserialize."""
    serialized = codec.serialize(root)
    print(f"Serialized: {serialized}")
    
    deserialized = codec.deserialize(serialized)
    reserialized = codec.serialize(deserialized)
    print(f"Reserialized: {reserialized}")
    
    return serialized == reserialized

# Create test tree:    1
#                     / \
#                    2   3
#                       / \
#                      4   5
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.right.left = TreeNode(4)
root.right.right = TreeNode(5)

print("BFS Approach:")
codec_bfs = Codec()
test_codec(codec_bfs, root)

print("\nDFS Approach:")
codec_dfs = CodecDFS()
test_codec(codec_dfs, root)
```

---

## Problem 4: Word Ladder II

**Difficulty:** Hard  
**Pattern:** BFS + Backtracking  
**Time:** O(n × m²) | **Space:** O(n × m)

### Problem Statement

Given two words `beginWord` and `endWord`, and a dictionary `wordList`, return all shortest transformation sequences from `beginWord` to `endWord`.

### Solution

```python
from collections import deque, defaultdict

def find_ladders(begin_word, end_word, word_list):
    """
    Find all shortest word ladders using BFS + backtracking.
    
    First use BFS to find shortest path length and build graph.
    Then use DFS to find all paths of that length.
    """
    if end_word not in word_list:
        return []
    
    word_set = set(word_list)
    word_set.add(begin_word)
    
    # BFS to find shortest path and build neighbor graph
    neighbors = defaultdict(list)
    queue = deque([begin_word])
    visited = {begin_word}
    found = False
    
    while queue and not found:
        # Process current level
        current_level = set()
        for _ in range(len(queue)):
            word = queue.popleft()
            
            # Try all possible transformations
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == word[i]:
                        continue
                    
                    new_word = word[:i] + c + word[i+1:]
                    
                    if new_word == end_word:
                        found = True
                        neighbors[word].append(new_word)
                    elif new_word in word_set and new_word not in visited:
                        if new_word not in current_level:
                            current_level.add(new_word)
                            queue.append(new_word)
                        neighbors[word].append(new_word)
        
        # Mark current level as visited
        visited.update(current_level)
    
    # DFS to find all paths
    def dfs(word, path, result):
        if word == end_word:
            result.append(path[:])
            return
        
        for neighbor in neighbors[word]:
            path.append(neighbor)
            dfs(neighbor, path, result)
            path.pop()
    
    result = []
    dfs(begin_word, [begin_word], result)
    return result

# Optimized version with bidirectional BFS
def find_ladders_bidirectional(begin_word, end_word, word_list):
    """
    Bidirectional BFS for better performance.
    """
    if end_word not in word_list:
        return []
    
    word_set = set(word_list)
    
    # Build neighbor graph
    neighbors = defaultdict(list)
    
    def build_neighbors():
        for word in word_set:
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                neighbors[pattern].append(word)
    
    build_neighbors()
    
    def get_neighbors(word):
        result = []
        for i in range(len(word)):
            pattern = word[:i] + '*' + word[i+1:]
            for neighbor in neighbors[pattern]:
                if neighbor != word:
                    result.append(neighbor)
        return result
    
    # BFS from both ends
    begin_set = {begin_word}
    end_set = {end_word}
    visited = set()
    parent_map = defaultdict(list)
    found = False
    
    while begin_set and end_set and not found:
        if len(begin_set) > len(end_set):
            begin_set, end_set = end_set, begin_set
        
        visited.update(begin_set)
        next_set = set()
        
        for word in begin_set:
            for neighbor in get_neighbors(word):
                if neighbor in end_set:
                    found = True
                    parent_map[neighbor].append(word)
                elif neighbor not in visited:
                    next_set.add(neighbor)
                    parent_map[neighbor].append(word)
        
        begin_set = next_set
    
    # Build result paths
    def build_path(word, path, result):
        if word == begin_word:
            result.append([begin_word] + path[::-1])
            return
        
        for parent in parent_map[word]:
            path.append(parent)
            build_path(parent, path, result)
            path.pop()
    
    result = []
    if found:
        build_path(end_word, [end_word], result)
    
    return result

# Test
begin_word = "hit"
end_word = "cog"
word_list = ["hot", "dot", "dog", "lot", "log", "cog"]

print("Regular BFS:")
result1 = find_ladders(begin_word, end_word, word_list)
for path in result1:
    print(path)

print("\nBidirectional BFS:")
result2 = find_ladders_bidirectional(begin_word, end_word, word_list)
for path in result2:
    print(path)
```

---

## Problem 5: Shortest Bridge

**Difficulty:** Hard  
**Pattern:** DFS + BFS  
**Time:** O(n²) | **Space:** O(n²)

### Problem Statement

Given a binary matrix with exactly two islands (connected 1s), return the shortest bridge connecting them.

### Solution

```python
from collections import deque

def shortest_bridge(grid):
    """
    Find shortest bridge between two islands.
    
    Step 1: DFS to find first island and mark it
    Step 2: BFS from first island to find second island
    """
    n = len(grid)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def dfs(i, j, island_cells):
        """Mark all cells of first island."""
        if (i < 0 or i >= n or j < 0 or j >= n or
            grid[i][j] != 1):
            return
        
        grid[i][j] = 2  # Mark as visited
        island_cells.append((i, j))
        
        for di, dj in directions:
            dfs(i + di, j + dj, island_cells)
    
    # Find first island using DFS
    first_island = []
    found = False
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                dfs(i, j, first_island)
                found = True
                break
        if found:
            break
    
    # BFS from first island to find second island
    queue = deque([(i, j, 0) for i, j in first_island])
    visited = set(first_island)
    
    while queue:
        i, j, dist = queue.popleft()
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            
            if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in visited:
                if grid[ni][nj] == 1:  # Found second island
                    return dist
                elif grid[ni][nj] == 0:  # Water, continue BFS
                    visited.add((ni, nj))
                    queue.append((ni, nj, dist + 1))
    
    return -1  # Should not reach here

# Test with sample grid
grid = [
    [0, 1],
    [1, 0]
]

result = shortest_bridge(grid)
print(f"Shortest bridge length: {result}")

# Another test case
grid2 = [
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 1]
]

# Note: Need to reset grid2 values since we modify the grid
grid2_copy = [row[:] for row in grid2]
result2 = shortest_bridge(grid2_copy)
print(f"Shortest bridge length (grid2): {result2}")
```

---

## Problem 6: Cut Off Trees for Golf Event

**Difficulty:** Hard  
**Pattern:** Multiple BFS  
**Time:** O(m²n²) | **Space:** O(mn)

### Problem Statement

You are asked to cut off all trees in a forest for a golf event. Trees are represented by positive integers, and you need to cut them in ascending order of height. Return the minimum steps to cut all trees, or -1 if impossible.

### Solution

```python
from collections import deque
import heapq

def cut_off_tree(forest):
    """
    Cut trees in ascending height order using multiple BFS.
    
    For each tree, use BFS to find shortest path from current position.
    """
    if not forest or not forest[0]:
        return -1
    
    rows, cols = len(forest), len(forest[0])
    
    # Collect all trees with their positions
    trees = []
    for i in range(rows):
        for j in range(cols):
            if forest[i][j] > 1:
                trees.append((forest[i][j], i, j))
    
    # Sort trees by height
    trees.sort()
    
    def bfs(start_r, start_c, end_r, end_c):
        """Find shortest path between two points."""
        if start_r == end_r and start_c == end_c:
            return 0
        
        queue = deque([(start_r, start_c, 0)])
        visited = set([(start_r, start_c)])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            r, c, dist = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and
                    (nr, nc) not in visited and forest[nr][nc] != 0):
                    
                    if nr == end_r and nc == end_c:
                        return dist + 1
                    
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
        
        return -1  # Path not found
    
    # Start from (0, 0)
    current_r, current_c = 0, 0
    total_steps = 0
    
    # Visit each tree in order
    for height, tree_r, tree_c in trees:
        steps = bfs(current_r, current_c, tree_r, tree_c)
        
        if steps == -1:
            return -1  # Cannot reach this tree
        
        total_steps += steps
        current_r, current_c = tree_r, tree_c
    
    return total_steps

# Optimized version using A* search
def cut_off_tree_astar(forest):
    """
    Use A* search for better performance.
    """
    if not forest or not forest[0]:
        return -1
    
    rows, cols = len(forest), len(forest[0])
    
    # Collect and sort trees
    trees = []
    for i in range(rows):
        for j in range(cols):
            if forest[i][j] > 1:
                trees.append((forest[i][j], i, j))
    
    trees.sort()
    
    def manhattan_distance(r1, c1, r2, c2):
        """Heuristic function for A*."""
        return abs(r1 - r2) + abs(c1 - c2)
    
    def astar(start_r, start_c, end_r, end_c):
        """A* search for shortest path."""
        if start_r == end_r and start_c == end_c:
            return 0
        
        heap = [(0, 0, start_r, start_c)]  # (f_score, g_score, r, c)
        visited = set()
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while heap:
            f_score, g_score, r, c = heapq.heappop(heap)
            
            if (r, c) in visited:
                continue
            
            visited.add((r, c))
            
            if r == end_r and c == end_c:
                return g_score
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and
                    (nr, nc) not in visited and forest[nr][nc] != 0):
                    
                    new_g = g_score + 1
                    new_h = manhattan_distance(nr, nc, end_r, end_c)
                    new_f = new_g + new_h
                    
                    heapq.heappush(heap, (new_f, new_g, nr, nc))
        
        return -1
    
    current_r, current_c = 0, 0
    total_steps = 0
    
    for height, tree_r, tree_c in trees:
        steps = astar(current_r, current_c, tree_r, tree_c)
        
        if steps == -1:
            return -1
        
        total_steps += steps
        current_r, current_c = tree_r, tree_c
    
    return total_steps

# Test
forest = [
    [1, 2, 3],
    [0, 0, 4],
    [7, 6, 5]
]

print(f"BFS result: {cut_off_tree(forest)}")

# Reset forest for A* test
forest_copy = [row[:] for row in forest]
print(f"A* result: {cut_off_tree_astar(forest_copy)}")
```

---

## 🎯 Advanced Problem-Solving Patterns

### 1. Multi-Source BFS

- **Use when:** Need to start from multiple points simultaneously
- **Pattern:** Initialize queue with all sources
- **Examples:** Walls and gates, rotting oranges

### 2. Bidirectional BFS

- **Use when:** Searching from start to end point
- **Pattern:** Search from both ends, meet in middle
- **Examples:** Word ladder, shortest path

### 3. BFS with State Compression

- **Use when:** Need to track multiple states
- **Pattern:** Encode state in queue elements
- **Examples:** Sliding puzzle, shortest path with keys

### 4. Lazy Deletion

- **Use when:** Can't efficiently remove elements
- **Pattern:** Mark for deletion, clean up later
- **Examples:** Sliding window median, data stream

## 💡 Expert Tips

!!! tip "Space Optimization"
    For problems with large state spaces, consider state compression or iterative deepening.

!!! warning "Time Complexity"
    BFS can be exponential in worst case. Use pruning, memoization, or bidirectional search.

!!! success "Hybrid Approaches"
    Combine DFS and BFS for complex problems. Use DFS for exploration, BFS for shortest paths.

## 🚀 Mastery Challenges

Congratulations on completing hard queue problems! Next challenges:

1. **Advanced Graph Algorithms:** Dijkstra, A*, Flow networks
2. **Competitive Programming:** TopCoder, Codeforces queue problems
3. **System Design:** Implement message queues, task schedulers

---

*🏆 Outstanding! You've mastered the most challenging queue problems. You're now ready for advanced algorithms and system design!*
