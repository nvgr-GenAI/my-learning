# Problem-Solving Patterns 🧩

This guide covers essential algorithmic patterns that serve as powerful tools for solving a wide range of coding problems. Understanding these patterns will help you recognize solution approaches faster and improve your problem-solving skills.

## Two Pointers Technique

=== "Overview"
    **Use Cases**: Array/string problems, particularly when working with sorted data or searching for pairs with constraints.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | O(n)       | Linear time as we typically traverse the array only once |
        | Space  | O(1)       | Constant extra space, only using pointers |

    ### What is the Two Pointers Technique?

    The Two Pointers technique uses two references (pointers) to traverse a data structure, often an array. It's an elegant way to solve problems that would otherwise require nested loops (O(n²)) in just a single pass (O(n)). This technique is especially powerful when working with sorted arrays or when searching for pairs that satisfy certain conditions.

    Think of it like using your left and right hand simultaneously to search through a bookshelf, rather than checking one book at a time.

    ### Approach

    1. Initialize two pointers (typically at the beginning, or at opposite ends)
    2. Move the pointers based on certain conditions
    3. Find the answer during pointer movement

    ### Common Two Pointer Patterns

    1. **Opposite Ends**: Start from both ends, move inward. Great for sorted arrays when finding pairs with sum constraints.
    
    2. **Fast & Slow**: One pointer moves faster than the other. Useful for detecting cycles in linked lists or finding middle elements.
    
    3. **Forward Pointers**: Both move forward, but at different paces. Ideal for removing duplicates or maintaining a window of elements.
    
    4. **Sliding Window**: Special case where pointers define window boundaries. Perfect for finding subarrays with specific properties.

=== "Example"
    ### Two Sum (Sorted Array)

    ```
    Given a sorted array of integers and a target sum, find two numbers that add up to the target.
    ```

    #### Implementation

    ```python
    def two_sum_sorted(arr, target):
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

=== "Practice Problems"
    - Container With Most Water
    - 3Sum
    - Remove Duplicates from Sorted Array
    - Valid Palindrome
    - Trapping Rain Water

## Sliding Window Pattern

=== "Overview"
    **Use Cases**: Problems involving subarrays/substrings, especially when looking for a contiguous sequence with certain properties.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | O(n)       | Linear time as we typically traverse the array only once |
        | Space  | O(1) or O(k) | Constant space in most cases, but may require O(k) space to store window elements in some problems |

    ### What is the Sliding Window Pattern?

    The Sliding Window pattern is a technique for efficiently processing arrays or strings by maintaining a "window" of elements that slides through the data structure. Instead of recomputing everything from scratch as the window moves, we update just the elements that enter and exit the window, making it much more efficient for processing continuous sequences.

    Imagine looking through a physical window that moves along a street - you only see what's currently in the frame, and as you slide the window, new elements come into view while others disappear.

    ### Approach

    1. Define a window with start and end pointers
    2. Expand/contract window according to problem constraints
    3. Update result as window changes
    4. Avoid recalculation by efficiently updating window state

    ### Common Sliding Window Patterns

    1. **Fixed Size**: Window size remains constant throughout traversal. Useful for problems like "find maximum sum subarray of size k" or "calculate moving average."
    
    2. **Variable Size**: Grow/shrink window based on conditions. Great for problems like "smallest subarray with sum >= target" or "longest substring without repeating characters."
    
    3. **Dynamic Constraint**: Window must satisfy certain property throughout. Used in problems like "longest substring with at most k distinct characters."

=== "Example"
    ### Maximum Sum Subarray of Size K

    ```python
    # Find the maximum sum of any contiguous subarray of size k.
    def max_sum_subarray(arr, k):
        if len(arr) < k:
            return None
        
        # Calculate sum of first window
        window_sum = sum(arr[:k])
        max_sum = window_sum
        
        # Slide the window
        for i in range(k, len(arr)):
            window_sum = window_sum - arr[i - k] + arr[i]
            max_sum = max(max_sum, window_sum)
        
        return max_sum
    ```

=== "Practice Problems"
    - Longest Substring Without Repeating Characters
    - Minimum Window Substring
    - Longest Repeating Character Replacement
    - Find All Anagrams in a String
    - Permutation in String

## Fast and Slow Pointers (Floyd's Algorithm)

=== "Overview"
    **Use Cases**: Linked list problems, cycle detection, finding middle elements.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | O(n)       | Linear time as we typically traverse the list only once |
        | Space  | O(1)       | Constant extra space, only using two pointers regardless of input size |

    ### What are Fast and Slow Pointers?

    The Fast and Slow Pointers technique (also known as Floyd's Tortoise and Hare algorithm) uses two pointers that move through a data structure at different speeds. This elegant approach solves various problems, particularly in linked lists, without requiring additional space.

    It's like having a tortoise and a hare racing on a track - if the track has a loop, the faster hare will eventually catch up to the tortoise from behind. If there's no loop, the hare will reach the end first.

    ### Approach

    1. Initialize two pointers at the head of the list
    2. Move slow pointer one step and fast pointer two steps per iteration
    3. If there's a cycle, the pointers will eventually meet
    4. If there's no cycle, the fast pointer will reach the end

    ### Common Fast & Slow Patterns

    1. **Cycle Detection**: Two pointers meet if there's a cycle. The classic application is detecting cycles in linked lists - if the pointers meet, there must be a cycle.
    
    2. **Middle Finding**: Stop when fast reaches the end. Since the fast pointer moves twice as quickly, when it reaches the end, the slow pointer will be at the middle. Perfect for finding the median element or splitting a list in half.
    
    3. **Nth From End**: Use a gap of N between pointers. By keeping the pointers exactly N nodes apart, when the fast pointer reaches the end, the slow pointer will be at the Nth node from the end.

=== "Example"
    ### Detect Cycle in a Linked List

    ```python
    # Given a linked list, determine if it has a cycle
    def has_cycle(head):
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

=== "Practice Problems"
    - Linked List Cycle II (find start of cycle)
    - Middle of the Linked List
    - Palindrome Linked List
    - Reorder List
    - Find the Duplicate Number

## Binary Search

=== "Overview"
    **Use Cases**: Searching in sorted arrays, optimization problems where search space can be divided.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | O(log n)   | Logarithmic time as we divide the search space in half with each step |
        | Space  | O(1)       | Constant space for iterative implementation (recursive implementations use O(log n) stack space) |

    ### What is Binary Search?

    Binary Search is a powerful divide-and-conquer algorithm that rapidly locates items in a sorted collection by repeatedly dividing the search space in half. Instead of checking every element linearly (O(n)), binary search achieves O(log n) efficiency by eliminating half of the remaining elements with each comparison.

    It's like finding a word in a dictionary - you don't check every page sequentially; you open to the middle, see if your target comes before or after, and repeat the process with the relevant half.

    ### Approach

    1. Define the search space (left and right boundaries)
    2. Find the middle element and compare with target
    3. Eliminate half of the search space based on the comparison
    4. Repeat until target is found or space is exhausted

    ### Common Binary Search Patterns

    1. **Classic Search**: Standard binary search in a sorted array to find an exact match. The simplest form where you're looking for a specific value.
    
    2. **Boundary Finding**: Find leftmost/rightmost occurrence of a value. Useful when duplicates exist and you need to find the first or last occurrence.
    
    3. **Search on Answer**: Use binary search to find optimal value in a range. Instead of searching for a value, you're searching for an answer that satisfies certain criteria. For example, "What's the minimum size needed to fit all items?"
    
    4. **Rotated Array**: Binary search with pivot consideration. Used when a sorted array has been rotated (shifted) and you need to account for the pivot point.

=== "Example"
    ### Classic Binary Search

    ```python
    # Find if a target value exists in a sorted array
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        
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

=== "Practice Problems"
    - Search in Rotated Sorted Array
    - Find First and Last Position of Element in Sorted Array
    - Peak Element
    - Search a 2D Matrix
    - Koko Eating Bananas

## Breadth-First Search (BFS)

=== "Overview"
    **Use Cases**: Graph/tree traversal, shortest path in unweighted graphs, level-order problems.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | O(V+E)     | Where V is the number of vertices/nodes and E is the number of edges - we visit each node once and explore all edges |
        | Space  | O(V)       | Queue may need to store all vertices in the worst case (e.g., in a complete graph) |

    ### What is Breadth-First Search?

    Breadth-First Search (BFS) is a traversal algorithm that explores all nodes at the present depth level before moving on to nodes at the next depth level. BFS spreads like a ripple in water, expanding outward evenly from the starting point.

    It's like exploring a maze by checking all possible paths one step away from you, then all paths two steps away, and so on, rather than following a single path as far as possible.

    ### Approach

    1. Use a queue to keep track of nodes to visit
    2. Start with initial node(s) in the queue
    3. Process current node, enqueue all its unvisited neighbors
    4. Mark nodes as visited to avoid cycles
    5. Continue until the queue is empty

    ### Common BFS Patterns

    1. **Level Order**: Process nodes level by level. Perfect for problems requiring exploration of trees in level order, like level-order traversal of a binary tree.
    
    2. **Shortest Path**: Track distance from source in unweighted graphs. Since BFS explores nodes by their distance from the start, the first time you reach a node is guaranteed to be along the shortest path.
    
    3. **Graph Connectivity**: Find all connected components by running BFS from unvisited nodes. Useful for determining which nodes can be reached from a given starting point.
    
    4. **Topological Sorting**: When combined with in-degree tracking. By keeping track of in-degrees and processing nodes with zero in-degree, BFS can determine a valid ordering for directed acyclic graphs.

=== "Example"
    ### Level Order Traversal of Binary Tree

    ```python
    # Given a binary tree, return level-by-level traversal (breadth-first)
    from collections import deque
    
    def level_order(root):
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level)
        
        return result
    ```

=== "Practice Problems"
    - Binary Tree Level Order Traversal
    - Minimum Depth of Binary Tree
    - Word Ladder
    - Rotting Oranges
    - Course Schedule

## Depth-First Search (DFS)

=== "Overview"
    **Use Cases**: Graph/tree traversal, path finding, backtracking, cycle detection.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | O(V+E)     | Where V is the number of vertices/nodes and E is the number of edges - we visit each node once and explore all edges |
        | Space  | O(H)       | H is the maximum depth of recursion (height of the call stack), which could be as large as V in the worst case |

    ### What is Depth-First Search?

    Depth-First Search (DFS) is a traversal algorithm that explores as far as possible along each branch before backtracking. DFS dives deep into the structure first, rather than exploring breadth-wise.

    It's like exploring a maze by following a single path until you hit a dead end, then backtracking to the last junction and trying a different path. This approach is naturally recursive and often elegant to implement.

    ### Approach

    1. Start at an initial node
    2. Mark the current node as visited
    3. Recursively explore each unvisited neighbor before backtracking
    4. Use a stack (implicit in recursion) to keep track of nodes
    5. Mark nodes as visited to avoid cycles

    ### Common DFS Patterns

    1. **Preorder/Inorder/Postorder**: Tree traversal variations based on when a node is processed relative to its children. Preorder (process node, then children), Inorder (left child, node, right child), Postorder (children first, then node).
    
    2. **Backtracking**: Explore all paths with constraints, abandoning paths that violate constraints. Perfect for problems like generating all valid combinations or permutations.
    
    3. **Cycle Detection**: Track visited nodes in the current path to identify cycles. A cycle exists if you encounter a node that's already in the current path.
    
    4. **Connected Components**: Find all nodes in a component by exploring from a single node. DFS naturally maps out the entire connected region from a starting point.

=== "Example"
    ### Path Sum in Binary Tree

    ```python
    # Given a binary tree and a sum, determine if the tree has a root-to-leaf path that adds up to the sum
    def has_path_sum(root, target_sum):
        if not root:
            return False
        
        # If it's a leaf node, check if the value equals the remaining sum
        if not root.left and not root.right:
            return root.val == target_sum
        
        # Recursively check the left and right subtrees
        left = has_path_sum(root.left, target_sum - root.val)
        right = has_path_sum(root.right, target_sum - root.val)
        
        return left or right
    ```

=== "Practice Problems"
    - Binary Tree Paths
    - Number of Islands
    - Max Area of Island
    - Pacific Atlantic Water Flow
    - Clone Graph

## Hash Map Pattern

=== "Overview"
    **Use Cases**: Problems requiring quick lookups, frequency counting, caching previously seen values.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | O(n)       | Linear time to process each element once, with O(1) lookup/insert operations |
        | Space  | O(n)       | Additional space needed to store up to n key-value pairs in the hash map |

    ### What is the Hash Map Pattern?

    The Hash Map pattern leverages hash tables (dictionaries in Python, objects in JavaScript) to achieve constant-time O(1) lookups, dramatically improving efficiency in many algorithms. This pattern trades memory for speed by storing information that can be quickly retrieved later.

    It's like creating an index for a book - instead of scanning every page to find a topic, you look it up directly in the index to find the exact page number.

    ### Approach

    1. Use a hash map to store key-value pairs for quick access
    2. Keys could be elements, patterns, or computed values from the input
    3. Values could be frequencies, positions, or aggregated results
    4. Check for existence or retrieve values in O(1) time instead of searching
    5. Process data in a single pass while referencing previously seen elements

    ### Common Hash Map Patterns

    1. **Value-to-Index Mapping**: Store positions for quick lookup. Used in problems like Two Sum where you need to find pairs with a specific relationship without checking every possible pair.
    
    2. **Frequency Counter**: Count occurrences of elements. Excellent for problems involving anagrams, character counting, or finding elements that appear more/less than a certain number of times.
    
    3. **Character/Pattern Mapping**: Map between characters/patterns. Useful for problems involving character substitution, pattern matching, or isomorphic strings.
    
    4. **Prefix Sum Storage**: Store cumulative sums for range queries. By storing running sums, you can efficiently calculate sums of any range without recomputing.

=== "Example"
    ### Two Sum (Unsorted Array)

    ```python
    # Find two numbers in an array that add up to a specific target
    def two_sum(nums, target):
        # Map of value -> index
        seen = {}
        
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        
        return []
    ```

=== "Practice Problems"
    - Group Anagrams
    - Valid Anagram
    - Longest Consecutive Sequence
    - Subarray Sum Equals K
    - LRU Cache

## Dynamic Programming Patterns

=== "Overview"
    **Use Cases**: Optimization problems with overlapping subproblems and optimal substructure.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | Varies     | Typically O(n²) or O(n×m) where n and m are input dimensions - we process each state once |
        | Space  | Varies     | Typically O(n) or O(n×m) to store the memoization table or DP array |

    ### What is Dynamic Programming?

    Dynamic Programming (DP) is a powerful technique for solving complex problems by breaking them down into simpler overlapping subproblems and avoiding redundant calculations. DP combines the elegance of recursion with the efficiency of caching to solve problems that would otherwise be exponentially complex.

    It's like filling out a crossword puzzle - you solve each square once, and then use that information when solving related squares, rather than re-evaluating each possibility from scratch.

    ### Key Principles of Dynamic Programming

    1. **Overlapping Subproblems**: The same subproblems are solved multiple times
    2. **Optimal Substructure**: An optimal solution contains optimal solutions to subproblems
    3. **State Definition**: Clearly define the meaning of each state in your DP table
    4. **State Transition**: Define how to move from one state to another
    5. **Base Cases**: Identify the simplest cases where answers are known

    ### Common DP Patterns

    1. **1D Tabulation**: Fill a 1D array bottom-up. Used when the state depends on previous values in a linear fashion, like in the Fibonacci sequence, climbing stairs, or house robber problems.
    
    2. **2D Tabulation**: Fill a 2D matrix bottom-up. Applied when the state depends on two factors, such as in longest common subsequence, edit distance, or grid-based problems.
    
    3. **State Compression**: Reduce space complexity by keeping only relevant states. Often, you only need the last few states to compute the next one, allowing you to use O(1) or O(k) space instead of O(n).
    
    4. **Memoization**: Top-down approach with cached results. Start with a recursive solution and add caching to avoid redundant calculations. This approach is often more intuitive but may have more overhead.

=== "Example"
    ### Fibonacci with Memoization

    ```python
    # Compute the nth Fibonacci number efficiently
    def fibonacci(n, memo={}):
        if n in memo:
            return memo[n]
        
        if n <= 1:
            return n
        
        memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
        return memo[n]
    ```

=== "Practice Problems"
    - Climbing Stairs
    - Coin Change
    - Longest Increasing Subsequence
    - Edit Distance
    - Maximum Subarray

## Greedy Algorithm Pattern

=== "Overview"
    **Use Cases**: Problems where making locally optimal choices leads to a global optimum.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | O(n) or O(n log n) | Linear in simple cases, may involve sorting (n log n) in others |
        | Space  | O(1) or O(n) | Often constant extra space, but may require O(n) for storing intermediate results |

    ### What is the Greedy Algorithm Pattern?

    The Greedy Algorithm pattern involves making the locally optimal choice at each step with the hope of finding a global optimum. Greedy algorithms are often straightforward and efficient, but they don't always guarantee the best solution for every problem.

    It's like hiking up a mountain by always taking the steepest path upward - sometimes this leads to the summit (global optimum), but sometimes it leads to a local peak that isn't the highest point.

    ### Approach

    1. Make the best choice at each step based on current information
    2. Never reconsider or backtrack on previous choices
    3. Often requires sorting or a priority queue to determine the "best" choice
    4. Prove that locally optimal choices lead to a globally optimal solution

    ### When to Use Greedy Algorithms

    Greedy algorithms work well when:
    - The problem has "optimal substructure" (optimal solution contains optimal solutions to subproblems)
    - A locally optimal choice leads to a globally optimal solution
    - The choice made at each step is independent of previous choices

    ### Common Greedy Patterns

    1. **Activity Selection**: Choose activities that finish earliest. Used in problems like meeting room scheduling where you want to maximize the number of activities completed.
    
    2. **Huffman Coding**: Build optimal prefix-free codes by combining least frequent symbols first. Creates variable-length codes where more frequent symbols have shorter codes.
    
    3. **Interval Scheduling**: Select non-overlapping intervals to maximize coverage or count. Used in problems involving time intervals like event scheduling.
    
    4. **Fractional Knapsack**: Take items with highest value/weight ratio. Unlike 0/1 knapsack, items can be divided, so always take the best value-to-weight ratio first.

=== "Example"
    ### Jump Game

    ```python
    # Given an array where each element represents max jump length, determine if you can reach the end
    def can_jump(nums):
        max_reach = 0
        
        for i, jump in enumerate(nums):
            # If we can't reach the current position, return False
            if max_reach < i:
                return False
            
            # Update the farthest we can reach
            max_reach = max(max_reach, i + jump)
            
            # If we can already reach the end, return True
            if max_reach >= len(nums) - 1:
                return True
        
        return True
    ```

=== "Practice Problems"
    - Jump Game II
    - Gas Station
    - Task Scheduler
    - Minimum Number of Arrows to Burst Balloons
    - Non-overlapping Intervals

## Union Find (Disjoint Set)

=== "Overview"
    **Use Cases**: Problems involving connected components, cycle detection in undirected graphs.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | Amortized O(α(n)) | Nearly constant time per operation with path compression and union by rank/size (α is the inverse Ackermann function which grows extremely slowly) |
        | Space  | O(n)       | Requires storage for parent pointers and rank/size information for n elements |

    ### What is Union Find (Disjoint Set)?

    Union Find (also known as Disjoint Set) is a data structure that tracks elements split into one or more disjoint sets. It provides near-constant time operations to add new sets, merge existing sets, and determine whether elements belong to the same set.

    It's like managing groups of people where you need to quickly determine if two people are in the same group and efficiently merge groups together.

    ### Key Operations

    1. **Find**: Determine which set an element belongs to (identify the representative)
    2. **Union**: Merge two sets together
    3. **MakeSet**: Create a new set with a single element

    ### Approach

    1. Initialize each element as its own set (each element points to itself as parent)
    2. Implement the find operation to find the set representative (root)
    3. Implement the union operation to merge two sets by connecting their roots
    4. Use path compression and union by rank for efficiency:
       - Path compression: When finding an element's representative, update all nodes in the path to point directly to the root
       - Union by rank: Attach the smaller tree to the root of the larger tree to minimize tree height

    ### Applications

    - Finding connected components in a graph
    - Detecting cycles in undirected graphs
    - Building minimum spanning trees (Kruskal's algorithm)
    - Network connectivity problems
    - Image processing for determining connected regions

=== "Example"
    ### Number of Connected Components

    ```python
    # Given n nodes and edges between them, find the number of connected components
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.count = n  # Number of components
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])  # Path compression
            return self.parent[x]
        
        def union(self, x, y):
            root_x = self.find(x)
            root_y = self.find(y)
            
            if root_x == root_y:
                return
            
            # Union by rank
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            
            self.count -= 1
    
    def count_components(n, edges):
        uf = UnionFind(n)
        
        for x, y in edges:
            uf.union(x, y)
        
        return uf.count
    ```

=== "Practice Problems"
    - Redundant Connection
    - Number of Provinces
    - Accounts Merge
    - Graph Valid Tree
    - Satisfiability of Equality Equations

### Practice Problems

- Redundant Connection
- Number of Provinces
- Accounts Merge
- Graph Valid Tree
- Satisfiability of Equality Equations

## Trie (Prefix Tree)

=== "Overview"
    **Use Cases**: String problems involving prefix matching, word dictionaries.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | O(m)       | Where m is the length of the key (word) - constant time per character for operations like insert, search, and prefix matching |
        | Space  | O(n×m)     | Where n is the number of keys and m is average key length - potentially large for storing many words with limited common prefixes |

    ### What is a Trie (Prefix Tree)?

    A Trie, pronounced "try" or "tree," is a specialized tree-like data structure used for efficient retrieval of keys in a dataset of strings. Unlike a binary search tree, a trie's nodes are not associated with actual values but rather with positions in a string.

    It's like an organized filing system where each path from the root to a marked node represents a valid word, and common prefixes are shared nodes to save space.

    ### Key Characteristics

    1. **Prefix Sharing**: Common prefixes are shared among multiple strings
    2. **Character-by-Character Navigation**: Each node represents a single character in a sequence
    3. **Path Meaning**: A path from root to leaf represents a complete string
    4. **Efficient Prefix Operations**: Checking if a string is a prefix takes O(m) time where m is prefix length
    
    ### Common Operations

    1. **Insert**: Add a new string to the trie
    2. **Search**: Check if a complete string exists in the trie
    3. **StartsWith**: Check if any string in the trie starts with a given prefix
    4. **Delete**: Remove a string from the trie

    ### Applications

    - Autocomplete and predictive text
    - Spell checkers
    - IP routing (longest prefix matching)
    - String searching in a text
    - Implementing dictionaries with prefix operations

=== "Example"
    ### Implement a Trie

    ```python
    # Implement a trie with insert, search, and startsWith methods
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False
    
    class Trie:
        def __init__(self):
            self.root = TrieNode()
        
        def insert(self, word):
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True
        
        def search(self, word):
            node = self.root
            for char in word:
                if char not in node.children:
                    return False
                node = node.children[char]
            return node.is_end_of_word
        
        def starts_with(self, prefix):
            node = self.root
            for char in prefix:
                if char not in node.children:
                    return False
                node = node.children[char]
            return True
    ```

=== "Practice Problems"
    - Word Search II
    - Design Add and Search Words Data Structure
    - Replace Words
    - Implement Magic Dictionary
    - Word Squares

## Topological Sort

=== "Overview"
    **Use Cases**: Scheduling problems, finding dependency order, detecting cycles in directed graphs.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | O(V+E)     | Where V is the number of vertices and E is the number of edges - we process each vertex and edge once |
        | Space  | O(V)       | Storage required for the visited array, recursion stack, and result array |

    ### What is Topological Sort?

    Topological Sort is an algorithm for ordering the vertices of a directed graph such that for every directed edge (u,v), vertex u comes before v in the ordering. In other words, it orders nodes in a way that respects all dependencies.

    It's like scheduling a sequence of courses where some courses have prerequisites - you need to find an order to take all courses while satisfying all prerequisites.

    ### Key Properties

    1. **Applicability**: Only works on Directed Acyclic Graphs (DAGs)
    2. **Cycle Detection**: Cannot topologically sort a graph with a cycle
    3. **Multiple Solutions**: A graph may have multiple valid topological orderings
    4. **Source Nodes**: Always begins with nodes that have no incoming edges

    ### Implementation Approaches

    1. **Kahn's Algorithm**: Iteratively remove nodes with no incoming edges
       - Track in-degree (incoming edges) for each node
       - Start with nodes having zero in-degree
       - Remove these nodes and their outgoing edges, updating in-degrees
       - Repeat until all nodes are processed or a cycle is detected

    2. **DFS-based**: Use depth-first search with a finish time ordering
       - Perform DFS traversal
       - Add nodes to the result list in post-order (after processing all neighbors)
       - Reverse the final list

    ### Applications

    - Course scheduling
    - Build systems and dependency resolution
    - Task scheduling with dependencies
    - Data processing pipelines
    - Symbol resolution in programming language compilers

=== "Example"
    ### Course Schedule

    ```python
    # Given prerequisites for courses, determine if it's possible to finish all courses
    from collections import defaultdict, deque
    
    def can_finish(num_courses, prerequisites):
        # Build adjacency list and in-degree count
        graph = defaultdict(list)
        in_degree = [0] * num_courses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # Add all courses with no prerequisites to the queue
        queue = deque()
        for i in range(num_courses):
            if in_degree[i] == 0:
                queue.append(i)
        
        # Process courses in topological order
        count = 0
        while queue:
            course = queue.popleft()
            count += 1
            
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        return count == num_courses
    ```

=== "Practice Problems"
    - Course Schedule II
    - Alien Dictionary
    - Sequence Reconstruction
    - Parallel Courses
    - Minimum Height Trees

## Bit Manipulation

=== "Overview"
    **Use Cases**: Problems involving binary operations, optimization of space.

    !!! info "Complexity Analysis"
        | Aspect | Complexity | Notes |
        |--------|------------|-------|
        | Time   | O(1) or O(n) | Basic bit operations are O(1), while operations on all bits in a number are O(log n), and operations on arrays of bits are O(n) |
        | Space  | O(1)       | Most bit manipulation uses constant extra space regardless of input size |

    ### What is Bit Manipulation?

    Bit manipulation involves directly manipulating individual bits within binary representations of data. These techniques provide extremely efficient operations for certain problems and can lead to elegant solutions that are both faster and more memory-efficient than conventional approaches.

    It's like working with the building blocks of computing itself - leveraging the fundamental binary nature of data to perform operations at the most basic level.

    ### Why Use Bit Manipulation?

    1. **Efficiency**: Bit operations are extremely fast at the hardware level
    2. **Space Optimization**: Represent multiple boolean values in a single integer
    3. **Elegant Solutions**: Some problems become much simpler with bit operations
    4. **Low-level Operations**: Essential for systems programming and optimizations

    ### Common Bit Operations

    1. **Check bit**: `(num >> i) & 1` - Determines if the i-th bit is set (1) or not (0)
    
    2. **Set bit**: `num | (1 << i)` - Sets the i-th bit to 1, leaving others unchanged
    
    3. **Clear bit**: `num & ~(1 << i)` - Sets the i-th bit to 0, leaving others unchanged
    
    4. **Toggle bit**: `num ^ (1 << i)` - Flips the i-th bit (0→1 or 1→0)
    
    5. **Count set bits**: `bin(num).count('1')` or Brian Kernighan's algorithm: 
       ```python
       count = 0
       while num:
           num &= (num - 1)  # Clear the least significant set bit
           count += 1
       ```
    
    6. **Check power of 2**: `n & (n-1) == 0` - True for powers of 2 (and zero)
    
    7. **Get/clear lowest set bit**: `n & -n` - Isolates the lowest set bit

    ### Applications

    - Low-level system programming
    - Cryptography and hash functions
    - Memory-efficient data structures (bitsets, bloom filters)
    - Optimization of algorithms
    - State representation in dynamic programming

=== "Example"
    ### Count Bits

    ```python
    # Count the number of 1's in the binary representation of each number from 0 to n
    def count_bits(n):
        result = [0] * (n + 1)
        
        for i in range(1, n + 1):
            # Number of 1's in i = number of 1's in (i & (i-1)) + 1
            result[i] = result[i & (i - 1)] + 1
        
        return result
    ```

=== "Practice Problems"
    - Single Number
    - Maximum XOR of Two Numbers in an Array
    - Sum of Two Integers
    - Number of 1 Bits
    - Bitwise AND of Numbers Range

---

**Key Takeaways:**

1. Recognize these patterns when analyzing new problems
2. Practice problems from each category to strengthen understanding
3. Remember that many complex problems involve combinations of these patterns
4. Learning these patterns will significantly improve your problem-solving speed and accuracy

These patterns are your toolkit for algorithmic problem solving - the more familiar you become with them, the more effectively you'll be able to apply them to new challenges.
