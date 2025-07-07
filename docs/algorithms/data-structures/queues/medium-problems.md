# Queues - Medium Problems

## 🎯 Learning Objectives

Master intermediate queue techniques and applications:

- Multi-queue processing strategies
- Queue with other data structures (monotonic queues)
- BFS traversal optimizations
- Priority queue implementations
- Circular buffer applications

=== "📋 Problem List"

    | # | Problem | Difficulty | Topics | Solution |
    |---|---------|------------|--------|----------|
    | 1 | Sliding Window Maximum | Medium | Monotonic Queue, Array | [Solution](#problem-1) |
    | 2 | Perfect Squares | Medium | BFS, DP | [Solution](#problem-2) |
    | 3 | Design Circular Queue | Medium | Implementation | [Solution](#problem-3) |
    | 4 | Number of Recent Calls | Medium | Queue Design | [Solution](#problem-4) |
    | 5 | Design Hit Counter | Medium | Design | [Solution](#problem-5) |
    | 6 | Task Scheduler | Medium | Greedy, Priority Queue | [Solution](#problem-6) |
    | 7 | Implement Stack using Queues | Medium | Implementation | [Solution](#problem-7) |
    | 8 | Moving Average from Data Stream | Medium | Sliding Window | [Solution](#problem-8) |
    | 9 | Find the Winner of the Circular Game | Medium | Simulation | [Solution](#problem-9) |
    | 10 | Design Snake Game | Medium | Design, Queue | [Solution](#problem-10) |
    | 11 | Product of the Last K Numbers | Medium | Design | [Solution](#problem-11) |
    | 12 | Reveal Cards In Increasing Order | Medium | Queue, Simulation | [Solution](#problem-12) |
    | 13 | Queue Reconstruction by Height | Medium | Greedy | [Solution](#problem-13) |
    | 14 | Walls and Gates | Medium | BFS | [Solution](#problem-14) |
    | 15 | Rotting Oranges | Medium | BFS | [Solution](#problem-15) |

=== "📚 Interview Tips"

    ## Queue Patterns to Master
    
    ### 1. Breadth-First Search (BFS)
    
    BFS is one of the most common queue applications in interviews. Key points:
    
    - Use for level-order traversals (trees, graphs)
    - Finding shortest paths in unweighted graphs
    - Expanding outward from source nodes
    
    ```python
    def bfs(graph, start):
        queue = deque([start])
        visited = {start}
        
        while queue:
            node = queue.popleft()
            # Process node
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    ```
    
    ### 2. Monotonic Queue
    
    A powerful technique where the queue maintains order (ascending or descending):
    
    - Used for sliding window maximums/minimums
    - Efficient range queries
    - Always maintains useful elements
    
    ```python
    # Monotonic decreasing queue for sliding window maximum
    dq = deque()
    for i, val in enumerate(nums):
        # Remove elements outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
            
        # Remove smaller elements (they'll never be maximum)
        while dq and nums[dq[-1]] < val:
            dq.pop()
            
        dq.append(i)
        # Front of queue has maximum element index
    ```
    
    ### 3. Queue with Two Stacks
    
    Implementing a queue using two stacks showcases understanding of data structures:
    
    ```python
    class MyQueue:
        def __init__(self):
            self.stack_push = []
            self.stack_pop = []
        
        def push(self, x):
            self.stack_push.append(x)
        
        def pop(self):
            self._move_if_needed()
            return self.stack_pop.pop()
        
        def peek(self):
            self._move_if_needed()
            return self.stack_pop[-1]
        
        def _move_if_needed(self):
            if not self.stack_pop:
                while self.stack_push:
                    self.stack_pop.append(self.stack_push.pop())
    ```
    
    ### 4. Circular Queue/Buffer
    
    Important for problems with fixed-size windows or buffers:
    
    - Efficient memory usage
    - Constant time operations
    - Useful for stream processing
    
    ```python
    class CircularQueue:
        def __init__(self, k):
            self.queue = [0] * k
            self.size = k
            self.front = self.rear = -1
        
        def enqueue(self, value):
            if self.is_full():
                return False
            
            if self.is_empty():
                self.front = 0
                
            self.rear = (self.rear + 1) % self.size
            self.queue[self.rear] = value
            return True
    ```
    
    ### 5. Priority Queue
    
    Key for optimization problems and scheduling:
    
    - Always extracts highest/lowest priority item
    - Useful for task scheduling
    - Common in Dijkstra's, Prim's algorithms
    
    ```python
    import heapq
    
    pq = []  # Min heap
    heapq.heappush(pq, (priority, item))
    priority, item = heapq.heappop(pq)
    ```
    
    ## Common Queue Interview Mistakes
    
    1. **Forgetting edge cases**: Empty queue, single element
    2. **Incorrect BFS implementation**: Not marking nodes as visited
    3. **Off-by-one errors**: In circular queue implementations
    4. **Inefficient operations**: Using list.pop(0) instead of deque.popleft()
    5. **Not using the right queue type**: Regular queue vs. priority queue

=== "📝 Study Plan"

    ## Queue Mastery Study Plan
    
    ### Week 1: Queue Basics
    
    - **Day 1-2**: Review queue fundamentals and implementations
      - Standard queue, deque, priority queue
      - Time complexities of operations
    
    - **Day 3-4**: Simple queue applications
      - Implement stack using queue
      - Design circular queue
      - Moving average from data stream
    
    - **Day 5-7**: Basic BFS problems
      - Number of islands
      - Binary tree level order traversal
      - Word ladder
    
    ### Week 2: Intermediate Techniques
    
    - **Day 8-10**: Monotonic queue problems
      - Sliding window maximum
      - Largest rectangle in histogram
      - Trapping rain water
    
    - **Day 11-13**: Priority queue applications
      - Top k frequent elements
      - Merge k sorted lists
      - Find median from data stream
    
    - **Day 14**: Circular buffer applications
      - Design hit counter
      - Moving average from data stream
    
    ### Week 3: Advanced Topics
    
    - **Day 15-17**: Complex BFS variations
      - 01 Matrix
      - Shortest path in binary matrix
      - Walls and gates
    
    - **Day 18-20**: Multi-queue problems
      - Task scheduler
      - Rearrange string k distance apart
    
    - **Day 21**: Specialized queue applications
      - LRU Cache implementation using queues
      - Snake game design
    
    ### Week 4: Mastery and Optimization
    
    - **Day 22-24**: Optimization techniques
      - Space-time trade-offs in queue problems
      - Lazy deletion approaches
    
    - **Day 25-26**: Advanced applications
      - Stream processing patterns
      - Producer-consumer patterns
    
    - **Day 27-28**: Review and practice
      - Mock interviews
      - Timed problem-solving
    
    ## Resources
    
    - **Books**:
      - "Elements of Programming Interviews"
      - "Cracking the Coding Interview"
    
    - **Online Platforms**:
      - LeetCode Queue Tag Problems
      - HackerRank Data Structures Path
    
    - **Visualization Tools**:
      - VisuAlgo Queue Visualization
      - Algorithm Visualizer

=== "Problem 1: Sliding Window Maximum"

    **LeetCode 239** | **Difficulty: Medium**

    ## Problem Statement

    You are given an array of integers `nums` and a sliding window of size `k` that moves from left to right. Return the maximum value in each sliding window.

    **Example:**
    ```
    Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
    Output: [3,3,5,5,6,7]
    ```

    ## Solution: Monotonic Queue

    ```python
    from collections import deque

    def maxSlidingWindow(nums, k):
        """
        Find maximum in each sliding window using monotonic deque.
        
        Time: O(n)
        Space: O(k)
        """
        if not nums or k == 0:
            return []
        
        dq = deque()  # Store indices
        result = []
        
        for i in range(len(nums)):
            # Remove indices outside current window
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # Remove indices with smaller values
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()
            
            dq.append(i)
            
            # Add maximum to result when window is complete
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    ```

    ## Alternative: Brute Force Approach

    ```python
    def maxSlidingWindow(nums, k):
        """
        Brute force approach.
        
        Time: O(n*k)
        Space: O(n-k+1)
        """
        n = len(nums)
        if n * k == 0:
            return []
        
        result = []
        for i in range(n - k + 1):
            max_val = max(nums[i:i + k])
            result.append(max_val)
            
        return result
    ```

    ## Key Insights

    - Monotonic queue maintains decreasing order (potential maximum values)
    - Each element enters and exits the queue exactly once: O(n)
    - We only need to store indices to track window boundaries
    - Front of the queue always contains the maximum element in current window
    - We can use a deque for efficient operations at both ends

=== "Problem 2: Perfect Squares"

    **LeetCode 279** | **Difficulty: Medium**

    ## Problem Statement

    Given an integer `n`, return the least number of perfect square numbers that sum to `n`.

    A perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself.

    **Example:**
    ```
    Input: n = 12
    Output: 3
    Explanation: 12 = 4 + 4 + 4
    ```

    ## Solution: BFS Approach

    ```python
    from collections import deque

    def numSquares(n):
        """
        BFS approach to find least number of perfect squares.
        
        Time: O(n * sqrt(n))
        Space: O(n)
        """
        if n <= 0:
            return 0
        
        # Generate all possible perfect squares
        squares = []
        i = 1
        while i * i <= n:
            squares.append(i * i)
            i += 1
        
        # BFS to find shortest path
        queue = deque([(n, 0)])  # (remaining sum, count)
        visited = {n}
        
        while queue:
            remaining, count = queue.popleft()
            
            if remaining == 0:
                return count
            
            for square in squares:
                if square > remaining:
                    break
                
                next_val = remaining - square
                if next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, count + 1))
        
        return -1  # Should never reach here
    ```

    ## Alternative: Dynamic Programming Approach

    ```python
    def numSquares(n):
        """
        Dynamic programming approach.
        
        Time: O(n * sqrt(n))
        Space: O(n)
        """
        # Initialize DP array
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # Calculate minimum number for each value
        for i in range(1, n + 1):
            j = 1
            while j * j <= i:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
        
        return dp[n]
    ```

    ## Key Insights

    - BFS guarantees the shortest path (minimum number of squares)
    - Using a visited set prevents redundant calculations
    - Each level in the BFS represents adding one more perfect square
    - Precomputing all possible squares improves efficiency
    - Dynamic programming offers an alternative approach with similar complexity

=== "Problem 3: Design Circular Queue"

    **LeetCode 622** | **Difficulty: Medium**

    ## Problem Statement

    Design your implementation of the circular queue. The circular queue is a linear data structure in which the operations are performed based on FIFO principle and the last position is connected back to the first position to make a circle.

    Implement the `MyCircularQueue` class:
    - `MyCircularQueue(k)`: Initializes the object with the size of the queue to be `k`.
    - `enQueue(value)`: Inserts an element into the circular queue. Return true if the operation is successful.
    - `deQueue()`: Deletes an element from the circular queue. Return true if the operation is successful.
    - `Front()`: Gets the front item from the queue. If the queue is empty, return -1.
    - `Rear()`: Gets the last item from the queue. If the queue is empty, return -1.
    - `isEmpty()`: Checks whether the circular queue is empty or not.
    - `isFull()`: Checks whether the circular queue is full or not.

    ## Solution: Using Array

    ```python
    class MyCircularQueue:
        def __init__(self, k):
            """
            Initialize your data structure here. Set the size of the queue to be k.
            """
            self.queue = [0] * k
            self.size = k
            self.head = -1
            self.tail = -1
        
        def enQueue(self, value):
            """
            Insert an element into the circular queue. Return true if the operation is successful.
            """
            if self.isFull():
                return False
            
            if self.isEmpty():
                self.head = 0
                
            self.tail = (self.tail + 1) % self.size
            self.queue[self.tail] = value
            return True
        
        def deQueue(self):
            """
            Delete an element from the circular queue. Return true if the operation is successful.
            """
            if self.isEmpty():
                return False
                
            if self.head == self.tail:
                self.head = -1
                self.tail = -1
                return True
                
            self.head = (self.head + 1) % self.size
            return True
        
        def Front(self):
            """
            Get the front item from the queue.
            """
            if self.isEmpty():
                return -1
            return self.queue[self.head]
        
        def Rear(self):
            """
            Get the last item from the queue.
            """
            if self.isEmpty():
                return -1
            return self.queue[self.tail]
        
        def isEmpty(self):
            """
            Checks whether the circular queue is empty.
            """
            return self.head == -1
        
        def isFull(self):
            """
            Checks whether the circular queue is full.
            """
            return (self.tail + 1) % self.size == self.head
    ```

    ## Alternative: Using Linked List

    ```python
    class Node:
        def __init__(self, value):
            self.value = value
            self.next = None

    class MyCircularQueue:
        def __init__(self, k):
            self.capacity = k
            self.size = 0
            self.head = None
            self.tail = None
        
        def enQueue(self, value):
            if self.isFull():
                return False
            
            new_node = Node(value)
            if self.isEmpty():
                self.head = new_node
                self.tail = new_node
                new_node.next = new_node  # point to itself
            else:
                new_node.next = self.head
                self.tail.next = new_node
                self.tail = new_node
            
            self.size += 1
            return True
        
        def deQueue(self):
            if self.isEmpty():
                return False
            
            if self.size == 1:
                self.head = None
                self.tail = None
            else:
                self.head = self.head.next
                self.tail.next = self.head
            
            self.size -= 1
            return True
        
        def Front(self):
            return -1 if self.isEmpty() else self.head.value
        
        def Rear(self):
            return -1 if self.isEmpty() else self.tail.value
        
        def isEmpty(self):
            return self.size == 0
        
        def isFull(self):
            return self.size == self.capacity
    ```

    ## Key Insights

    - Circular buffer elegantly handles wraparound with modulo arithmetic
    - Special handling required for empty/full states
    - Array-based implementation is more memory efficient
    - Linked list implementation simplifies insertion/deletion logic
    - Both maintain O(1) time complexity for all operations

=== "Problem 4: Number of Recent Calls"

    **LeetCode 933** | **Difficulty: Medium**

    ## Problem Statement

    Implement the `RecentCounter` class:

    - `RecentCounter()` Initializes the counter with zero recent requests.
    - `ping(int t)` Adds a new request at time t, where t represents some time in milliseconds, and returns the number of requests that has happened in the past 3000 milliseconds (including the new request).

    Specifically, return the number of requests that have happened in the inclusive range [t - 3000, t].

    **Example:**
    ```
    Input:
    ["RecentCounter", "ping", "ping", "ping", "ping"]
    [[], [1], [100], [3001], [3002]]
    Output:
    [null, 1, 2, 3, 3]
    ```

    ## Solution: Queue Approach

    ```python
    from collections import deque

    class RecentCounter:
        def __init__(self):
            """
            Initialize your data structure here.
            """
            self.queue = deque()
        
        def ping(self, t):
            """
            Adds new request at time t, returns count of recent requests.
            
            Time: O(n) in worst case, but amortized O(1)
            Space: O(n) where n is number of calls in 3000ms window
            """
            self.queue.append(t)
            
            # Remove requests older than t - 3000
            while self.queue and self.queue[0] < t - 3000:
                self.queue.popleft()
            
            return len(self.queue)
    ```

    ## Alternative: Binary Search with Array

    ```python
    import bisect

    class RecentCounter:
        def __init__(self):
            self.requests = []
        
        def ping(self, t):
            """
            Using sorted array with binary search.
            
            Time: O(log n) for search + O(n) for insert
            Space: O(n)
            """
            self.requests.append(t)
            
            # Find index of first request >= t - 3000
            idx = bisect.bisect_left(self.requests, t - 3000)
            
            # Return count of requests in range
            return len(self.requests) - idx
    ```

    ## Key Insights

    - Queue naturally models the sliding window of time
    - We only need to maintain requests within the 3000ms window
    - Requests arrive in chronological order (increasing t)
    - Queue approach maintains amortized O(1) complexity
    - Each request enters and exits the queue exactly once

=== "Problem 5: Design Hit Counter"

    **LeetCode 362** | **Difficulty: Medium**

    ## Problem Statement

    Design a hit counter which counts the number of hits received in the past 5 minutes (300 seconds).

    Implement the `HitCounter` class:
    - `HitCounter()` Initializes the object of the hit counter system.
    - `hit(timestamp)` Records a hit at the given timestamp.
    - `getHits(timestamp)` Returns the number of hits in the past 5 minutes from the given timestamp.

    **Example:**
    ```
    Input:
    ["HitCounter", "hit", "hit", "hit", "getHits", "hit", "getHits", "getHits"]
    [[], [1], [2], [3], [4], [300], [300], [301]]
    Output:
    [null, null, null, null, 3, null, 4, 3]
    ```

    ## Solution: Queue-Based Approach

    ```python
    from collections import deque

    class HitCounter:
        def __init__(self):
            """
            Initialize your data structure here.
            """
            self.hits = deque()
        
        def hit(self, timestamp):
            """
            Record a hit at the given timestamp.
            
            Time: O(1)
            Space: O(n) where n is number of hits in 5 min window
            """
            self.hits.append(timestamp)
        
        def getHits(self, timestamp):
            """
            Return the number of hits in the past 5 minutes.
            
            Time: O(n) worst case, but amortized O(1)
            Space: O(1)
            """
            # Remove hits older than 5 minutes (300 seconds)
            while self.hits and self.hits[0] <= timestamp - 300:
                self.hits.popleft()
            
            return len(self.hits)
    ```

    ## Alternative: Optimized for Multiple Same Timestamps

    ```python
    class HitCounter:
        def __init__(self):
            # Store (timestamp, count) pairs
            self.hits = deque()
            self.total = 0
        
        def hit(self, timestamp):
            """
            Optimized for repeated timestamps.
            
            Time: O(1)
            Space: O(min(300, n)) where n is number of unique timestamps
            """
            if self.hits and self.hits[-1][0] == timestamp:
                # Increment count for existing timestamp
                self.hits[-1] = (timestamp, self.hits[-1][1] + 1)
            else:
                # Add new timestamp
                self.hits.append((timestamp, 1))
            
            self.total += 1
        
        def getHits(self, timestamp):
            # Remove hits older than 5 minutes
            while self.hits and self.hits[0][0] <= timestamp - 300:
                self.total -= self.hits[0][1]
                self.hits.popleft()
            
            return self.total
    ```

    ## Key Insights

    - Queue efficiently handles time-based sliding window
    - Timestamps arrive in chronological order (increasing)
    - Optimization can be made for repeated timestamps
    - Counter needs to handle both single hits and range queries
    - For high volume systems, aggregation reduces memory usage

=== "Problem 6: Task Scheduler"

    **LeetCode 621** | **Difficulty: Medium**

    ## Problem Statement

    Given a characters array `tasks`, representing the tasks a CPU needs to do, where each letter represents a different task. Tasks could be done in any order. Each task is done in one unit of time. For each unit of time, the CPU could complete either one task or just be idle.

    However, there is a non-negative integer `n` that represents the cooldown period between two same tasks (the same letter in the array), that is that there must be at least `n` units of time between any two same tasks.

    Return the least number of units of time that the CPU will take to finish all the given tasks.

    **Example:**
    ```
    Input: tasks = ["A","A","A","B","B","B"], n = 2
    Output: 8
    Explanation: 
    A -> B -> idle -> A -> B -> idle -> A -> B
    ```

    ## Solution: Greedy Approach

    ```python
    from collections import Counter
    import heapq

    def leastInterval(tasks, n):
        """
        Greedy approach with priority queue.
        
        Time: O(n log(26))
        Space: O(26)
        """
        # Count frequency of each task
        task_counts = Counter(tasks)
        
        # Max heap for task frequencies
        max_heap = [-count for count in task_counts.values()]
        heapq.heapify(max_heap)
        
        time = 0
        queue = []  # (frequency, next_available_time)
        
        while max_heap or queue:
            time += 1
            
            # Process most frequent task if available
            if max_heap:
                freq = heapq.heappop(max_heap) + 1  # +1 because of negative frequencies
                if freq < 0:
                    # Task still has instances left, add to cooldown queue
                    queue.append((freq, time + n))
            
            # Check if any cooled-down task can be added back to heap
            if queue and queue[0][1] <= time:
                freq, _ = queue.pop(0)
                heapq.heappush(max_heap, freq)
        
        return time
    ```

    ## Alternative: Mathematical Approach

    ```python
    from collections import Counter

    def leastInterval(tasks, n):
        """
        Mathematical approach.
        
        Time: O(n)
        Space: O(26)
        """
        # Count task frequencies
        task_counts = Counter(tasks)
        
        # Find the maximum frequency
        max_freq = max(task_counts.values())
        
        # Count how many tasks have the maximum frequency
        max_freq_tasks = sum(1 for count in task_counts.values() if count == max_freq)
        
        # Calculate time required
        # Formula: (max_freq - 1) * (n + 1) + max_freq_tasks
        time_required = (max_freq - 1) * (n + 1) + max_freq_tasks
        
        # Return maximum of time required and total tasks
        # (in case when idle slots aren't needed)
        return max(time_required, len(tasks))
    ```

    ## Key Insights

    - Most frequent tasks determine the minimum time needed
    - Greedy approach: always schedule the most frequent remaining task
    - Cooldown periods can create "idle" slots that must be counted
    - Mathematical solution: calculate idle slots based on most frequent task
    - Edge case: when many different tasks exist, idle slots may not be needed

=== "Problem 7: Implement Stack using Queues"

    **LeetCode 225** | **Difficulty: Medium**

    ## Problem Statement

    Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (`push`, `pop`, `top`, and `empty`).

    Implement the `MyStack` class:
    - `MyStack()` Initializes the stack object.
    - `push(int x)` Pushes element x to the top of the stack.
    - `pop()` Removes the element on the top of the stack and returns it.
    - `top()` Returns the element on the top of the stack.
    - `empty()` Returns `true` if the stack is empty, `false` otherwise.

    **Example:**
    ```
    Input:
    ["MyStack", "push", "push", "top", "pop", "empty"]
    [[], [1], [2], [], [], []]
    Output:
    [null, null, null, 2, 2, false]
    ```

    ## Solution: Two Queues Approach

    ```python
    from collections import deque

    class MyStack:
        def __init__(self):
            """
            Initialize your data structure here.
            """
            self.queue1 = deque()
            self.queue2 = deque()
        
        def push(self, x):
            """
            Push element x onto stack.
            
            Time: O(1)
            Space: O(n)
            """
            self.queue1.append(x)
        
        def pop(self):
            """
            Removes the element on top of the stack and returns it.
            
            Time: O(n)
            Space: O(1)
            """
            # Move all elements except the last one to queue2
            while len(self.queue1) > 1:
                self.queue2.append(self.queue1.popleft())
            
            # Get the last element (stack top)
            result = self.queue1.popleft()
            
            # Swap queue1 and queue2
            self.queue1, self.queue2 = self.queue2, self.queue1
            
            return result
        
        def top(self):
            """
            Get the top element.
            
            Time: O(n)
            Space: O(1)
            """
            # Move all elements except the last one to queue2
            while len(self.queue1) > 1:
                self.queue2.append(self.queue1.popleft())
            
            # Get the last element (stack top)
            result = self.queue1[0]
            
            # Move the last element to queue2 as well
            self.queue2.append(self.queue1.popleft())
            
            # Swap queue1 and queue2
            self.queue1, self.queue2 = self.queue2, self.queue1
            
            return result
        
        def empty(self):
            """
            Returns whether the stack is empty.
            
            Time: O(1)
            Space: O(1)
            """
            return len(self.queue1) == 0
    ```

    ## Alternative: Single Queue Approach

    ```python
    from collections import deque

    class MyStack:
        def __init__(self):
            self.queue = deque()
        
        def push(self, x):
            """
            Push with rotation to maintain stack order.
            
            Time: O(n)
            Space: O(n)
            """
            # Add the new element
            self.queue.append(x)
            
            # Rotate the queue to make the newest element at the front
            for _ in range(len(self.queue) - 1):
                self.queue.append(self.queue.popleft())
        
        def pop(self):
            """
            Time: O(1)
            Space: O(1)
            """
            return self.queue.popleft()
        
        def top(self):
            """
            Time: O(1)
            Space: O(1)
            """
            return self.queue[0]
        
        def empty(self):
            """
            Time: O(1)
            Space: O(1)
            """
            return len(self.queue) == 0
    ```

    ## Key Insights

    - Queue (FIFO) and stack (LIFO) have opposite behaviors
    - Two-queue approach: maintain elements in one queue, use second for reordering
    - Single-queue approach: rotate queue after each push
    - Both approaches have O(n) complexity for some operations
    - Trade-off between push and pop efficiency in different implementations

=== "Problem 8: Moving Average from Data Stream"

    **LeetCode 346** | **Difficulty: Medium**

    ## Problem Statement

    Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

    Implement the `MovingAverage` class:
    - `MovingAverage(int size)` Initializes the object with the window size.
    - `double next(int val)` Returns the moving average of the last `size` values.

    **Example:**
    ```
    Input:
    ["MovingAverage", "next", "next", "next", "next"]
    [[3], [1], [10], [3], [5]]
    Output:
    [null, 1.0, 5.5, 4.66667, 6.0]
    ```

    ## Solution: Queue-Based Approach

    ```python
    from collections import deque

    class MovingAverage:
        def __init__(self, size):
            """
            Initialize your data structure here.
            """
            self.size = size
            self.queue = deque()
            self.window_sum = 0
        
        def next(self, val):
            """
            Returns the moving average of current window.
            
            Time: O(1)
            Space: O(size)
            """
            # If window is full, remove oldest element
            if len(self.queue) == self.size:
                self.window_sum -= self.queue.popleft()
            
            # Add new element
            self.queue.append(val)
            self.window_sum += val
            
            # Calculate average
            return self.window_sum / len(self.queue)
    ```

    ## Alternative: Circular Buffer Approach

    ```python
    class MovingAverage:
        def __init__(self, size):
            self.size = size
            self.buffer = [0] * size
            self.window_sum = 0
            self.count = 0
            self.head = 0
        
        def next(self, val):
            """
            Using circular buffer for efficient memory usage.
            
            Time: O(1)
            Space: O(size)
            """
            # Calculate position to update
            tail = (self.head + self.count) % self.size if self.count < self.size else self.head
            
            # Update sum by removing old value and adding new one
            self.window_sum = self.window_sum - self.buffer[tail] + val
            
            # Store new value
            self.buffer[tail] = val
            
            # Update count and head
            if self.count < self.size:
                self.count += 1
            else:
                self.head = (self.head + 1) % self.size
            
            # Return average
            return self.window_sum / self.count
    ```

    ## Key Insights

    - Queue naturally models sliding window operations
    - Track running sum to avoid recalculation
    - Fixed-size window limits space complexity
    - Circular buffer optimizes memory usage
    - Both implementations achieve O(1) time complexity

=== "Problem 9: Find the Winner of the Circular Game"

    **LeetCode 1823** | **Difficulty: Medium**

    ## Problem Statement

    There are `n` friends sitting in a circle and numbered from 1 to n. Starting from the 1st friend, the counting begins from 1 to k. The counting stops at the kth friend, who is then removed from the circle. The next counting begins from the friend immediately after the removed one. The process continues until only one friend remains.

    Return the ID of the last remaining friend.

    **Example:**
    ```
    Input: n = 5, k = 2
    Output: 3
    Explanation: Here are the steps:
    1) Start at friend 1.
    2) Count 2 friends clockwise: 1, 2. Remove friend 2.
    3) Count 2 friends: 3, 4. Remove friend 4.
    4) Count 2 friends: 5, 1. Remove friend 1.
    5) Count 2 friends: 3, 5. Remove friend 5.
    6) Only friend 3 remains.
    ```

    ## Solution: Queue Simulation

    ```python
    from collections import deque

    def findTheWinner(n, k):
        """
        Simulate the game using queue.
        
        Time: O(n*k)
        Space: O(n)
        """
        # Initialize queue with all friends
        queue = deque(range(1, n + 1))
        
        while len(queue) > 1:
            # Move k-1 friends from front to back
            for _ in range(k - 1):
                queue.append(queue.popleft())
            
            # Remove the kth friend
            queue.popleft()
        
        # Return the last remaining friend
        return queue[0]
    ```

    ## Alternative: Josephus Problem (Mathematical)

    ```python
    def findTheWinner(n, k):
        """
        Mathematical solution to Josephus problem.
        
        Time: O(n)
        Space: O(1)
        """
        # Base case: with one person, they always win
        winner = 0
        
        # Build solution incrementally
        for i in range(1, n + 1):
            # Formula: (previous_position + k) % current_size
            winner = (winner + k) % i
        
        # Convert to 1-indexed
        return winner + 1
    ```

    ## Key Insights

    - Queue simulation directly models the circle of friends
    - For each step, move (k-1) people to the back, then remove the kth
    - This is a classical Josephus problem with a mathematical solution
    - Mathematical approach avoids simulation but is less intuitive
    - Queue solution is easier to understand but less efficient for large n and k

=== "Problem 10: Design Snake Game"

    **LeetCode 353** | **Difficulty: Medium**

    ## Problem Statement

    Design a Snake game that is played on a device with screen size `width x height`. The snake starts at position `(0, 0)` and food appears at a random position `(foodX, foodY)`. The snake moves in four directions: up, down, left, and right.

    The goal is to grow the snake as long as possible by eating food. When the snake eats the food, the food disappears and another piece of food appears at a random position. However, the snake will die if it hits the border or itself.

    Implement the `SnakeGame` class:
    - `SnakeGame(width, height, food)` Initializes the game with screen width and height, and food positions.
    - `move(direction)` Returns the score after the snake moves. -1 if game over.

    **Example:**
    ```
    Input:
    ["SnakeGame", "move", "move", "move", "move", "move", "move"]
    [[3, 2, [[1, 2], [0, 1]]], ["R"], ["D"], ["R"], ["U"], ["L"], ["U"]]
    Output:
    [null, 0, 0, 1, 1, 2, -1]
    ```

    ## Solution: Queue + Set Approach

    ```python
    from collections import deque

    class SnakeGame:
        def __init__(self, width, height, food):
            """
            Initialize your data structure here.
            
            @param width - screen width
            @param height - screen height
            @param food - A list of food positions
            """
            self.width = width
            self.height = height
            self.food = food
            self.food_index = 0
            self.score = 0
            
            # Snake represented as deque [(0,0)]
            self.snake = deque([(0, 0)])
            
            # Set of positions occupied by snake body
            self.snake_set = {(0, 0)}
            
            # Direction vectors: up, right, down, left
            self.directions = {'U': (-1, 0), 'R': (0, 1), 'D': (1, 0), 'L': (0, -1)}
        
        def move(self, direction):
            """
            Moves the snake.
            
            @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down
            @return The game's score after the move. -1 if game over.
            
            Time: O(1)
            Space: O(width * height)
            """
            # Get current head position
            head_row, head_col = self.snake[0]
            
            # Calculate new head position
            dr, dc = self.directions[direction]
            new_row, new_col = head_row + dr, head_col + dc
            
            # Check if game over: hit border
            if not (0 <= new_row < self.height and 0 <= new_col < self.width):
                return -1
            
            # Check if snake eats food
            if self.food_index < len(self.food) and [new_row, new_col] == self.food[self.food_index]:
                # Eat food and increment score and food index
                self.score += 1
                self.food_index += 1
            else:
                # Remove tail if no food eaten
                tail = self.snake.pop()
                self.snake_set.remove(tail)
                
                # Check if game over: hit itself (except for old tail)
                if (new_row, new_col) in self.snake_set:
                    return -1
            
            # Add new head to snake
            self.snake.appendleft((new_row, new_col))
            self.snake_set.add((new_row, new_col))
            
            return self.score
    ```

    ## Alternative: Array-Based Approach

    ```python
    class SnakeGame:
        def __init__(self, width, height, food):
            self.width = width
            self.height = height
            self.food = food
            self.food_index = 0
            self.score = 0
            
            # Snake body positions stored as array
            self.snake = [(0, 0)]
            
            self.directions = {'U': (-1, 0), 'R': (0, 1), 'D': (1, 0), 'L': (0, -1)}
        
        def move(self, direction):
            """
            Array-based implementation.
            
            Time: O(n) where n is snake length
            Space: O(width * height)
            """
            # Get current head position
            head = self.snake[0]
            
            # Calculate new head position
            dr, dc = self.directions[direction]
            new_head = (head[0] + dr, head[1] + dc)
            
            # Check if game over: hit border
            if not (0 <= new_head[0] < self.height and 0 <= new_head[1] < self.width):
                return -1
            
            # Check if snake eats food
            if self.food_index < len(self.food) and list(new_head) == self.food[self.food_index]:
                self.food_index += 1
                self.score += 1
            else:
                # Remove tail if no food eaten
                self.snake.pop()
            
            # Check if game over: hit itself
            if new_head in self.snake:
                return -1
            
            # Add new head to snake
            self.snake.insert(0, new_head)
            
            return self.score
    ```

    ## Key Insights

    - Queue (deque) naturally models snake body with efficient head/tail operations
    - Set provides O(1) lookups for collision detection
    - Snake grows when food is eaten, otherwise maintain length
    - Game over conditions: hitting border or hitting snake body
    - Time complexity is amortized O(1) with deque + set approach

=== "Problem 11: Product of the Last K Numbers"

    **LeetCode 1352** | **Difficulty: Medium**

    ## Problem Statement

    Implement the `ProductOfNumbers` class:
    - `ProductOfNumbers()` Initializes the object with an empty array of integers.
    - `add(int num)` Adds the number num to the back of the array.
    - `getProduct(int k)` Returns the product of the last k numbers in the current array.

    **Example:**
    ```
    Input:
    ["ProductOfNumbers","add","add","add","add","add","getProduct","getProduct","getProduct","add","getProduct"]
    [[],[3],[0],[2],[5],[4],[2],[3],[4],[8],[2]]
    Output:
    [null,null,null,null,null,null,20,40,0,null,32]
    ```

    ## Solution: Prefix Product Approach

    ```python
    class ProductOfNumbers:
        def __init__(self):
            """
            Initialize your data structure here.
            """
            self.products = [1]  # Start with 1 for prefix product
        
        def add(self, num):
            """
            Add a number to the stream.
            
            Time: O(1)
            Space: O(n)
            """
            if num == 0:
                # Reset products list when encountering zero
                self.products = [1]
            else:
                # Multiply the newest number with the previous product
                self.products.append(self.products[-1] * num)
        
        def getProduct(self, k):
            """
            Returns the product of the last k numbers.
            
            Time: O(1)
            Space: O(1)
            """
            if k >= len(self.products):
                # If k is larger than our list, there must be a zero
                return 0
            
            # Calculate product using prefix products
            return self.products[-1] // self.products[-k-1]
    ```

    ## Alternative: Queue with Running Product

    ```python
    from collections import deque

    class ProductOfNumbers:
        def __init__(self):
            self.numbers = deque()
            self.running_product = 1
        
        def add(self, num):
            """
            Queue-based approach.
            
            Time: O(1)
            Space: O(n)
            """
            self.numbers.append(num)
            if num == 0:
                self.running_product = 1
                # Mark zeros explicitly
                self.numbers = deque([0 if x != 0 else 0 for x in self.numbers])
            else:
                self.running_product *= num
        
        def getProduct(self, k):
            """
            Time: O(k)
            Space: O(1)
            """
            product = 1
            # Check if any zero in the last k numbers
            for i in range(1, k + 1):
                if i > len(self.numbers):
                    return 0
                
                if self.numbers[-i] == 0:
                    return 0
                
                product *= self.numbers[-i]
                
            return product
    ```

    ## Key Insights

    - Prefix product allows O(1) time for queries
    - Special handling required for zeros
    - Division is used to get product of specific range
    - Zeros require resetting the prefix product array
    - Trade-off between space and query time complexity

=== "Problem 12: Reveal Cards In Increasing Order"

    **LeetCode 950** | **Difficulty: Medium**

    ## Problem Statement

    You are given an integer array `deck`. You have to deal the cards in a specific way: initially, all cards are placed in a deck face-down, in order from top to bottom.

    Then, you reveal the top card, place the next card at the bottom, reveal the next top card, place the next card at the bottom, and so on until all cards are revealed.

    Return an ordering of the deck that would reveal the cards in increasing order.

    **Example:**
    ```
    Input: deck = [17,13,11,2,3,5,7]
    Output: [2,13,3,11,5,17,7]
    Explanation:
    We get the deck in the order [2,13,3,11,5,17,7] (from top to bottom).
    Reveal: 2, Place: 13, Reveal: 3, Place: 11, ...
    ```

    ## Solution: Queue Simulation

    ```python
    from collections import deque

    def deckRevealedIncreasing(deck):
        """
        Simulate the card revealing process in reverse.
        
        Time: O(n log n)
        Space: O(n)
        """
        n = len(deck)
        
        # Sort deck in ascending order
        deck.sort()
        
        # Initialize queue with indices
        queue = deque(range(n))
        
        # Result array
        result = [0] * n
        
        for card in deck:
            # Get position for current card
            idx = queue.popleft()
            result[idx] = card
            
            # Put next position at the bottom
            if queue:
                queue.append(queue.popleft())
        
        return result
    ```

    ## Alternative: Reverse Simulation

    ```python
    def deckRevealedIncreasing(deck):
        """
        Directly simulate the reverse process.
        
        Time: O(n log n)
        Space: O(n)
        """
        # Sort deck in ascending order
        deck.sort()
        
        # Result will be built by simulating the reverse process
        result = []
        
        # Process cards from largest to smallest
        for card in reversed(deck):
            if result:
                # Move the last card to the front
                result.insert(0, result.pop())
            
            # Insert current card at the front
            result.insert(0, card)
        
        return result
    ```

    ## Key Insights

    - Queue simulates the card dealing process
    - Working backward from sorted cards simplifies the problem
    - Alternating between revealing and placing cards
    - Time complexity dominated by sorting
    - Both approaches use simulation, but from different directions

=== "Problem 13: Queue Reconstruction by Height"

    **LeetCode 406** | **Difficulty: Medium**

    ## Problem Statement

    You are given an array of people `people` where `people[i] = [hi, ki]` represents the ith person with height `hi` and with exactly `ki` other people in front who have a height greater than or equal to `hi`.

    Reconstruct and return the queue that would result in the given array.

    **Example:**
    ```
    Input: people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
    Output: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
    ```

    ## Solution: Sort and Insert Approach

    ```python
    def reconstructQueue(people):
        """
        Sort and insert to correct positions.
        
        Time: O(n²)
        Space: O(n)
        """
        # Sort by height (descending) and then by k (ascending)
        people.sort(key=lambda x: (-x[0], x[1]))
        
        result = []
        
        for person in people:
            # Insert each person at their k position
            result.insert(person[1], person)
        
        return result
    ```

    ## Alternative: Binary Indexed Tree Approach

    ```python
    def reconstructQueue(people):
        """
        Using binary indexed tree for counting available positions.
        
        Time: O(n log n)
        Space: O(n)
        """
        n = len(people)
        
        # Sort by height (ascending) and then by k (descending)
        people.sort(key=lambda x: (x[0], -x[1]))
        
        # Initialize result array with placeholders
        result = [None] * n
        
        # Initialize binary indexed tree (BIT)
        bit = [1] * (n + 1)
        
        def update(i, val):
            while i <= n:
                bit[i] += val
                i += i & -i
        
        def query(i):
            s = 0
            while i > 0:
                s += bit[i]
                i -= i & -i
            return s
        
        # Fill positions from shortest to tallest
        for h, k in people:
            # Find the position with exactly k empty spots before it
            left, right = 1, n
            while left < right:
                mid = (left + right) // 2
                if query(mid) <= k:
                    left = mid + 1
                else:
                    right = mid
            
            # Convert to 0-indexed position
            pos = left - 1
            
            # Place the person in the result
            result[pos] = [h, k]
            
            # Mark this position as filled
            update(left, -1)
        
        return result
    ```

    ## Key Insights

    - Greedy approach: place tallest people first, then shorter ones
    - Inserting at index k guarantees correct relative positioning
    - Each person's position depends only on people of greater or equal height
    - Binary indexed tree approach improves time complexity but is more complex
    - The sort order is crucial for the greedy approach to work

=== "Problem 14: Walls and Gates"

    **LeetCode 286** | **Difficulty: Medium**

    ## Problem Statement

    You are given an m x n grid `rooms` filled with three kinds of values:
    - `-1`: A wall or an obstacle.
    - `0`: A gate.
    - `INF`: An empty room, where `INF = 2^31 - 1`.

    Fill each empty room with the distance to its nearest gate. If there's no accessible gate, leave it as `INF`.

    **Example:**
    ```
    Input: rooms = [
      [2147483647,-1,0,2147483647],
      [2147483647,2147483647,2147483647,-1],
      [2147483647,-1,2147483647,-1],
      [0,-1,2147483647,2147483647]
    ]
    Output: [
      [3,-1,0,1],
      [2,2,1,-1],
      [1,-1,2,-1],
      [0,-1,3,4]
    ]
    ```

    ## Solution: BFS from Gates

    ```python
    from collections import deque

    def wallsAndGates(rooms):
        """
        BFS starting from all gates simultaneously.
        
        Time: O(m*n)
        Space: O(m*n)
        """
        if not rooms or not rooms[0]:
            return
        
        m, n = len(rooms), len(rooms[0])
        INF = 2147483647
        queue = deque()
        
        # Add all gates to queue as starting points
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    queue.append((i, j))
        
        # Directions: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # BFS to fill distances
        while queue:
            row, col = queue.popleft()
            
            for dr, dc in directions:
                r, c = row + dr, col + dc
                
                # Check if in bounds and is empty room
                if 0 <= r < m and 0 <= c < n and rooms[r][c] == INF:
                    # Update distance
                    rooms[r][c] = rooms[row][col] + 1
                    queue.append((r, c))
    ```

    ## Alternative: DFS Approach

    ```python
    def wallsAndGates(rooms):
        """
        DFS from each gate.
        
        Time: O(m*n)
        Space: O(m*n) for recursion stack
        """
        if not rooms or not rooms[0]:
            return
        
        m, n = len(rooms), len(rooms[0])
        
        def dfs(i, j, distance):
            # Base cases: out of bounds, wall, or already shorter path found
            if i < 0 or i >= m or j < 0 or j >= n or rooms[i][j] < distance:
                return
            
            # Update distance
            rooms[i][j] = distance
            
            # Visit neighbors
            dfs(i-1, j, distance + 1)  # Up
            dfs(i, j+1, distance + 1)  # Right
            dfs(i+1, j, distance + 1)  # Down
            dfs(i, j-1, distance + 1)  # Left
        
        # Start DFS from each gate
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    dfs(i, j, 0)
    ```

    ## Key Insights

    - BFS guarantees shortest distance from gates to empty rooms
    - Starting from all gates simultaneously avoids redundant calculations
    - Each cell is visited at most once (when its shortest distance is found)
    - DFS approach is simpler but less efficient (could revisit cells)
    - Queue-based BFS is more memory efficient than recursive DFS

=== "Problem 15: Rotting Oranges"

    **LeetCode 994** | **Difficulty: Medium**

    ## Problem Statement

    You are given an `m x n` grid where each cell can have one of three values:
    - `0`: an empty cell.
    - `1`: a fresh orange.
    - `2`: a rotten orange.

    Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

    Return the minimum number of minutes that must elapse until no fresh oranges are left. If this is impossible, return -1.

    **Example:**
    ```
    Input: grid = [
      [2,1,1],
      [1,1,0],
      [0,1,1]
    ]
    Output: 4
    ```

    ## Solution: BFS Approach

    ```python
    from collections import deque

    def orangesRotting(grid):
        """
        BFS to simulate rotting process.
        
        Time: O(m*n)
        Space: O(m*n)
        """
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh_count = 0
        
        # Count fresh oranges and add rotten oranges to queue
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c, 0))  # (row, col, time)
                elif grid[r][c] == 1:
                    fresh_count += 1
        
        # If no fresh oranges, return 0
        if fresh_count == 0:
            return 0
        
        # Directions: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        max_time = 0
        
        # BFS to rot oranges
        while queue:
            row, col, time = queue.popleft()
            max_time = max(max_time, time)
            
            for dr, dc in directions:
                r, c = row + dr, col + dc
                
                # Check if valid and fresh
                if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 1:
                    # Mark as rotten
                    grid[r][c] = 2
                    fresh_count -= 1
                    queue.append((r, c, time + 1))
        
        # Check if any fresh oranges remain
        return max_time if fresh_count == 0 else -1
    ```

    ## Alternative: In-Place BFS

    ```python
    def orangesRotting(grid):
        """
        In-place BFS with multiple iterations.
        
        Time: O(m*n*max_time)
        Space: O(1)
        """
        rows, cols = len(grid), len(grid[0])
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        minutes = 0
        fresh_count = sum(1 for r in range(rows) for c in range(cols) if grid[r][c] == 1)
        
        # If no fresh oranges, return 0
        if fresh_count == 0:
            return 0
        
        # Simulate minutes passing
        while True:
            rotted = False
            
            # Mark oranges that will rot this minute with 3
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 2:
                        for dr, dc in directions:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                                grid[nr][nc] = 3  # Will rot in this minute
                                rotted = True
            
            # Update grid for this minute
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 3:
                        grid[r][c] = 2
                        fresh_count -= 1
            
            # If oranges rotted this minute, increment time
            if rotted:
                minutes += 1
            else:
                break
        
        # Check if any fresh oranges remain
        return minutes if fresh_count == 0 else -1
    ```

    ## Key Insights

    - BFS simulates the rotting process minute by minute
    - Starting from all rotten oranges ensures minimum time calculation
    - Keep track of fresh oranges to detect unreachable ones
    - Time complexity is O(m*n) since each cell is processed at most once
    - Alternative approach trades time efficiency for space efficiency
