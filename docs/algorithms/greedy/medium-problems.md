# Greedy Algorithms - Medium Problems

## 🎯 Learning Objectives

Master intermediate greedy algorithm patterns and optimization techniques:

- Multi-pass greedy algorithms
- Interval scheduling and optimization
- Priority queue-based greedy approaches
- Complex constraint handling
- Greedy with mathematical insights

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Candy Distribution | Two-Pass Greedy | Medium | O(n) | O(n) |
    | 2 | Task Scheduler | Priority Queue/Math | Medium | O(n log k) | O(k) |
    | 3 | Partition Labels | Greedy with Last Index | Medium | O(n) | O(1) |
    | 4 | Minimum Arrows to Burst Balloons | Interval Merging | Medium | O(n log n) | O(1) |
    | 5 | Non-overlapping Intervals | Interval Scheduling | Medium | O(n log n) | O(1) |
    | 6 | Jump Game II | Greedy BFS | Medium | O(n) | O(1) |
    | 7 | Reconstruct Queue by Height | Insertion Greedy | Medium | O(n²) | O(n) |
    | 8 | Advantage Shuffle | Greedy Pairing | Medium | O(n log n) | O(n) |
    | 9 | Minimum Cost to Connect Sticks | Priority Queue | Medium | O(n log n) | O(n) |
    | 10 | Car Pooling | Event Sorting | Medium | O(n log n) | O(n) |
    | 11 | Meeting Rooms II | Sweep Line | Medium | O(n log n) | O(n) |
    | 12 | Remove K Digits | Stack Greedy | Medium | O(n) | O(n) |
    | 13 | Wiggle Subsequence | State Machine | Medium | O(n) | O(1) |
    | 14 | Bag of Tokens | Two Pointers + Greedy | Medium | O(n log n) | O(1) |
    | 15 | Hand of Straights | Greedy Grouping | Medium | O(n log n) | O(n) |

=== "🎯 Advanced Patterns"

    **🔄 Multi-Pass Greedy:**
    - Left-to-right and right-to-left passes
    - Constraint satisfaction from both directions
    - Global optimization through local passes
    
    **⏰ Interval Problems:**
    - Sorting by start/end times
    - Greedy interval selection
    - Event-based processing
    
    **📊 Priority-Based Greedy:**
    - Use heaps for dynamic optimization
    - Maintain optimal order during processing
    - Frequency and priority-based decisions
    
    **🎯 Mathematical Greedy:**
    - Formula-based optimization
    - Pattern recognition and exploitation
    - State machine approaches

=== "💡 Solutions"

    === "Candy Distribution"
        ```python
        def candy(ratings):
            """
            Distribute minimum candies satisfying rating constraints
            Two-pass greedy: left-to-right, then right-to-left
            """
            n = len(ratings)
            candies = [1] * n
            
            # Left to right: handle increasing ratings
            for i in range(1, n):
                if ratings[i] > ratings[i-1]:
                    candies[i] = candies[i-1] + 1
            
            # Right to left: handle decreasing ratings
            for i in range(n-2, -1, -1):
                if ratings[i] > ratings[i+1]:
                    candies[i] = max(candies[i], candies[i+1] + 1)
            
            return sum(candies)
        ```
    
    === "Task Scheduler"
        ```python
        from collections import Counter
        import heapq
        
        def leastInterval(tasks, n):
            """
            Minimum time to complete all tasks with cooling period
            Mathematical approach or priority queue
            """
            # Mathematical approach
            counter = Counter(tasks)
            max_freq = max(counter.values())
            max_count = sum(1 for freq in counter.values() if freq == max_freq)
            
            # Minimum time based on most frequent task
            min_time = (max_freq - 1) * (n + 1) + max_count
            
            # Can't be less than total tasks
            return max(min_time, len(tasks))
            
            # Alternative: Priority Queue approach
            # heap = [-freq for freq in counter.values()]
            # heapq.heapify(heap)
            # time = 0
            # 
            # while heap:
            #     temp = []
            #     for _ in range(n + 1):
            #         if heap:
            #             temp.append(heapq.heappop(heap))
            #     
            #     for freq in temp:
            #         if freq < -1:
            #             heapq.heappush(heap, freq + 1)
            #     
            #     time += n + 1 if heap else len(temp)
            # 
            # return time
        ```
    
    === "Partition Labels"
        ```python
        def partitionLabels(s):
            """
            Partition string so each letter appears in at most one part
            Greedy: track last occurrence of each character
            """
            # Find last occurrence of each character
            last = {char: i for i, char in enumerate(s)}
            
            partitions = []
            start = end = 0
            
            for i, char in enumerate(s):
                end = max(end, last[char])
                
                # If we've reached the end of current partition
                if i == end:
                    partitions.append(end - start + 1)
                    start = i + 1
            
            return partitions
        ```
    
    === "Minimum Arrows to Burst Balloons"
        ```python
        def findMinArrowShots(points):
            """
            Minimum arrows to burst all balloons
            Greedy: sort by end point, shoot at earliest end
            """
            if not points:
                return 0
            
            # Sort by end coordinate
            points.sort(key=lambda x: x[1])
            
            arrows = 1
            arrow_pos = points[0][1]
            
            for start, end in points[1:]:
                if start > arrow_pos:
                    # Need new arrow
                    arrows += 1
                    arrow_pos = end
            
            return arrows
        ```
    
    === "Jump Game II"
        ```python
        def jump(nums):
            """
            Minimum jumps to reach last index
            Greedy BFS: track current and next reachable range
            """
            if len(nums) <= 1:
                return 0
            
            jumps = 0
            current_max = 0  # Max reach with current jumps
            next_max = 0     # Max reach with one more jump
            
            for i in range(len(nums) - 1):
                next_max = max(next_max, i + nums[i])
                
                # If we've reached the limit of current jumps
                if i == current_max:
                    jumps += 1
                    current_max = next_max
                    
                    # Early termination if we can reach the end
                    if current_max >= len(nums) - 1:
                        break
            
            return jumps
        ```
    
    === "Queue Reconstruction by Height"
        ```python
        def reconstructQueue(people):
            """
            Reconstruct queue based on height and position
            Greedy: sort by height desc, then insert by k value
            """
            # Sort by height (descending), then by k (ascending)
            people.sort(key=lambda x: (-x[0], x[1]))
            
            result = []
            for person in people:
                # Insert at the position specified by k
                result.insert(person[1], person)
            
            return result
        ```
    
    === "Remove K Digits"
        ```python
        def removeKdigits(num, k):
            """
            Remove k digits to get smallest possible number
            Stack-based greedy: remove larger digits when possible
            """
            stack = []
            to_remove = k
            
            for digit in num:
                # Remove larger digits from stack
                while stack and to_remove > 0 and stack[-1] > digit:
                    stack.pop()
                    to_remove -= 1
                stack.append(digit)
            
            # Remove remaining digits from end if needed
            while to_remove > 0:
                stack.pop()
                to_remove -= 1
            
            # Build result, handle leading zeros
            result = ''.join(stack).lstrip('0')
            return result if result else '0'
        ```
    
    === "Meeting Rooms II"
        ```python
        import heapq
        
        def minMeetingRooms(intervals):
            """
            Minimum meeting rooms needed
            Greedy: track end times with heap
            """
            if not intervals:
                return 0
            
            # Sort by start time
            intervals.sort(key=lambda x: x[0])
            
            # Min heap to track end times
            min_heap = []
            
            for interval in intervals:
                # If a room is free, reuse it
                if min_heap and min_heap[0] <= interval[0]:
                    heapq.heappop(min_heap)
                
                # Add current meeting's end time
                heapq.heappush(min_heap, interval[1])
            
            return len(min_heap)
        ```

=== "📊 Advanced Techniques"

    **🔧 Algorithm Design:**
    - **Multi-Pass Strategy**: Handle different constraints separately
    - **Event Processing**: Sort events by time, process sequentially
    - **Priority Queues**: Maintain optimal ordering dynamically
    - **Mathematical Insights**: Find closed-form solutions when possible
    - **State Machines**: Track different states and transitions
    
    **⚡ Optimization Strategies:**
    - **Early Termination**: Stop when optimal solution found
    - **Constraint Relaxation**: Handle easier constraints first
    - **Data Structure Choice**: Heap, stack, or sorted arrays
    - **Space-Time Trade-offs**: Memory for faster lookups
    
    **🎯 Pattern Recognition:**
    - **Interval Scheduling**: Activity selection variants
    - **Multi-dimensional Sorting**: Handle multiple criteria
    - **Frequency-based**: Count, sort, then process
    - **Range Problems**: Track overlaps and boundaries

=== "🚀 Expert Tips"

    **💡 Problem-Solving Strategy:**
    1. **Identify Constraints**: What rules must be satisfied?
    2. **Find Greedy Choice**: What local decision leads to global optimum?
    3. **Prove Correctness**: Exchange argument or induction
    4. **Handle Edge Cases**: Empty input, single element, all same
    5. **Optimize Implementation**: Choose right data structures
    
    **🔍 Common Challenges:**
    - **Multiple Constraints**: May need multiple passes or complex logic
    - **Tie Breaking**: Define clear rules for equal elements
    - **Optimal Ordering**: Choose sort criteria carefully
    - **State Management**: Track complex state during processing
    
    **🏆 Advanced Techniques:**
    - **Sweep Line**: For interval and geometric problems
    - **Two Pointers**: After sorting for pairing/matching
    - **Stack Monotonicity**: Maintain increasing/decreasing property
    - **Mathematical Formulation**: Derive formulas instead of simulation

## 📝 Summary

These medium greedy problems demonstrate:

- **Multi-Pass Algorithms** for complex constraint satisfaction
- **Interval Processing** with sorting and sweep line techniques
- **Priority-Based Decisions** using heaps and queues
- **Mathematical Optimization** with closed-form solutions
- **Advanced Data Structures** for maintaining optimal states

These techniques are essential for:
- **System Design** resource allocation and scheduling
- **Operations Research** optimization problems
- **Real-time Systems** decision making under constraints
- **Competitive Programming** advanced greedy problems

Master these patterns to tackle complex optimization problems efficiently!
