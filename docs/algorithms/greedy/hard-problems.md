# Greedy Algorithms - Hard Problems

## 🎯 Learning Objectives

Master advanced greedy algorithm techniques and complex optimization:

- Multi-dimensional greedy optimization
- Greedy with advanced data structures
- Mathematical optimization and insights
- Complex constraint satisfaction
- Real-world optimization scenarios

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Job Scheduling with Profits | Union-Find + Greedy | Hard | O(n log n) | O(n) |
    | 2 | Minimum Taps to Water Garden | Greedy Range Coverage | Hard | O(n log n) | O(n) |
    | 3 | Maximum Performance of Team | Heap + Greedy | Hard | O(n² log n) | O(n) |
    | 4 | Minimum Cost to Cut Sticks | DP + Greedy Insight | Hard | O(n³) | O(n²) |
    | 5 | IPO (Capital and Profits) | Two Heaps | Hard | O(n log n) | O(n) |
    | 6 | Shortest Subarray Sum ≥ K | Deque + Prefix Sum | Hard | O(n) | O(n) |
    | 7 | Candy Distribution (Complex) | Multi-pass + Math | Hard | O(n) | O(n) |
    | 8 | Video Stitching | Greedy Interval | Hard | O(n log n) | O(1) |
    | 9 | Course Schedule III | Priority Queue | Hard | O(n log n) | O(n) |
    | 10 | Minimum Cost Hiring K Workers | Ratio Sorting + Heap | Hard | O(n² log n) | O(n) |
    | 11 | Split Array into Fibonacci | Greedy + Validation | Hard | O(n) | O(n) |
    | 12 | Create Maximum Number | Monotonic Stack | Hard | O((m+n)³) | O(m+n) |
    | 13 | Remove Duplicate Letters II | Stack + Counter | Hard | O(n) | O(n) |
    | 14 | Minimum Domino Rotations | Greedy Validation | Hard | O(n) | O(1) |
    | 15 | Patching Array | Greedy Range Building | Hard | O(n + log m) | O(1) |

=== "🎯 Advanced Patterns"

    **🔗 Data Structure Integration:**
    - Union-Find for dynamic connectivity
    - Heaps for maintaining optimal order
    - Stacks for monotonic properties
    - Segment trees for range operations
    
    **📊 Multi-dimensional Optimization:**
    - Sort by one dimension, optimize others
    - Trade-offs between multiple objectives
    - Pareto optimal solutions
    - Weighted optimization criteria
    
    **🎯 Mathematical Greedy:**
    - Closed-form solutions
    - Mathematical insights and proofs
    - Optimization theory applications
    - Constraint analysis
    
    **🔄 Complex Constraint Handling:**
    - Multiple passes with different criteria
    - Constraint relaxation and tightening
    - Feasibility checking
    - Dynamic constraint updates

=== "💡 Solutions"

    === "Job Scheduling with Profits"
        ```python
        def jobScheduling(startTime, endTime, profit):
            """
            Schedule jobs to maximize profit (no overlap)
            Sort by end time, use DP with binary search
            """
            jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])
            dp = [0]  # dp[i] = max profit up to job i
            
            def findLatestNonOverlapping(index):
                left, right = 0, len(dp) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if jobs[mid][1] <= jobs[index][0]:
                        left = mid + 1
                    else:
                        right = mid - 1
                return right
            
            for i, (start, end, profit) in enumerate(jobs):
                latest = findLatestNonOverlapping(i)
                dp.append(max(dp[-1], dp[latest + 1] + profit))
            
            return dp[-1]
        ```
    
    === "Minimum Taps to Water Garden"
        ```python
        def minTaps(n, ranges):
            """
            Minimum taps to water garden [0, n]
            Greedy: for each position, use tap with maximum reach
            """
            # Convert to intervals
            intervals = []
            for i, r in enumerate(ranges):
                intervals.append([max(0, i - r), min(n, i + r)])
            
            # Sort by start position
            intervals.sort()
            
            taps = 0
            current_end = 0
            next_end = 0
            i = 0
            
            while current_end < n:
                # Find all taps that can start from current_end
                while i < len(intervals) and intervals[i][0] <= current_end:
                    next_end = max(next_end, intervals[i][1])
                    i += 1
                
                # If no progress possible
                if next_end <= current_end:
                    return -1
                
                taps += 1
                current_end = next_end
            
            return taps
        ```
    
    === "Maximum Performance of Team"
        ```python
        import heapq
        
        def maxPerformance(n, speed, efficiency, k):
            """
            Choose k engineers to maximize performance
            Performance = sum(speed) * min(efficiency)
            """
            MOD = 10**9 + 7
            
            # Combine and sort by efficiency (descending)
            engineers = sorted(zip(efficiency, speed), reverse=True)
            
            min_heap = []  # To maintain k engineers with highest speeds
            speed_sum = 0
            max_performance = 0
            
            for eff, spd in engineers:
                # Add current engineer
                heapq.heappush(min_heap, spd)
                speed_sum += spd
                
                # Remove slowest engineer if team size > k
                if len(min_heap) > k:
                    speed_sum -= heapq.heappop(min_heap)
                
                # Update max performance (current efficiency is minimum)
                max_performance = max(max_performance, speed_sum * eff)
            
            return max_performance % MOD
        ```
    
    === "IPO (Capital and Profits)"
        ```python
        import heapq
        
        def findMaximizedCapital(k, w, profits, capital):
            """
            Choose k projects to maximize capital
            Greedy: always pick highest profit among affordable projects
            """
            # Sort projects by capital requirement
            projects = sorted(zip(capital, profits))
            
            available = []  # Max heap for profits of affordable projects
            i = 0
            
            for _ in range(k):
                # Add all affordable projects to heap
                while i < len(projects) and projects[i][0] <= w:
                    heapq.heappush(available, -projects[i][1])  # Negative for max heap
                    i += 1
                
                # If no affordable projects, break
                if not available:
                    break
                
                # Pick project with maximum profit
                w += -heapq.heappop(available)
            
            return w
        ```
    
    === "Course Schedule III"
        ```python
        import heapq
        
        def scheduleCourse(courses):
            """
            Maximum courses that can be taken within deadlines
            Greedy: sort by deadline, use heap to manage time
            """
            # Sort by deadline
            courses.sort(key=lambda x: x[1])
            
            max_heap = []  # Store negative durations for max heap
            time = 0
            
            for duration, deadline in courses:
                # Try to take this course
                heapq.heappush(max_heap, -duration)
                time += duration
                
                # If deadline exceeded, remove longest course
                if time > deadline:
                    time += heapq.heappop(max_heap)  # Add negative = subtract
            
            return len(max_heap)
        ```
    
    === "Video Stitching"
        ```python
        def videoStitching(clips, T):
            """
            Minimum clips to cover [0, T]
            Greedy: for each position, choose clip with maximum reach
            """
            clips.sort()
            
            current_end = 0
            next_end = 0
            count = 0
            i = 0
            
            while current_end < T:
                # Find all clips that start within current coverage
                while i < len(clips) and clips[i][0] <= current_end:
                    next_end = max(next_end, clips[i][1])
                    i += 1
                
                # If no progress possible
                if next_end <= current_end:
                    return -1
                
                count += 1
                current_end = next_end
            
            return count
        ```
    
    === "Patching Array"
        ```python
        def minPatches(nums, n):
            """
            Minimum numbers to add so all sums [1, n] are possible
            Greedy: maintain current reachable range
            """
            patches = 0
            current_max = 0  # Maximum sum we can form
            i = 0
            
            while current_max < n:
                if i < len(nums) and nums[i] <= current_max + 1:
                    # Use existing number
                    current_max += nums[i]
                    i += 1
                else:
                    # Add patch: current_max + 1
                    current_max += current_max + 1
                    patches += 1
            
            return patches
        ```

=== "📊 Advanced Techniques"

    **🔧 Algorithm Design:**
    - **Multi-stage Greedy**: Different criteria at different stages
    - **Constraint Propagation**: Use constraints to guide choices
    - **Approximation Algorithms**: Near-optimal solutions for NP-hard problems
    - **Online Algorithms**: Make decisions without future knowledge
    - **Competitive Analysis**: Analyze worst-case performance
    
    **⚡ Optimization Strategies:**
    - **Data Structure Augmentation**: Add metadata for faster queries
    - **Lazy Evaluation**: Defer expensive computations
    - **Preprocessing**: Sort or index for faster access
    - **Mathematical Insights**: Use formulas instead of simulation
    
    **🎯 Problem Classification:**
    - **Scheduling**: Activity selection variants
    - **Resource Allocation**: Optimize limited resources
    - **Coverage**: Minimum set cover problems
    - **Matching**: Bipartite matching variants
    - **Network Flow**: Greedy flow algorithms

=== "🚀 Expert Insights"

    **💡 Advanced Problem-Solving:**
    1. **Identify Optimization Objective**: What exactly are we optimizing?
    2. **Analyze Constraints**: What limits our choices?
    3. **Find Greedy Property**: Why does local optimum work?
    4. **Prove Correctness**: Exchange argument or contradiction
    5. **Handle Edge Cases**: Empty input, impossible solutions
    
    **🔍 Complex Challenges:**
    - **Multiple Objectives**: Balance competing goals
    - **Dynamic Constraints**: Constraints change during execution
    - **Approximation Ratios**: How close to optimal?
    - **Robustness**: Handle uncertain or noisy input
    
    **🏆 Professional Applications:**
    - **System Resource Management**: CPU, memory, bandwidth allocation
    - **Financial Optimization**: Portfolio selection, trading strategies
    - **Operations Research**: Supply chain, logistics optimization
    - **Machine Learning**: Feature selection, model compression
    - **Network Design**: Routing, topology optimization

## 📝 Summary

These hard greedy problems demonstrate:

- **Advanced Data Structures** for maintaining optimal states
- **Multi-dimensional Optimization** with complex trade-offs
- **Mathematical Insights** for closed-form solutions
- **Real-world Constraints** and practical considerations
- **Theoretical Foundations** of greedy algorithm design

These techniques are crucial for:

- **System Design** at scale with resource constraints
- **Operations Research** and optimization consulting
- **Competitive Programming** advanced problem solving
- **Research** in approximation algorithms
- **Industry Applications** requiring real-time optimization

Master these patterns to tackle the most challenging optimization problems in computer science and industry!
