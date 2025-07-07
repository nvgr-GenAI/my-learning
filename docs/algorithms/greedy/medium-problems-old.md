# Greedy Medium Problems

## Overview
Medium-level greedy algorithm problems that require deeper understanding of problem-solving techniques.

## Core Problems

### 1. Gas Station Problem
**Problem**: Determine if you can complete a circular tour visiting all gas stations.

```python
def canCompleteCircuit(gas, cost):
    """
    Greedy approach to gas station problem
    Time: O(n), Space: O(1)
    """
    total_surplus = 0
    surplus = 0
    start = 0
    
    for i in range(len(gas)):
        total_surplus += gas[i] - cost[i]
        surplus += gas[i] - cost[i]
        
        if surplus < 0:
            surplus = 0
            start = i + 1
    
    return -1 if total_surplus < 0 else start
```

### 2. Candy Distribution
**Problem**: Minimum candies to distribute to children based on ratings.

```python
def candy(ratings):
    """
    Two-pass greedy solution
    Time: O(n), Space: O(n)
    """
    n = len(ratings)
    candies = [1] * n
    
    # Left to right pass
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1
    
    # Right to left pass
    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)
    
    return sum(candies)
```

### 3. Task Scheduler
**Problem**: Minimum time to complete tasks with cooldown.

```python
def leastInterval(tasks, n):
    """
    Greedy task scheduling
    Time: O(m), Space: O(1) where m = len(tasks)
    """
    from collections import Counter
    import heapq
    
    task_counts = Counter(tasks)
    max_heap = [-count for count in task_counts.values()]
    heapq.heapify(max_heap)
    
    time = 0
    
    while max_heap:
        temp = []
        
        # Process n+1 time slots
        for _ in range(n + 1):
            if max_heap:
                count = heapq.heappop(max_heap)
                if count < -1:
                    temp.append(count + 1)
            time += 1
            
            if not max_heap and not temp:
                break
        
        # Add back remaining tasks
        for count in temp:
            heapq.heappush(max_heap, count)
    
    return time
```

### 4. Partition Labels
**Problem**: Partition string into maximum number of parts.

```python
def partitionLabels(s):
    """
    Greedy partitioning using last occurrence
    Time: O(n), Space: O(1)
    """
    last_occurrence = {}
    for i, char in enumerate(s):
        last_occurrence[char] = i
    
    result = []
    start = 0
    end = 0
    
    for i, char in enumerate(s):
        end = max(end, last_occurrence[char])
        
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    
    return result
```

### 5. Minimum Number of Arrows
**Problem**: Minimum arrows to burst all balloons.

```python
def findMinArrowShots(points):
    """
    Interval scheduling greedy approach
    Time: O(n log n), Space: O(1)
    """
    if not points:
        return 0
    
    # Sort by end point
    points.sort(key=lambda x: x[1])
    
    arrows = 1
    end = points[0][1]
    
    for i in range(1, len(points)):
        if points[i][0] > end:
            arrows += 1
            end = points[i][1]
    
    return arrows
```

## Problem-Solving Patterns

### 1. Two-Pass Technique
```python
# Left-to-right pass for one constraint
# Right-to-left pass for another constraint
# Example: Candy problem
```

### 2. Interval Scheduling
```python
# Sort intervals by end time
# Greedily select non-overlapping intervals
# Example: Arrow shooting, meeting rooms
```

### 3. Priority Queue Approach
```python
# Use heap for optimal selection
# Example: Task scheduler, Huffman coding
```

## Key Insights

1. **Multiple Constraints**: Often require multiple passes
2. **Sorting Strategy**: Choose sorting key carefully
3. **Local Optimal**: Ensure local choices lead to global optimum
4. **Proof of Correctness**: Always verify greedy choice property

## Practice Problems

- [ ] Gas Station
- [ ] Candy
- [ ] Task Scheduler
- [ ] Partition Labels
- [ ] Minimum Arrows for Balloons
- [ ] Queue Reconstruction by Height
- [ ] Non-overlapping Intervals
- [ ] Minimum Number of Taps
