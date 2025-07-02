# Greedy Algorithms - Hard Problems

This section covers challenging greedy algorithm problems that require deep understanding of greedy choice properties and complex optimization strategies.

## Problem Categories

### Advanced Scheduling Problems
Complex scheduling scenarios with multiple constraints and optimization criteria.

### Multi-dimensional Optimization
Problems involving multiple greedy choices across different dimensions.

### Complex Data Structure Integration
Combining greedy strategies with advanced data structures.

---

## Problems

### 1. Job Scheduling with Deadlines and Profits

**Problem**: Given jobs with deadlines and profits, schedule maximum profit jobs within deadlines.

**Approach**: 
- Sort jobs by profit (descending)
- Use union-find to track available time slots
- Greedily assign jobs to latest possible slots

```python
def job_scheduling_with_profits(jobs):
    """
    Schedule jobs to maximize profit within deadlines.
    
    Args:
        jobs: List of (profit, deadline) tuples
    
    Returns:
        Maximum profit and scheduled jobs
    """
    # Sort by profit in descending order
    jobs.sort(key=lambda x: x[0], reverse=True)
    
    max_deadline = max(job[1] for job in jobs)
    
    # Parent array for union-find
    parent = list(range(max_deadline + 1))
    
    def find_parent(x):
        if parent[x] != x:
            parent[x] = find_parent(parent[x])
        return parent[x]
    
    scheduled = []
    total_profit = 0
    
    for profit, deadline in jobs:
        # Find latest available slot <= deadline
        available_slot = find_parent(deadline)
        
        if available_slot > 0:
            # Schedule job at this slot
            scheduled.append((profit, deadline))
            total_profit += profit
            
            # Update parent to point to previous slot
            parent[available_slot] = find_parent(available_slot - 1)
    
    return total_profit, scheduled

# Example usage
jobs = [(20, 1), (15, 2), (10, 1), (5, 3), (1, 3)]
max_profit, schedule = job_scheduling_with_profits(jobs)
print(f"Maximum profit: {max_profit}")
```

**Time Complexity**: O(n log n + n α(n)) where α is inverse Ackermann function
**Space Complexity**: O(max_deadline)

---

### 2. Minimum Number of Taps to Water Garden

**Problem**: Water a garden using minimum number of taps with given ranges.

**Approach**:
- Convert taps to intervals
- Greedy interval covering with optimization
- Track furthest reachable point

```python
def min_taps_to_water_garden(n, ranges):
    """
    Find minimum taps needed to water entire garden.
    
    Args:
        n: Garden length (0 to n)
        ranges: Array where ranges[i] is range of tap i
    
    Returns:
        Minimum number of taps, or -1 if impossible
    """
    # Convert taps to intervals
    intervals = []
    for i, r in enumerate(ranges):
        if r == 0:
            continue
        start = max(0, i - r)
        end = min(n, i + r)
        intervals.append((start, end))
    
    # Sort by start position
    intervals.sort()
    
    taps = 0
    current_end = 0
    farthest = 0
    i = 0
    
    while current_end < n:
        # Find all intervals that can cover current_end
        while i < len(intervals) and intervals[i][0] <= current_end:
            farthest = max(farthest, intervals[i][1])
            i += 1
        
        # If no progress can be made
        if farthest <= current_end:
            return -1
        
        # Use the tap that reaches farthest
        current_end = farthest
        taps += 1
    
    return taps

# Example usage
n = 5
ranges = [3, 4, 1, 1, 0, 0]
result = min_taps_to_water_garden(n, ranges)
print(f"Minimum taps needed: {result}")
```

**Time Complexity**: O(n log n)
**Space Complexity**: O(n)

---

### 3. Maximum Performance of Team

**Problem**: Choose k engineers to maximize team performance (sum of speed × min efficiency).

**Approach**:
- Sort by efficiency (descending)
- Use min-heap to maintain top k speeds
- Greedily try each efficiency as minimum

```python
import heapq

def max_performance_team(n, speed, efficiency, k):
    """
    Find maximum performance of team of k engineers.
    
    Args:
        n: Number of engineers
        speed: Array of engineer speeds
        efficiency: Array of engineer efficiencies
        k: Team size
    
    Returns:
        Maximum team performance
    """
    MOD = 10**9 + 7
    
    # Combine and sort by efficiency (descending)
    engineers = list(zip(efficiency, speed))
    engineers.sort(reverse=True)
    
    max_performance = 0
    speed_sum = 0
    speed_heap = []  # min-heap
    
    for eff, spd in engineers:
        # Add current engineer
        heapq.heappush(speed_heap, spd)
        speed_sum += spd
        
        # Remove slowest if team size exceeds k
        if len(speed_heap) > k:
            speed_sum -= heapq.heappop(speed_heap)
        
        # Update maximum performance
        # Current efficiency is minimum for this team
        performance = speed_sum * eff
        max_performance = max(max_performance, performance)
    
    return max_performance % MOD

# Example usage
n = 6
speed = [2, 10, 3, 1, 5, 8]
efficiency = [5, 4, 3, 9, 7, 2]
k = 2
result = max_performance_team(n, speed, efficiency, k)
print(f"Maximum performance: {result}")
```

**Time Complexity**: O(n log n + n log k)
**Space Complexity**: O(n + k)

---

### 4. Minimum Cost to Cut Sticks

**Problem**: Cut a stick into pieces with minimum cost where cost equals stick length.

**Approach**:
- Use interval DP with greedy optimization
- Always cut at position that minimizes total cost
- Dynamic programming with memoization

```python
def min_cost_cut_sticks(n, cuts):
    """
    Find minimum cost to cut stick into pieces.
    
    Args:
        n: Length of stick
        cuts: Array of cut positions
    
    Returns:
        Minimum cost to make all cuts
    """
    # Add boundaries
    cuts = [0] + sorted(cuts) + [n]
    m = len(cuts)
    
    # dp[i][j] = minimum cost to cut stick between cuts[i] and cuts[j]
    dp = {}
    
    def solve(left, right):
        if right - left <= 1:
            return 0
        
        if (left, right) in dp:
            return dp[(left, right)]
        
        min_cost = float('inf')
        stick_length = cuts[right] - cuts[left]
        
        # Try each cut position between left and right
        for k in range(left + 1, right):
            cost = stick_length + solve(left, k) + solve(k, right)
            min_cost = min(min_cost, cost)
        
        dp[(left, right)] = min_cost
        return min_cost
    
    return solve(0, m - 1)

# Example usage
n = 7
cuts = [1, 3, 4, 5]
result = min_cost_cut_sticks(n, cuts)
print(f"Minimum cost: {result}")
```

**Time Complexity**: O(m³) where m is number of cuts
**Space Complexity**: O(m²)

---

### 5. Candy Distribution (Advanced)

**Problem**: Distribute candies to children with complex rating constraints.

**Approach**:
- Two-pass greedy algorithm
- Handle local minima and maxima
- Optimize for minimum candies while satisfying constraints

```python
def candy_distribution_advanced(ratings):
    """
    Distribute minimum candies with rating constraints.
    Children with higher ratings get more candies than neighbors.
    
    Args:
        ratings: Array of children's ratings
    
    Returns:
        Minimum candies needed
    """
    n = len(ratings)
    if n == 0:
        return 0
    
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

def candy_with_constraints(ratings, constraints):
    """
    Advanced candy distribution with additional constraints.
    
    Args:
        ratings: Array of children's ratings
        constraints: List of (i, j, diff) meaning candies[i] - candies[j] >= diff
    
    Returns:
        Minimum candies or -1 if impossible
    """
    n = len(ratings)
    candies = candy_distribution_advanced(ratings)
    
    # Check if additional constraints can be satisfied
    candy_array = [1] * n
    
    # Apply basic rating constraints
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candy_array[i] = candy_array[i-1] + 1
    
    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candy_array[i] = max(candy_array[i], candy_array[i+1] + 1)
    
    # Apply additional constraints
    changed = True
    while changed:
        changed = False
        for i, j, diff in constraints:
            if candy_array[i] - candy_array[j] < diff:
                candy_array[i] = candy_array[j] + diff
                changed = True
    
    return sum(candy_array)

# Example usage
ratings = [1, 0, 2]
result = candy_distribution_advanced(ratings)
print(f"Minimum candies: {result}")
```

**Time Complexity**: O(n) for basic, O(n × c) for constraints where c is number of constraints
**Space Complexity**: O(n)

---

## Key Patterns for Hard Greedy Problems

### 1. **Union-Find Integration**
- Use union-find for efficient slot/resource tracking
- Particularly useful in scheduling problems

### 2. **Multi-Pass Algorithms**
- Sometimes need multiple passes to satisfy all constraints
- Each pass optimizes different aspects

### 3. **Heap-Based Optimization**
- Use heaps to maintain optimal choices
- Efficient for dynamic selection problems

### 4. **Interval Processing**
- Convert problems to interval coverage/scheduling
- Use sweepline techniques

### 5. **Dynamic Greedy Choices**
- Greedy choice may depend on current state
- Combine with other algorithmic techniques

## Advanced Optimization Techniques

### Lazy Propagation
For problems involving range updates and queries.

### Segment Trees
When greedy choices need efficient range operations.

### Binary Search on Answer
When greedy strategy can validate a solution.

## Common Pitfalls

1. **Overlooking Edge Cases**: Handle empty inputs, single elements
2. **Incorrect Sorting**: Choose the right sorting criteria
3. **Greedy Choice Verification**: Ensure choices are actually optimal
4. **Constraint Satisfaction**: Verify all constraints are met
5. **Optimization Opportunities**: Look for redundant computations

These hard problems demonstrate the power of greedy algorithms when combined with sophisticated data structures and careful analysis of optimal substructure properties.
