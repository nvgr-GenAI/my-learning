# Dynamic Programming Fundamentals

## 📋 What is Dynamic Programming?

Dynamic Programming (DP) is an algorithmic technique for solving optimization problems by breaking them down into simpler subproblems. It works when the problem has:

1. **Overlapping Subproblems**: Same subproblems are solved multiple times
2. **Optimal Substructure**: Optimal solution can be constructed from optimal solutions to subproblems

## 🔍 Key Characteristics

### Overlapping Subproblems

Consider the naive Fibonacci implementation:

```python
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)
```

For `fib(5)`, we recalculate `fib(3)` multiple times:

```
fib(5)
├── fib(4)
│   ├── fib(3)
│   │   ├── fib(2)
│   │   │   ├── fib(1)
│   │   │   └── fib(0)
│   │   └── fib(1)
│   └── fib(2)
│       ├── fib(1)
│       └── fib(0)
└── fib(3)  ← RECALCULATED!
    ├── fib(2)
    │   ├── fib(1)
    │   └── fib(0)
    └── fib(1)
```

### Optimal Substructure

The optimal solution contains optimal solutions to subproblems:

```python
# Maximum path sum in a triangle
# If we know optimal path sums to adjacent cells above,
# we can find optimal path sum to current cell
dp[i][j] = triangle[i][j] + min(dp[i-1][j-1], dp[i-1][j])
```

## 🎯 Two Main Approaches

### 1. Memoization (Top-Down)

Start from the original problem and cache results to avoid recomputation:

```python
def fib_memo(n, memo=None):
    """
    Top-down approach with memoization
    """
    if memo is None:
        memo = {}
    
    # Check if already computed
    if n in memo:
        return memo[n]
    
    # Base cases
    if n <= 1:
        return n
    
    # Recursive call with memoization
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Alternative using functools.lru_cache
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo_decorator(n):
    if n <= 1:
        return n
    return fib_memo_decorator(n-1) + fib_memo_decorator(n-2)
```

**Advantages:**
- Natural recursive structure
- Only computes needed subproblems
- Easy to implement from recursive solution

**Disadvantages:**
- Recursion overhead
- May hit recursion depth limits
- Stack space usage

### 2. Tabulation (Bottom-Up)

Build solution iteratively from smallest subproblems to the target:

```python
def fib_tabulation(n):
    """
    Bottom-up approach with tabulation
    """
    if n <= 1:
        return n
    
    # Create table to store results
    dp = [0] * (n + 1)
    
    # Base cases
    dp[0] = 0
    dp[1] = 1
    
    # Fill table bottom-up
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Space-optimized version
def fib_optimized(n):
    """
    O(1) space optimization - only keep last two values
    """
    if n <= 1:
        return n
    
    prev2 = 0  # fib(0)
    prev1 = 1  # fib(1)
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1
```

**Advantages:**
- No recursion overhead
- Better space efficiency (can often optimize)
- No stack overflow issues

**Disadvantages:**
- May compute unnecessary subproblems
- Less intuitive than recursive approach

## 🔧 DP Problem-Solving Framework

### Step 1: Identify if it's a DP Problem

Look for these signals:

✅ **Optimization problem** (maximize/minimize/count)  
✅ **Decision at each step** (include/exclude, choose path)  
✅ **Subproblems overlap** (same calculations repeated)  
✅ **Optimal substructure** (optimal solution uses optimal subsolutions)

### Step 2: Define the State

What information do we need to represent a subproblem?

```python
# Examples of state definitions:
# dp[i] = answer for first i elements
# dp[i][j] = answer for elements from i to j
# dp[i][j][k] = answer with additional constraint k
```

### Step 3: Find the Recurrence Relation

How does the current state relate to previous states?

```python
# Common patterns:
# dp[i] = dp[i-1] + something           # Linear dependency
# dp[i] = max(dp[i-1], dp[i-2] + val)   # Choice between options
# dp[i][j] = dp[i-1][j] + dp[i][j-1]   # 2D grid paths
```

### Step 4: Identify Base Cases

What are the simplest subproblems we can solve directly?

```python
# Typical base cases:
# dp[0] = initial_value
# dp[1] = first_element_value
# dp[i][0] = boundary_condition
# dp[0][j] = boundary_condition
```

### Step 5: Implement and Optimize

Choose between memoization and tabulation, then optimize space if possible.

## 🎨 Common DP Patterns

### 1. Linear DP

State depends on a constant number of previous states:

```python
def climbing_stairs(n):
    """
    Pattern: dp[i] = dp[i-1] + dp[i-2]
    """
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
```

### 2. 2D DP

State depends on 2D grid or matrix:

```python
def unique_paths(m, n):
    """
    Pattern: dp[i][j] = dp[i-1][j] + dp[i][j-1]
    """
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]
```

### 3. Knapsack Pattern

Choice at each step (include/exclude):

```python
def knapsack_01(weights, values, capacity):
    """
    Pattern: dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]] + value[i])
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't include item i
            dp[i][w] = dp[i-1][w]
            
            # Include item i if it fits
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], 
                              dp[i-1][w-weights[i-1]] + values[i-1])
    
    return dp[n][capacity]
```

### 4. String DP

Operations on strings/sequences:

```python
def longest_common_subsequence(text1, text2):
    """
    Pattern: Compare characters, extend or start new
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

## 📊 Complexity Analysis

### Time Complexity

- **Number of states** × **Time per state**
- Often O(n), O(n²), or O(n³) depending on state dimensions

### Space Complexity

- Can often be optimized from state space to O(1) or O(n)
- Use rolling arrays or track only necessary previous states

```python
# Space optimization example: Fibonacci
def fib_space_optimized(n):
    if n <= 1:
        return n
    
    # Only need last two values
    prev2, prev1 = 0, 1
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1
```

## 🚀 Practice Strategy

### For Beginners

1. Start with **1D DP** problems (Fibonacci, Climbing Stairs)
2. Practice **identifying recurrence relations**
3. Master both **memoization and tabulation**

### For Intermediate

1. Move to **2D DP** problems (Grid paths, Edit Distance)
2. Learn **space optimization** techniques
3. Practice **different DP patterns** (knapsack, string DP)

### For Advanced

1. Tackle **complex state spaces** (3D DP, bitmask DP)
2. Study **interval DP** and **tree DP**
3. Learn **advanced optimizations** (matrix exponentiation, convex hull trick)

## 💡 Common Pitfalls

1. **Off-by-one errors** in indexing
2. **Incorrect base case** definitions
3. **Wrong state transition** logic
4. **Forgetting edge cases** (empty input, single element)
5. **Not optimizing space** when possible

## 🔍 Recognition Patterns

**DP is likely if you see:**

- "Maximum/minimum" path, sum, cost
- "Number of ways" to do something
- "Can you reach/achieve" something
- Decisions at each step
- Recursive structure with overlapping subproblems

**DP is unlikely if:**

- Need actual path/sequence (not just optimal value)
- Real-time/online processing required
- No overlapping subproblems
- Greedy approach works

---

**Next Steps**: Practice with [Easy DP Problems](easy-problems.md) to apply these concepts!
