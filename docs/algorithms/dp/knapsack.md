# Knapsack Pattern

## Introduction

The 0/1 Knapsack pattern is a fundamental dynamic programming pattern that involves selecting items to maximize value while respecting a weight constraint. Each item can be either fully included or excluded.

=== "Overview"
    **Core Idea**: Choose items to maximize value while respecting a weight constraint, with each item being either fully included or excluded.
    
    **When to Use**:
    
    - When you need to select a subset of items to optimize some value
    - When each item can only be used once (taken or not taken)
    - When you have capacity constraints (weight, space, time, etc.)
    - When making yes/no decisions for each item in a collection
    
    **Recurrence Relation**: `dp[i][w] = max(dp[i-1][w], val[i-1] + dp[i-1][w-wt[i-1]])`
    
    **Real-World Applications**:
    
    - Portfolio optimization in finance
    - Resource allocation in project management
    - Cargo loading problems
    - Budget allocation across projects or investments
    - Server resource allocation in cloud computing

=== "Example Problems"
    - **0/1 Knapsack**: Maximize value of items in a knapsack without exceeding weight capacity
      - Classic problem: Given weights and values of n items, find the maximum value subset that fits in a knapsack of capacity W
      - Example: Items with values [60, 100, 120] and weights [10, 20, 30], capacity = 50 → Maximum value = 220
    
    - **Subset Sum**: Determine if a subset of numbers can sum to a target value
      - Variation: Set values equal to weights and check if dp[n][target] is true
      - Example: [3, 34, 4, 12, 5, 2], target=9 → True (4+5=9)
    
    - **Equal Sum Partition**: Can the array be divided into two subsets with equal sum?
      - Approach: Calculate total sum, if odd then impossible, else find subset with sum = total/2
      - Tests understanding of reducing problems to the knapsack framework
    
    - **Target Sum**: Assign + and - signs to array elements to get a specific sum
      - Clever reduction: Convert to subset sum by separating positive and negative numbers
      - Shows how seemingly different problems can map to the knapsack pattern
    
    - **Partition Equal Subset Sum**: Partition array into two subsets with equal sum
      - Application: Fair division of assets or workload
      - Insight: Special case of subset sum with target = sum/2

=== "Visualization"
    For the 0/1 Knapsack with values [60, 100, 120], weights [10, 20, 30], and capacity = 50:
    
    ```text
    dp table (rows = items considered, cols = capacity):
    
         | 0 | 1 | 2 | ... | 10 | ... | 20 | ... | 30 | ... | 50 |
    -----|---|---|---|-----|----|----|----|----|----|----|-----|
      0  | 0 | 0 | 0 | ... |  0 | ... |  0 | ... |  0 | ... |  0 |
    -----|---|---|---|-----|----|----|----|----|----|----|-----|
     [60]| 0 | 0 | 0 | ... | 60 | ... | 60 | ... | 60 | ... | 60 |
    -----|---|---|---|-----|----|----|----|----|----|----|-----|
    [100]| 0 | 0 | 0 | ... | 60 | ... |100 | ... |160 | ... |160 |
    -----|---|---|---|-----|----|----|----|----|----|----|-----|
    [120]| 0 | 0 | 0 | ... | 60 | ... |100 | ... |160 | ... |220 |
    ```
    
    The final answer is 220 (bottom-right cell).
    
    ![0/1 Knapsack Visualization](https://i.imgur.com/BbUXyYE.png)

=== "Implementation"
    **Recursive Implementation (with memoization)**:
    
    ```python
    def knapsack_memo(weights, values, n, capacity, memo={}):
        if (n, capacity) in memo:
            return memo[(n, capacity)]
        
        if n == 0 or capacity == 0:
            return 0
        
        if weights[n-1] > capacity:
            result = knapsack_memo(weights, values, n-1, capacity, memo)
        else:
            include = values[n-1] + knapsack_memo(weights, values, n-1, capacity - weights[n-1], memo)
            exclude = knapsack_memo(weights, values, n-1, capacity, memo)
            result = max(include, exclude)
        
        memo[(n, capacity)] = result
        return result
    ```
    
    **Standard Implementation (Tabulation)**:
    
    ```python
    def knapsack_01(values, weights, capacity):
        n = len(values)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(
                        values[i-1] + dp[i-1][w-weights[i-1]],  # Include item
                        dp[i-1][w]  # Exclude item
                    )
                else:
                    dp[i][w] = dp[i-1][w]  # Can't include item
        
        return dp[n][capacity]
    ```
    
    **Space-Optimized Version**:
    
    ```python
    def knapsack_01_optimized(values, weights, capacity):
        n = len(values)
        dp = [0] * (capacity + 1)
        
        for i in range(n):
            for w in range(capacity, weights[i]-1, -1):
                dp[w] = max(dp[w], values[i] + dp[w-weights[i]])
        
        return dp[capacity]
    ```
    
    **Subset Sum Implementation**:
    
    ```python
    def subset_sum(nums, target):
        n = len(nums)
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        
        # Base case: empty set can form sum 0
        for i in range(n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                if nums[i-1] <= j:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]
        
        return dp[n][target]
    ```
    
    **Equal Sum Partition**:
    
    ```python
    def can_partition(nums):
        total = sum(nums)
        
        # If sum is odd, cannot partition into equal sums
        if total % 2 != 0:
            return False
        
        target = total // 2
        return subset_sum(nums, target)
    ```

=== "Tips and Insights"
    - **Problem Recognition**: Look for problems involving selecting a subset of items with constraints
    - **State Definition**: `dp[i][w]` represents the maximum value achievable considering first i items with capacity w
    - **Loop Order**: For space-optimized version, always process capacity in reverse to avoid using items multiple times
    - **Related Problems**: 
      - Subset Sum (find if a subset can sum to target)
      - Partition Equal Subset Sum (divide array into two equal sum parts)
      - Target Sum (assign + or - to get target sum)
      - Minimum Subset Sum Difference (partition with minimum difference)
    - **Variations**:
      - Boolean version: Just check if something is possible vs. optimizing value
      - Count variations: Count number of ways instead of finding maximum value
      - Minimum variations: Find minimum weight/cost to achieve a target value
    - **Optimization Techniques**:
      - Use boolean arrays for subset sum problems
      - Reduce space complexity from O(n×W) to O(W) by using 1D arrays
      - Preprocess by sorting items when appropriate
    - **Time Complexity**: O(n×W) where n is number of items and W is the capacity
    - **Space Complexity**: O(n×W) for standard approach, O(W) for optimized
    - **Practical Considerations**:
      - For large weights, consider a different approach if actual weights aren't important
      - For large number of items with small weights, consider dynamic programming
      - For large number of items with large weights, consider branch and bound or approximation
    - **Common Mistakes**:
      - Incorrectly handling the case when an item's weight exceeds capacity
      - Forgetting to initialize the dp array properly
      - Using updated values in space-optimized version (always iterate capacity backwards)

## Pattern Recognition

The 0/1 Knapsack pattern appears when:

1. **Selection problems** with yes/no decisions for each item
2. **Optimization with constraints** (typically capacity or budget)
3. **Problems involving subset selection** to maximize or minimize some value
4. **Decision problems** about whether something is possible given constraints
def knapsack_memo(weights, values, n, capacity, memo={}):
    if (n, capacity) in memo:
        return memo[(n, capacity)]
    
    if n == 0 or capacity == 0:
        return 0
    
    if weights[n-1] > capacity:
        result = knapsack_memo(weights, values, n-1, capacity, memo)
    else:
        include = values[n-1] + knapsack_memo(weights, values, n-1, capacity - weights[n-1], memo)
        exclude = knapsack_memo(weights, values, n-1, capacity, memo)
        result = max(include, exclude)
    
    memo[(n, capacity)] = result
    return result
```

### Tabulation Solution

```python
def knapsack_dp(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    values[i-1] + dp[i-1][w - weights[i-1]],  # Include item
                    dp[i-1][w]  # Exclude item
                )
            else:
                dp[i][w] = dp[i-1][w]  # Can't include item
    
    return dp[n][capacity]
```

### Space Optimized Solution

```python
def knapsack_optimized(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # Traverse backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]
```

## Unbounded Knapsack

### Problem Statement

Given weights and values of n items and a knapsack of capacity W, find the maximum value that can be obtained. You can take unlimited quantities of each item.

### Solution

```python
def unbounded_knapsack(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for w in range(1, capacity + 1):
        for i in range(n):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]
```

## Variations and Applications

### Subset Sum

```python
def subset_sum(nums, target):
    """Check if there's a subset that sums to target"""
    dp = [False] * (target + 1)
    dp[0] = True  # Empty subset sums to 0
    
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    return dp[target]
```

### Partition Equal Subset Sum

```python
def can_partition(nums):
    """Check if array can be partitioned into two equal sum subsets"""
    total = sum(nums)
    if total % 2 != 0:
        return False
    
    target = total // 2
    return subset_sum(nums, target)
```

### Coin Change

```python
def coin_change(coins, amount):
    """Minimum number of coins to make amount"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

### Coin Change II

```python
def change(amount, coins):
    """Number of ways to make amount using coins"""
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]
```

## Time and Space Complexity

### 0/1 Knapsack

| Approach | Time | Space |
|----------|------|-------|
| Recursive | O(2^n) | O(n) |
| Memoization | O(n × W) | O(n × W) |
| Tabulation | O(n × W) | O(n × W) |
| Optimized | O(n × W) | O(W) |

### Unbounded Knapsack

- **Time Complexity**: O(n × W)
- **Space Complexity**: O(W)

## Practice Problems

- [ ] 0/1 Knapsack
- [ ] Subset Sum
- [ ] Partition Equal Subset Sum
- [ ] Target Sum
- [ ] Coin Change
- [ ] Coin Change II
- [ ] Combination Sum IV
- [ ] Perfect Squares
