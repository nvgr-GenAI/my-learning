# Knapsack Problems

## 0/1 Knapsack

### Problem Statement

Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value.

### Recursive Solution

```python
def knapsack_recursive(weights, values, n, capacity):
    # Base case
    if n == 0 or capacity == 0:
        return 0
    
    # If weight exceeds capacity, skip item
    if weights[n-1] > capacity:
        return knapsack_recursive(weights, values, n-1, capacity)
    
    # Maximum of including or excluding the item
    include = values[n-1] + knapsack_recursive(weights, values, n-1, capacity - weights[n-1])
    exclude = knapsack_recursive(weights, values, n-1, capacity)
    
    return max(include, exclude)
```

### Memoization Solution

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
