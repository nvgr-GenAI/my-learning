# Unbounded Knapsack Pattern

## Introduction

The Unbounded Knapsack pattern is an extension of the 0/1 Knapsack problem where you can use items multiple times. This pattern is particularly useful for problems involving resource allocation with repeatable choices.

=== "Overview"
    **Core Idea**: Choose items to maximize value while respecting a weight constraint, with each item available in unlimited quantity.
    
    **When to Use**:
    
    - When items can be selected multiple times
    - When dealing with problems involving repetitive choices
    - When optimizing with unlimited supply of resources
    - When solving problems related to coin change or cutting optimization
    
    **Recurrence Relation**: `dp[w] = max(dp[w], dp[w-wt[i]] + val[i])` for each item i
    
    **Real-World Applications**:
    
    - Currency exchange and denominations problems
    - Manufacturing with repeatable processes
    - Stock trading with unlimited shares
    - Resource allocation with renewable resources
    - Cutting stock problems in manufacturing

=== "Example Problems"
    - **Unbounded Knapsack**: Maximize value by selecting items with unlimited supply
      - Problem: Given weights and values of n items, find the maximum value with items that can be used multiple times
      - Example: Items with values [10, 30, 20] and weights [5, 10, 15], capacity = 100 → Maximum value = 300 (10 items of value 30)
    
    - **Rod Cutting**: Cut a rod into pieces to maximize profit
      - Problem: Given a rod of length n and prices for different lengths, maximize profit
      - Example: Rod length 8, prices [1,5,8,9,10,17,17,20] → Max value = 22 (cut into pieces of length 2+6 or 2+2+4)
    
    - **Coin Change**: Find minimum number of coins that make a given amount
      - Problem: Find minimum coins needed to make amount n with given coin denominations
      - Example: Coins [1,2,5], amount 11 → 3 coins (5+5+1)
    
    - **Coin Change II**: Count the number of ways to make a given amount
      - Variation: Instead of minimizing coins, count all possible combinations
      - Example: Coins [1,2,5], amount 5 → 4 ways (5), (2+2+1), (2+1+1+1), (1+1+1+1+1)
    
    - **Integer Break**: Break a number into sum of integers to maximize their product
      - Problem: Split n into k integers (k ≥ 2) to maximize product
      - Example: n=10 → 3+3+4 = 36 (max product)

=== "Visualization"
    For Coin Change ways with coins [1,2,5] and amount = 5:
    
    ```text
    dp array (index = amount, value = ways to make that amount):
    
    [1, 1, 2, 2, 3, 4]
     0  1  2  3  4  5
    
    For amount = 0: 1 way (use no coins)
    For amount = 1: 1 way (one 1-coin)
    For amount = 2: 2 ways (two 1-coins or one 2-coin)
    For amount = 3: 2 ways (three 1-coins or one 1-coin + one 2-coin)
    For amount = 4: 3 ways (four 1-coins, two 1-coins + one 2-coin, or two 2-coins)
    For amount = 5: 4 ways (five 1-coins, three 1-coins + one 2-coin, one 1-coin + two 2-coins, or one 5-coin)
    ```
    
    ![Unbounded Knapsack Visualization](https://i.imgur.com/reMtlWX.png)

=== "Implementation"
    **Standard Implementation**:
    
    ```python
    def unbounded_knapsack(values, weights, capacity):
        dp = [0] * (capacity + 1)
        
        for w in range(capacity + 1):
            for i in range(len(values)):
                if weights[i] <= w:
                    dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
        
        return dp[capacity]
    ```
    
    **More Efficient Implementation** (process by item first):
    
    ```python
    def unbounded_knapsack_efficient(values, weights, capacity):
        dp = [0] * (capacity + 1)
        
        for i in range(len(values)):
            for w in range(weights[i], capacity + 1):
                dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
        
        return dp[capacity]
    ```
    
    **Coin Change (Minimum Coins) Implementation**:
    
    ```python
    def coin_change_min(coins, amount):
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    ```
    
    **Coin Change (Ways to Make Amount) Implementation**:
    
    ```python
    def coin_change_ways(coins, amount):
        dp = [0] * (amount + 1)
        dp[0] = 1  # Base case: 1 way to make amount 0
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        
        return dp[amount]
    ```

=== "Tips and Insights"
    - **Key Difference from 0/1 Knapsack**: Items can be used multiple times
    - **Loop Order Matters**: 
      - For 0/1 Knapsack: Process items in outer loop, capacity in reverse inner loop
      - For Unbounded: Can process capacity in outer loop and items in inner loop, or vice versa
    - **State Definition**: `dp[w]` typically represents the maximum value achievable with capacity w
    - **Initialization**: `dp[0] = 0` (base case: 0 value with 0 capacity)
    - **Variations**:
      - Maximize value (standard unbounded knapsack)
      - Minimize items (coin change minimum)
      - Count ways (coin change ways)
    - **Complexity**:
      - Time: O(n*W) where n is number of items and W is capacity
      - Space: O(W) - only need a 1D array
    - **Problem Recognition**: Look for:
      - Unlimited usage of items/resources
      - Terms like "as many times as you want"
      - Minimizing or maximizing with repeated choices
    - **Common Mistake**: Using a 2D array like in 0/1 Knapsack (unnecessary for unbounded)
    - **Optimization**: When the weights are large but the number of items is small, consider a different approach like branch and bound

## Comparison with 0/1 Knapsack

| Feature | 0/1 Knapsack | Unbounded Knapsack |
|---------|-------------|-------------------|
| Item usage | At most once | Unlimited |
| DP state | dp[i][w] | dp[w] |
| Space complexity | O(n*W) or O(W) | O(W) |
| Loop order (optimized) | Items outer, capacity inner reverse | Either way works |
| Applications | Portfolio optimization, Project selection | Currency exchange, Resource allocation |

## Pattern Recognition

The Unbounded Knapsack pattern appears when:

1. **Unlimited item selection**
2. **Optimization with repetition allowed**
3. **Resource allocation with renewable resources**
