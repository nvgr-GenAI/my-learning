# Multi-dimensional Knapsack Pattern

## Introduction

The Multi-dimensional Knapsack pattern is an extension of the classic knapsack problem where items have multiple constraint dimensions. This pattern is useful for complex resource allocation problems with multiple limited resources.

=== "Overview"
    **Core Idea**: Choose items to maximize value while respecting multiple capacity constraints simultaneously.
    
    **When to Use**:
    
    - When optimizing with multiple capacity constraints
    - When each item has multiple "weight" dimensions
    - When dealing with complex resource allocation problems
    - When resources are limited along multiple dimensions (time, space, budget, etc.)
    
    **Recurrence Relation**: 
    `dp[i][w1][w2]...[wn] = max(dp[i-1][w1][w2]...[wn], val[i-1] + dp[i-1][w1-wt1[i-1]][w2-wt2[i-1]]...[wn-wtn[i-1]])`
    
    **Real-World Applications**:
    
    - Cloud computing resource allocation (CPU, memory, network, etc.)
    - Project portfolio selection with multiple budget constraints
    - Manufacturing with multiple resource limitations
    - Logistics optimization with volume and weight constraints
    - Class scheduling with time and room constraints

=== "Example Problems"
    - **Multi-dimensional Knapsack Problem**: Items have multiple weight dimensions and multiple capacity constraints
      - Problem: Choose projects with values, but limited by both budget and manpower
      - Example: 4 projects with values [10, 40, 30, 50], costs [2, 4, 6, 8], and manpower [1, 3, 2, 4] with budget = 10 and manpower = 6
    
    - **Task Scheduling with Resource Constraints**: Schedule tasks to maximize value while respecting multiple resource constraints
      - Problem: Each task has a value and consumes multiple resources
      - Example: Server tasks with CPU, memory, and network bandwidth requirements
    
    - **Course Selection Problem**: Select courses to maximize credit value while respecting time constraints
      - Problem: Courses have credit values and time slots, can't take overlapping courses
      - Shows how time constraints can be modeled as multiple dimensions
    
    - **Constrained Job Scheduling**: Schedule jobs to maximize profit under time and resource constraints
      - Problem: Each job has a profit, deadline, and resource requirements
      - Real application: Scheduling in manufacturing systems

=== "Visualization"
    For a 2D knapsack with:
    - Values: [10, 40, 30, 50]
    - Weights1 (cost): [2, 4, 6, 8]
    - Weights2 (manpower): [1, 3, 2, 4]
    - Capacity1 (budget): 10
    - Capacity2 (manpower): 6
    
    ```text
    3D dp array visualization (slice for each item):
    
    dp[0][*][*] = 0 (no items)
    
    dp[1][*][*] = (after considering item 1 with value=10, cost=2, manpower=1)
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 10 10 10 10 10 10
    0 10 10 10 10 10 10
    ...
    
    dp[2][*][*] = (after considering items 1 & 2)
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 10 10 10 10 10 10
    0 10 10 10 10 10 10
    0 10 40 40 40 40 40
    0 10 40 40 40 40 40
    0 10 40 40 40 40 40
    ...
    
    Final result: dp[4][10][6] = 90 (items 2 and 4 with values 40+50)
    ```
    
    ![Multi-dimensional Knapsack Visualization](https://i.imgur.com/DiXJGxl.png)

=== "Implementation"
    **Standard Implementation (2D constraints)**:
    
    ```python
    def multi_dimensional_knapsack(values, weights1, weights2, capacity1, capacity2):
        n = len(values)
        dp = [[[0 for _ in range(capacity2 + 1)] 
              for _ in range(capacity1 + 1)] 
              for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w1 in range(capacity1 + 1):
                for w2 in range(capacity2 + 1):
                    # Skip this item if it doesn't fit either constraint
                    if weights1[i-1] <= w1 and weights2[i-1] <= w2:
                        dp[i][w1][w2] = max(
                            dp[i-1][w1][w2],  # Skip this item
                            values[i-1] + dp[i-1][w1-weights1[i-1]][w2-weights2[i-1]]  # Include this item
                        )
                    else:
                        dp[i][w1][w2] = dp[i-1][w1][w2]  # Can't include this item
        
        return dp[n][capacity1][capacity2]
    ```
    
    **Space-Optimized Version** (for 2D constraints):
    
    ```python
    def multi_dimensional_knapsack_optimized(values, weights1, weights2, capacity1, capacity2):
        n = len(values)
        dp = [[0 for _ in range(capacity2 + 1)] for _ in range(capacity1 + 1)]
        
        for i in range(n):
            # Important: traverse the weights in reverse to avoid using the item multiple times
            for w1 in range(capacity1, weights1[i]-1, -1):
                for w2 in range(capacity2, weights2[i]-1, -1):
                    dp[w1][w2] = max(dp[w1][w2], values[i] + dp[w1-weights1[i]][w2-weights2[i]])
        
        return dp[capacity1][capacity2]
    ```
    
    **Implementation for Three Dimensions**:
    
    ```python
    def three_dimensional_knapsack(values, weights1, weights2, weights3, capacity1, capacity2, capacity3):
        n = len(values)
        dp = [[0 for _ in range(capacity3 + 1)] for _ in range(capacity2 + 1)]
        prev_dp = [[0 for _ in range(capacity3 + 1)] for _ in range(capacity2 + 1)]
        
        for i in range(n):
            for w1 in range(capacity1 + 1):
                if w1 >= weights1[i]:
                    for w2 in range(capacity2 + 1):
                        if w2 >= weights2[i]:
                            for w3 in range(capacity3 + 1):
                                if w3 >= weights3[i]:
                                    dp[w2][w3] = max(prev_dp[w2][w3], 
                                                     values[i] + prev_dp[w2-weights2[i]][w3-weights3[i]])
                                else:
                                    dp[w2][w3] = prev_dp[w2][w3]
                        else:
                            for w3 in range(capacity3 + 1):
                                dp[w2][w3] = prev_dp[w2][w3]
            
            # Swap dp arrays for next iteration
            prev_dp, dp = dp, prev_dp
        
        return prev_dp[capacity2][capacity3]
    ```

=== "Tips and Insights"
    - **Dimensionality**: Each constraint adds a dimension to the DP table
    - **Space Complexity**: Grows exponentially with the number of dimensions
    - **Optimization Techniques**:
      - Use bit manipulation for small constraints
      - Apply heuristics for large problems
      - Consider approximation algorithms for very high dimensions
    - **State Representation**: For many dimensions, consider sparse matrix representations
    - **Common Variations**:
      - Binary constraints (item taken or not)
      - Multiple units of each item allowed (unbounded)
      - Dependent constraints (one constraint affects another)
    - **Practical Limitations**:
      - Beyond 3-4 dimensions, exact DP solutions become impractical
      - Consider decomposition or approximation techniques
    - **Algorithm Selection**:
      - Few dimensions: Standard DP approach
      - Many dimensions, small weights: Meet-in-the-middle
      - Large weights, few items: Branch and bound
    - **Real-world Modeling Tips**:
      - Focus on the most constraining dimensions first
      - Consider if some constraints can be combined
      - For large instances, consider linear programming relaxation

## Complexity Analysis

| Dimensions | Time Complexity | Space Complexity |
|------------|----------------|-----------------|
| 2D         | O(n * W1 * W2) | O(W1 * W2) or O(n * W1 * W2) |
| 3D         | O(n * W1 * W2 * W3) | O(W2 * W3) with optimization |
| kD         | O(n * W1 * ... * Wk) | Depends on optimization technique |

## Pattern Recognition

The Multi-dimensional Knapsack pattern appears when:

1. **Multiple resource constraints** must be satisfied simultaneously
2. **Complex decision making** with multiple factors
3. **Resource allocation** problems with multiple limited resources
