# Matrix Chain Multiplication Pattern

## Introduction

The Matrix Chain Multiplication pattern is a classic dynamic programming approach for finding the most efficient way to perform a sequence of operations, particularly when the order of operations affects the cost but not the final result.

=== "Overview"
    **Core Idea**: Find the optimal way to perform a sequence of operations, typically by determining the order that minimizes cost.
    
    **When to Use**:
    
    - When the order of operations matters for performance but doesn't affect the result
    - When working with associative operations that have different costs based on order
    - When solving optimization problems over ranges or intervals
    - When the solution depends on breaking the problem at different positions
    
    **Recurrence Relation**: `dp[i][j] = min(dp[i][k] + dp[k+1][j] + cost(i,j,k))` for all k from i to j-1
    
    **Real-World Applications**:
    
    - Optimizing matrix multiplication sequences in linear algebra libraries
    - Query optimization in database systems
    - Expression evaluation optimization in compilers
    - Optimizing network packet fragmentation and reassembly
    - File merging operations in version control systems

=== "Example Problems"
    - **Matrix Chain Multiplication**: Find the most efficient way to multiply a sequence of matrices
      - Problem: Given dimensions of a sequence of matrices, find the minimum number of scalar multiplications
      - Example: Matrices with dimensions [10×30, 30×5, 5×60] → Optimal order: ((A×B)×C) with cost 4500
    
    - **Burst Balloons**: Burst balloons to maximize collected coins
      - Problem: Each balloon has coins, and when burst you get coins[left] × coins[i] × coins[right]
      - Clever approach: Think of it as the last balloon to burst in each interval
    
    - **Minimum Cost Tree from Leaf Values**: Build a tree where internal nodes are products of their children
      - Interval approach: Determine the optimal partitioning of the array
      - Shows how interval DP can solve problems with tree structures
    
    - **Optimal Binary Search Tree**: Construct a BST with minimum expected search cost
      - Problem: Given keys and their frequencies, arrange them to minimize average lookup time
      - Applications in compiler design for optimal switch statements
    
    - **Palindrome Partitioning**: Cut a string into palindromes with minimum cuts
      - Related interval problem: Find optimal cutting points

=== "Visualization"
    For Matrix Chain Multiplication with dimensions [10, 30, 5, 60]:
    
    ```text
    dp table (diagonal entries show costs for intervals of increasing length):
    
         | 0 | 1 | 2 | 3 |
    -----|---|---|---|---|
      0  | 0 | 1500| 4500|18000|
    -----|---|---|---|---|
      1  | - | 0  | 9000|10500|
    -----|---|---|---|---|
      2  | - | -  | 0  |18000|
    -----|---|---|---|---|
      3  | - | -  | -  | 0  |
    ```
    
    The minimum cost is 4500 (top-right cell), achieved by multiplying (A×B)×C.
    
    Optimal parenthesization: ((A×B)×C)
    
    For matrices with dimensions [10×30, 30×5, 5×60]:
    - A×B: 10×30×5 = 1500 multiplications
    - (A×B)×C: 1500 + 10×5×60 = 4500 multiplications
    
    Alternative order B×C first:
    - B×C: 30×5×60 = 9000 multiplications
    - A×(B×C): 9000 + 10×30×60 = 27000 multiplications
    
    ![Matrix Chain Multiplication](https://i.imgur.com/XFqRTmn.png)

=== "Implementation"
    **Standard Implementation**:
    
    ```python
    def matrixChainMultiplication(dimensions):
        n = len(dimensions) - 1  # Number of matrices
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        # s[i][j] stores the optimal split point k for the subproblem dimensions[i:j+1]
        s = [[0] * (n + 1) for _ in range(n + 1)]
        
        # Length is the chain length being considered
        for length in range(2, n + 1):
            for i in range(1, n - length + 2):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                # Try each possible split point and find the minimum
                for k in range(i, j):
                    cost = dp[i][k] + dp[k+1][j] + dimensions[i-1] * dimensions[k] * dimensions[j]
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        s[i][j] = k
        
        return dp[1][n]
    
    # Time Complexity: O(n³)
    # Space Complexity: O(n²)
    ```
    
    **Retrieving the Optimal Parenthesization**:
    
    ```python
    def print_optimal_parenthesization(s, i, j):
        if i == j:
            return f"A{i}"
        else:
            k = s[i][j]
            left = print_optimal_parenthesization(s, i, k)
            right = print_optimal_parenthesization(s, k+1, j)
            return f"({left}×{right})"
    
    # Example usage:
    # dimensions = [10, 30, 5, 60]
    # n = len(dimensions) - 1
    # dp, s = matrixChainMultiplication(dimensions)
    # print(print_optimal_parenthesization(s, 1, n))  # Should output: "((A1×A2)×A3)"
    ```
    
    **Burst Balloons Implementation** (similar pattern):
    
    ```python
    def maxCoins(nums):
        # Add boundary balloons with value 1
        nums = [1] + nums + [1]
        n = len(nums)
        
        # dp[i][j] = maximum coins obtainable from bursting all balloons from i to j
        dp = [[0] * n for _ in range(n)]
        
        # Length is the chain length being considered
        for length in range(1, n-1):
            for left in range(1, n - length):
                right = left + length - 1
                
                # Last balloon to burst in range [left, right]
                for last in range(left, right + 1):
                    # Coins for bursting the last balloon
                    coins = nums[left-1] * nums[last] * nums[right+1]
                    # Add coins from subproblems
                    if last > left:
                        coins += dp[left][last-1]
                    if last < right:
                        coins += dp[last+1][right]
                    dp[left][right] = max(dp[left][right], coins)
        
        return dp[1][n-2]
    ```

=== "Tips and Insights"
    - **Parenthesization Pattern**: This pattern is about finding optimal ways to parenthesize a sequence of operations
    - **Intervals**: Always process smaller intervals before larger ones (bottom-up)
    - **Split Points**: The key is finding the optimal split point (k) for each interval
    - **Optimal Substructure**: If k is the optimal split for [i,j], then we already have optimal solutions for [i,k] and [k+1,j]
    - **Visualization**: Drawing tables helps understand the pattern, with diagonal entries representing costs for intervals
    - **Common Mistake**: Forgetting to add the cost of combining results from subproblems
    - **State Definition**: dp[i][j] typically represents the optimal cost for the subproblem from index i to j
    - **Direction**: Often filled diagonally in the matrix (by increasing chain length)
    - **Beyond Matrices**: This pattern applies to any associative operation where order affects cost
    - **Reconstruction**: Use a separate matrix to track optimal split points for reconstructing the solution
