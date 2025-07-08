# Unique Paths Pattern

## Introduction

The Unique Paths pattern is a fundamental grid-based dynamic programming pattern used to count paths or find optimal routes through 2D matrices with movement restrictions.

=== "Overview"
    **Core Idea**: Count the number of ways to reach a destination cell in a grid when movement is restricted (typically to right and down only).
    
    **When to Use**:
    
    - When counting different ways to navigate through a grid
    - When finding paths in a maze with restrictions
    - When working with robot movement planning problems
    - When calculating combinatorial path problems
    
    **Recurrence Relation**: `dp[i][j] = dp[i-1][j] + dp[i][j-1]` (sum of ways to reach the cell from above and from left)
    
    **Real-World Applications**:
    
    - Robot path planning in warehouse automation
    - Chip design and circuit routing algorithms
    - Network routing optimization
    - Game level design and pathfinding
    - Calculating possible chess piece movements

=== "Example Problems"
    - **Unique Paths**: Count ways for a robot to move from top-left to bottom-right of an m×n grid
      - Problem: Robot can only move down or right at each step
      - Mathematical insight: This is equivalent to choosing which steps should be "down" (or right) from all total steps
      - Formula connection: C(m+n-2, m-1) = (m+n-2)! / ((m-1)! * (n-1)!)
    
    - **Unique Paths II**: Same as above but with obstacles in certain cells
      - Variation: If a cell contains an obstacle, dp[i][j] = 0 (can't reach that cell)
      - Tests your ability to handle constraints in DP problems
    
    - **Minimum Path Sum**: Find the path with the smallest sum of numbers along the path
      - Variation: Instead of counting paths, we're optimizing the path cost
      - Shows how the same grid structure can be used for different optimization goals
    
    - **Maximum Path Sum**: Find the path with the largest sum of numbers along the path
      - Opposite optimization goal from Minimum Path Sum
      - Often appears in game scenarios where you want to collect maximum points

=== "Visualization"
    For a 3×3 grid, the number of unique paths to each cell:
    
    ```text
    ┌───┬───┬───┐
    │ 1 │ 1 │ 1 │
    ├───┼───┼───┤
    │ 1 │ 2 │ 3 │
    ├───┼───┼───┤
    │ 1 │ 3 │ 6 │
    └───┴───┴───┘
    ```
    
    The bottom-right cell shows there are 6 unique paths from top-left to bottom-right.
    
    With an obstacle (X) in the middle:
    
    ```text
    ┌───┬───┬───┐
    │ 1 │ 1 │ 1 │
    ├───┼───┼───┤
    │ 1 │ X │ 1 │
    ├───┼───┼───┤
    │ 1 │ 1 │ 2 │
    └───┴───┴───┘
    ```
    
    The obstacle blocks paths, reducing the total to 2.
    
    ![Unique Paths Visualization](https://i.imgur.com/jDGkm4g.png)

=== "Implementation"
    **Standard Implementation for Unique Paths**:
    
    ```python
    def uniquePaths(m, n):
        # Create a 2D DP table
        dp = [[1] * n for _ in range(m)]
        
        # Fill the DP table
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[m-1][n-1]
    
    # Time Complexity: O(m*n)
    # Space Complexity: O(m*n)
    ```
    
    **Implementation with Obstacles (Unique Paths II)**:
    
    ```python
    def uniquePathsWithObstacles(obstacleGrid):
        if not obstacleGrid or obstacleGrid[0][0] == 1:
            return 0
            
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        
        # Initialize first cell
        dp[0][0] = 1
        
        # Initialize first row
        for j in range(1, n):
            if obstacleGrid[0][j] == 0:
                dp[0][j] = dp[0][j-1]
        
        # Initialize first column
        for i in range(1, m):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = dp[i-1][0]
        
        # Fill the DP table
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[m-1][n-1]
    ```
    
    **Space-Optimized Version**:
    
    ```python
    def uniquePaths_optimized(m, n):
        # We only need to store the current and previous row
        prev_row = [1] * n
        
        for i in range(1, m):
            curr_row = [1] * n
            for j in range(1, n):
                curr_row[j] = curr_row[j-1] + prev_row[j]
            prev_row = curr_row
        
        return prev_row[n-1]
    
    # Time Complexity: O(m*n)
    # Space Complexity: O(n)
    ```

=== "Tips and Insights"
    - **Mathematical Formula**: The number of unique paths without obstacles can be calculated using the combination formula C(m+n-2, m-1)
    - **Common Mistake**: Forgetting to initialize the first row and column correctly, especially with obstacles
    - **Memory Optimization**: You only need to keep track of the previous row to calculate the current row
    - **Generalization**: This pattern can be extended to:
      - Allow diagonal moves (dp[i][j] += dp[i-1][j-1])
      - Handle obstacles (check before adding paths)
      - Optimize for different criteria (min/max sum, product, etc.)
    - **Visualization**: Drawing out small examples helps understand the pattern
    - **Recursive vs Iterative**: Iterative approach is usually more efficient for grid problems
    - **Connection to Pascal's Triangle**: The unique paths grid without obstacles forms a pattern similar to Pascal's triangle
    - **Direction Variations**: The same pattern works if you start from the bottom-right and move to the top-left
