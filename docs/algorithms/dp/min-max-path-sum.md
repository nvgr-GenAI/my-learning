# Minimum/Maximum Path Sum Pattern

## Introduction

The Minimum/Maximum Path Sum pattern is a dynamic programming approach used to find optimal paths through grids based on cumulative values.

=== "Overview"
    **Core Idea**: Find the path through a grid that optimizes (minimizes or maximizes) the sum of values along the path.
    
    **When to Use**:
    
    - When finding the optimal path through a cost or value grid
    - When solving problems involving resource collection or cost minimization in a grid
    - When dealing with grid-based games where score matters
    - When working with terrain navigation where different paths have different costs
    
    **Recurrence Relation**:
    
    - For minimization: `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`
    - For maximization: `dp[i][j] = grid[i][j] + max(dp[i-1][j], dp[i][j-1])`
    
    **Real-World Applications**:
    
    - Optimizing delivery routes in a city grid
    - Finding least-cost paths in network routing
    - Terrain navigation algorithms
    - Image processing techniques like seam carving
    - Resource gathering in strategy games

=== "Example Problems"
    - **Minimum Path Sum**: Find path from top-left to bottom-right with minimum sum of numbers
      - Classic application: Cost optimization in grid traversal
      - Practical usage: Routing in networks with varying costs
    
    - **Maximum Path Sum**: Similar to above, but maximizing the sum
      - Variation: Collecting maximum value in resource gathering games
      - Challenge: May need to handle negative values
    
    - **Dungeon Game**: Calculate minimum initial health needed to reach bottom-right
      - Interesting twist: Working backwards from the destination
      - Shows how DP direction can sometimes be reversed for certain problems
    
    - **Triangle**: Find the minimum path sum from top to bottom in a triangle
      - Variant: Different grid structure (triangle) but same optimization concept
      - Can be solved bottom-up to avoid edge cases
    
    - **Cherry Pickup**: Collect maximum cherries in a grid making two passes
      - Advanced variation: Coordinate two paths through the same grid
      - Shows how multiple passes can be modeled in a single DP solution

=== "Visualization"
    For a grid with values:
    
    ```text
    ┌───┬───┬───┐
    │ 1 │ 3 │ 1 │
    ├───┼───┼───┤
    │ 1 │ 5 │ 1 │
    ├───┼───┼───┤
    │ 4 │ 2 │ 1 │
    └───┴───┴───┘
    ```
    
    The minimum path sum DP table becomes:
    
    ```text
    ┌───┬───┬───┐
    │ 1 │ 4 │ 5 │
    ├───┼───┼───┤
    │ 2 │ 7 │ 6 │
    ├───┼───┼───┤
    │ 6 │ 8 │ 7 │
    └───┴───┴───┘
    ```
    
    The minimum path sum is 7 (path: 1→1→1→1→1→2).
    
    Triangle problem example:
    
    ```text
       2
      3 4
     6 5 7
    4 1 8 3
    ```
    
    Bottom-up DP solution (each cell is the minimum sum ending at that position):
    
    ```text
       2 (2)
      3 4 (5,6)
     6 5 7 (7,10,13)
    4 1 8 3 (4,1,8,3)
    ```
    
    The minimum path sum is 7 (path: 2→3→1→1).
    
    ![Min Path Sum Visualization](https://i.imgur.com/WNVxKhi.png)

=== "Implementation"
    **Minimum Path Sum Implementation**:
    
    ```python
    def minPathSum(grid):
        if not grid:
            return 0
        
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = grid[0][0]
        
        # Fill first row
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        
        # Fill first column
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        
        # Fill the rest of the grid
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
        
        return dp[m-1][n-1]
    
    # Time Complexity: O(m*n)
    # Space Complexity: O(m*n)
    ```
    
    **Space-Optimized Version**:
    
    ```python
    def minPathSum_optimized(grid):
        if not grid:
            return 0
        
        m, n = len(grid), len(grid[0])
        dp = [0] * n
        dp[0] = grid[0][0]
        
        # Initialize first row
        for j in range(1, n):
            dp[j] = dp[j-1] + grid[0][j]
        
        # Process remaining rows
        for i in range(1, m):
            dp[0] += grid[i][0]  # Update first element
            for j in range(1, n):
                dp[j] = grid[i][j] + min(dp[j], dp[j-1])
        
        return dp[n-1]
    
    # Time Complexity: O(m*n)
    # Space Complexity: O(n)
    ```
    
    **Triangle Problem Solution** (bottom-up):
    
    ```python
    def minimumTotal(triangle):
        n = len(triangle)
        # Start from the bottom row
        dp = triangle[-1][:]
        
        # Work our way up
        for i in range(n-2, -1, -1):
            for j in range(len(triangle[i])):
                dp[j] = triangle[i][j] + min(dp[j], dp[j+1])
        
        return dp[0]
    
    # Time Complexity: O(n²)
    # Space Complexity: O(n)
    ```

=== "Tips and Insights"
    - **Direction Matters**: For some problems, working from bottom-right to top-left is easier
    - **Space Optimization**: You usually only need one row or column of the previous state
    - **Path Reconstruction**: To find the actual path, maintain a separate matrix to track decisions
    - **Out of Bounds**: Be careful with grid boundaries, especially in problems with unusual shapes like triangles
    - **Initialization**: First row and first column need special handling
    - **Four-Direction Movement**: For problems allowing up/down/left/right movement, consider using BFS with a priority queue (Dijkstra)
    - **Multiple Passes**: Some problems require making multiple passes through the grid
    - **Alternative Goals**: Instead of sum, the objective might be product, difference, or other metrics
    - **State Compression**: For 3D grids or multiple agents, consider compressing states to reduce dimensions
    - **Memoization**: For irregular grids, a top-down memoization approach might be cleaner than bottom-up
