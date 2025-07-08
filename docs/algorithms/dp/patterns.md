# Dynamic Programming Patterns

## Introduction

Dynamic Programming (DP) is a method for solving complex problems by breaking them down into simpler subproblems. Rather than solving overlapping subproblems repeatedly, DP stores solutions to avoid redundant calculations, dramatically improving efficiency.

### When to Use Dynamic Programming

DP is appropriate when a problem has these two key properties:

1. **Overlapping subproblems** - The same smaller problems are solved multiple times
2. **Optimal substructure** - The optimal solution can be constructed from optimal solutions of its subproblems

### Recognizing DP Problems

Consider using dynamic programming when you encounter:

- Problems asking for optimization (maximum/minimum)
- Problems requiring counting the number of ways to do something
- Problems involving making sequences of choices to reach a goal
- Problems where you need to examine all possible scenarios

### General Approach to Solving DP Problems

1. **Define the state**: What information do we need to represent a subproblem?
2. **Formulate recurrence relation**: How does the current state relate to previous states?
3. **Identify base cases**: What are the simplest scenarios we can solve directly?
4. **Determine the computation order**: Bottom-up or top-down?
5. **Implement the solution**: Code the solution with attention to efficiency
6. **Optimize if needed**: Reduce space complexity if possible

This guide categorizes common DP patterns to help identify and solve problems efficiently. Each pattern has its own detailed page with comprehensive explanations, examples, visualizations, and implementation templates.

## Pattern Categories

Each pattern below links to a detailed page with comprehensive explanations, examples, and implementations.

### 1. Linear Sequence Patterns

Linear sequence patterns deal with problems where decisions are made at each element in a sequence, and the solution is built incrementally. The state typically depends on a fixed number of previous states.

- [Fibonacci Sequence Pattern](fibonacci.md) - Current state depends on the sum or function of a fixed number of previous states
- [Longest Increasing Subsequence (LIS)](lis.md) - Find subsequence of elements that are in increasing order

=== "Overview"
    **Core Idea**: Current state depends on the sum or some function of a fixed number of previous states.
    
    **When to Use**:
    
    - When each new value in a sequence depends on the previous k values
    - When calculating the nth term of a sequence with fixed recurrence relation
    - When there are multiple ways to reach the current state by combining previous states
    
    **Recurrence Relation**: `dp[i] = dp[i-1] + dp[i-2]`
    
    **Real-World Applications**:
    
    - Growth of a rabbit population where adults produce new pairs
    - Calculating ways to ascend stairs taking 1 or 2 steps at a time
    - Determining ways to place dominos on a board of size n

=== "Example Problems"
    - **Fibonacci Numbers**: Calculate the nth Fibonacci number
      - Problem: F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2) for n > 1
      - DP solution directly implements the recurrence relation
    
    - **Climbing Stairs**: Count ways to reach the top taking 1 or 2 steps at a time
      - Problem: To reach stair n, you can come from stair n-1 (taking 1 step) or from stair n-2 (taking 2 steps)
      - Recurrence: `dp[i] = dp[i-1] + dp[i-2]`
    
    - **House Robber**: Maximum money you can rob without taking from adjacent houses
      - Problem: You can't rob consecutive houses, so you need to decide which ones to rob
      - Recurrence: `dp[i] = max(dp[i-1], dp[i-2] + nums[i])` (either skip current house or rob it and skip previous)

=== "Visualization"
    For the Climbing Stairs problem:
    ```
    n = 4 stairs
    dp[1] = 1 (1 way: take one step)
    dp[2] = 2 (2 ways: take two 1-steps or take one 2-step)
    dp[3] = dp[2] + dp[1] = 3 (ways to reach stair 2 + ways to reach stair 1)
    dp[4] = dp[3] + dp[2] = 5 (ways to reach stair 3 + ways to reach stair 2)
    ```
    
    ![Fibonacci Pattern Visualization](https://i.imgur.com/FmOEybj.png)

=== "Implementation"
    **Standard Implementation**:
    
    ```python
    def fibonacci_pattern(n):
        if n <= 1:
            return n
        
        dp = [0] * (n + 1)
        dp[0], dp[1] = 0, 1
        
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n]
    ```
    
    **Space-Optimized Version**:
    
    ```python
    def fibonacci_optimized(n):
        if n <= 1:
            return n
        
        prev2, prev1 = 0, 1
        
        for _ in range(2, n + 1):
            curr = prev1 + prev2
            prev2, prev1 = prev1, curr
        
        return prev1
    ```

### 1.2 Longest Increasing Subsequence (LIS) Pattern

=== "Overview"
    **Core Idea**: Find a subsequence (not necessarily contiguous) of elements that are in increasing order.
    
    **When to Use**:
    
    - When you need to find a subsequence with elements in a specific order
    - When looking for the longest chain of elements with a certain property
    - When the problem involves finding an optimal subsequence with ordering constraints
    
    **Recurrence Relation**: `dp[i] = max(dp[j] + 1)` for all `j < i` where `nums[j] < nums[i]`
    
    **Real-World Applications**:
    
    - Finding the longest chain of activities that can be performed in order
    - Determining the maximum length of nested boxes or envelopes
    - Identifying the longest sequence of increasing temperatures

=== "Example Problems"
    - **Longest Increasing Subsequence**: Find the length of the longest subsequence that is strictly increasing
      - Problem: Given an array, find a subsequence where each element is larger than the previous
      - DP Solution: `dp[i]` represents the length of LIS ending at index i
    
    - **Maximum Sum Increasing Subsequence**: Find the increasing subsequence with the maximum possible sum
      - Problem: Like LIS, but we want to maximize the sum rather than just the length
      - Recurrence: `dp[i] = max(dp[j] + nums[i])` for all `j < i` where `nums[j] < nums[i]`
    
    - **Longest Chain of Pairs**: Given pairs of numbers (a,b), find the longest chain such that b of one pair < a of next pair
      - Problem: Sort by second element, then apply LIS pattern
      - Demonstrates how LIS can be applied to more complex ordering relationships

=== "Visualization"
    For the LIS problem with array `[10, 9, 2, 5, 3, 7, 101, 18]`:
    
    ```
    Array: [10, 9, 2, 5, 3, 7, 101, 18]
    dp[0] = 1 (just the element 10)
    dp[1] = 1 (just the element 9)
    dp[2] = 1 (just the element 2)
    dp[3] = 2 (elements [2, 5])
    dp[4] = 2 (elements [2, 3])
    dp[5] = 3 (elements [2, 3, 7])
    dp[6] = 4 (elements [2, 3, 7, 101])
    dp[7] = 4 (elements [2, 3, 7, 18])
    ```
    
    The LIS is of length 4 (e.g., [2, 3, 7, 101] or [2, 3, 7, 18])
    
    ![LIS Pattern Visualization](https://i.imgur.com/UCM2RuT.png)

=== "Implementation"
    **Standard Implementation**:
    
    ```python
    def lis_pattern(nums):
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n  # Minimum length is 1
        
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    ```
    
    **Binary Search Optimization** (O(n log n)):
    
    ```python
    def lis_optimized(nums):
        if not nums:
            return 0
        
        tails = []
        
        for num in nums:
            idx = bisect_left(tails, num)
            if idx == len(tails):
                tails.append(num)
            else:
                tails[idx] = num
        
        return len(tails)
    ```

### 2. Two-Sequence Patterns

Two-sequence patterns deal with problems involving two arrays or strings where we need to find relationships, similarities, or transformations between them.

- [Longest Common Subsequence (LCS)](lcs.md) - Find the longest subsequence that appears in both sequences
- [Edit Distance](edit-distance.md) - Calculate minimum operations required to transform one string into another

=== "Overview"
    **Core Idea**: Find the longest subsequence (not necessarily contiguous) that appears in both sequences in the same relative order.
    
    **When to Use**:
    
    - When comparing similarities between two sequences
    - When finding elements that appear in both sequences in the same order
    - When problems involve string or array matching with gaps allowed
    - When needing to find the minimum operations to transform one sequence to another
    
    **Recurrence Relation**:
    
    - If characters match: `dp[i][j] = 1 + dp[i-1][j-1]`
    - If not: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`
    
    **Real-World Applications**:
    
    - DNA sequence alignment in bioinformatics
    - File difference algorithms
    - Plagiarism detection systems
    - Autocorrect and text suggestion features

=== "Example Problems"
    - **Longest Common Subsequence**: Find the longest subsequence common to two strings
      - Problem: Given strings "abcde" and "ace", the LCS is "ace" with length 3
      - Insight: We build the solution incrementally, matching characters when possible
    
    - **Shortest Common Supersequence**: Find the shortest string that has both input strings as subsequences
      - Problem: For "abac" and "cab", the shortest supersequence is "cabac" with length 5
      - Solution: Find LCS first, then merge both strings by including the LCS only once
    
    - **Delete Operation for Two Strings**: Find minimum number of deletions to make two strings equal
      - Problem: To make "sea" and "eat" equal, delete 's' from first and 't' from second
      - Insight: Delete all characters not in LCS (length = total length - 2*LCS length)
    
    - **Minimum ASCII Delete Sum**: Delete characters to make strings equal, minimizing ASCII sum
      - Variation: Instead of counting deletions, we consider the ASCII values
      - Shows how the LCS pattern can be adapted to different cost metrics

=== "Visualization"
    For the LCS of "abcde" and "ace":
    
    ```text
        | "" | a | c | e |
    --------------------- 
    "" | 0  | 0 | 0 | 0 |
    --------------------- 
    a  | 0  | 1 | 1 | 1 |
    ---------------------
    b  | 0  | 1 | 1 | 1 |
    ---------------------
    c  | 0  | 1 | 2 | 2 |
    ---------------------
    d  | 0  | 1 | 2 | 2 |
    ---------------------
    e  | 0  | 1 | 2 | 3 |
    ```
    
    The table shows dp[i][j] = length of LCS for first i characters of string1 and first j characters of string2.
    
    ![LCS Pattern Visualization](https://i.imgur.com/RkoBlS5.png)

=== "Implementation"
    ```python
    def lcs_pattern(text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    ```
    
    **With Backtracking to Reconstruct LCS**:
    
    ```python
    def lcs_with_reconstruction(text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the dp table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Reconstruct the LCS
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if text1[i-1] == text2[j-1]:
                lcs.append(text1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
                
        return ''.join(reversed(lcs))
    ```

### 2.2 Edit Distance Pattern

=== "Overview"
    **Core Idea**: Calculate the minimum number of operations required to transform one string into another.
    
    **When to Use**:
    
    - When comparing similarity between strings or sequences
    - When you need to find the minimum transformations between sequences
    - When implementing features like autocorrect, spell checking, or DNA sequence comparison
    - When you need to measure how different two strings are
    
    **Recurrence Relation**:
    
    - If characters match: `dp[i][j] = dp[i-1][j-1]` (no operation needed)
    - If not: `dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])` (insert, delete, replace)
    
    **Real-World Applications**:
    
    - Spell checking and autocorrect algorithms
    - DNA sequence alignment in computational biology
    - Plagiarism detection systems
    - Natural language processing for text similarity

=== "Example Problems"
    - **Edit Distance (Levenshtein Distance)**: Find minimum operations to convert one string to another
      - Problem: Convert "horse" to "ros" using minimum operations
      - Solution: Delete 'h', replace 'r' with 'o', delete 's' and 'e' = 3 operations
    
    - **One Edit Distance**: Determine if two strings are one edit away from each other
      - Problem: Check if strings differ by at most one edit operation
      - Application: Used in "Did you mean?" suggestions in search engines
    
    - **Delete Operation for Two Strings**: Find minimum number of characters to delete to make two strings equal
      - Variation: Only deletion operations are allowed
      - Solution approach: Find LCS and delete characters not in it
    
    - **Word Break**: Determine if a string can be segmented into dictionary words
      - Problem: Given a string and a dictionary of words, can the string be split into words?
      - Approach: Use DP to track which prefixes can be segmented

=== "Visualization"
    For the Edit Distance between "horse" and "ros":
    
    ```text
        |   | r | o | s |
    --------------------- 
        | 0 | 1 | 2 | 3 |
    --------------------- 
    h   | 1 | 1 | 2 | 3 |
    ---------------------
    o   | 2 | 2 | 1 | 2 |
    ---------------------
    r   | 3 | 2 | 2 | 2 |
    ---------------------
    s   | 4 | 3 | 3 | 2 |
    ---------------------
    e   | 5 | 4 | 4 | 3 |
    ```
    
    The final edit distance is 3 (bottom-right cell).
    
    ![Edit Distance Visualization](https://i.imgur.com/JQSxz7j.png)

=== "Implementation"
    **Standard Implementation**:
    
    ```python
    def edit_distance_pattern(word1, word2):
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base cases
        for i in range(m + 1):
            dp[i][0] = i  # Delete operations
        for j in range(n + 1):
            dp[0][j] = j  # Insert operations
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Delete
                        dp[i][j-1],    # Insert
                        dp[i-1][j-1]   # Replace
                    )
        
        return dp[m][n]
    ```
    
    **Space-Optimized Implementation**:
    
    ```python
    def edit_distance_optimized(word1, word2):
        m, n = len(word1), len(word2)
        
        # Ensure word1 is shorter for space optimization
        if m > n:
            word1, word2 = word2, word1
            m, n = n, m
        
        # Previous and current row
        prev_row = list(range(n + 1))
        curr_row = [0] * (n + 1)
        
        for i in range(1, m + 1):
            curr_row[0] = i
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    curr_row[j] = prev_row[j-1]
                else:
                    curr_row[j] = 1 + min(
                        prev_row[j],    # Delete
                        curr_row[j-1],  # Insert
                        prev_row[j-1]   # Replace
                    )
            prev_row, curr_row = curr_row, [0] * (n + 1)
        
        return prev_row[n]
    ```

### 3. Grid Patterns

Grid patterns deal with problems involving 2D matrices or grids where you typically need to find a path or count paths between cells. These patterns are common in robotics, game development, and optimization problems.

- [Unique Paths](unique-paths.md) - Count ways to reach a destination in a grid with restricted movements
- [Minimum/Maximum Path Sum](min-max-path-sum.md) - Find optimal path through a grid to minimize or maximize values

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

=== "Visualization"
    For a 3×3 grid, the number of unique paths to each cell:
    
    ```
    ┌───┬───┬───┐
    │ 1 │ 1 │ 1 │
    ├───┼───┼───┤
    │ 1 │ 2 │ 3 │
    ├───┼───┼───┤
    │ 1 │ 3 │ 6 │
    └───┴───┴───┘
    ```
    
    The bottom-right cell shows there are 6 unique paths from top-left to bottom-right.
    
    ![Unique Paths Visualization](https://i.imgur.com/jDGkm4g.png)

=== "Implementation Template"
    ```python
    def unique_paths_pattern(m, n):
        dp = [[1] * n for _ in range(m)]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[m-1][n-1]
    ```
    
    **Space-Optimized Version**:
    
    ```python
    def unique_paths_optimized(m, n):
        # We only need to store the current and previous row
        prev_row = [1] * n
        
        for i in range(1, m):
            curr_row = [1] * n
            for j in range(1, n):
                curr_row[j] = curr_row[j-1] + prev_row[j]
            prev_row = curr_row
        
        return prev_row[n-1]
    ```

### 3.2 Minimum/Maximum Path Sum Pattern

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
    
    The minimum path sum is 7.
    
    ![Min Path Sum Visualization](https://i.imgur.com/WNVxKhi.png)

=== "Implementation Template"
    **Standard Implementation**:
    
    ```python
    def min_path_sum_pattern(grid):
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
    ```
    
    **Space-Optimized Version**:
    
    ```python
    def min_path_sum_optimized(grid):
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
    ```

### 4. Knapsack Patterns

Knapsack patterns are among the most versatile and widely applicable DP patterns. They deal with resource allocation problems where you have constraints and need to maximize or minimize some value.

- [0/1 Knapsack](knapsack.md) - Choose items to maximize value with weight constraint, each item used at most once
- [Unbounded Knapsack](unbounded-knapsack.md) - Similar to 0/1 knapsack but items can be used multiple times
- [Multi-dimensional Knapsack](multi-dimensional-knapsack.md) - Extension with multiple constraints

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

=== "Implementation Template"
    **Standard Implementation**:
    
    ```python
    def knapsack_01_pattern(values, weights, capacity):
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

### 4.2 Unbounded Knapsack Pattern

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

=== "Example Problems"
    - **Unbounded Knapsack**: Maximize value by selecting items with unlimited supply
      - Difference from 0/1: Each item can be used multiple times
      - Example: Same values and weights as 0/1, but can reuse items → potentially higher value
    
    - **Rod Cutting**: Cut a rod into pieces to maximize profit
      - Problem: Given a rod of length n and prices for different lengths, maximize profit
      - Example: Rod length 8, prices [1,5,8,9,10,17,17,20] → Max value = 22
    
    - **Coin Change**: Find minimum number of coins that make a given amount
      - Problem: Find minimum coins needed to make amount n with given coin denominations
      - Example: Coins [1,2,5], amount 11 → 3 coins (5+5+1)
    
    - **Coin Change II**: Count the number of ways to make a given amount
      - Variation: Instead of minimizing coins, count all possible ways
      - Shows how the same base pattern can answer different questions
    
    - **Integer Break**: Break a number into sum of integers to maximize their product
      - Problem: Split n into k integers (k ≥ 2) to maximize product
      - Example: n=10 → 3+3+4 = 36 (max product)

=== "Visualization"
    For Coin Change with coins [1,2,5] and amount = 11:
    
    ```text
    dp array (index = amount, value = ways to make that amount):
    
    [1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 10, 11]
     0  1  2  3  4  5  6  7  8  9  10  11
    ```
    
    There are 11 ways to make amount 11.
    
    ![Unbounded Knapsack Visualization](https://i.imgur.com/reMtlWX.png)

=== "Implementation Template"
    **Standard Implementation**:
    
    ```python
    def unbounded_knapsack_pattern(values, weights, capacity):
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
    
    **Coin Change Variation**:
    
    ```python
    def coin_change_ways(coins, amount):
        dp = [0] * (amount + 1)
        dp[0] = 1  # Base case: 1 way to make amount 0
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        
        return dp[amount]
    ```

### 4.3 Multi-dimensional Knapsack

=== "Overview"
    **Core Idea**: Extension of the knapsack problem with multiple constraints.
    
    **When to Use**:
    
    - When optimizing with multiple capacity constraints
    - When each item has multiple "weight" dimensions
    - When dealing with complex resource allocation problems
    - When resources are limited along multiple dimensions (time, space, budget, etc.)
    
    **Recurrence Relation**: `dp[i][w1][w2]...[wn] = max(dp[i-1][w1][w2]...[wn], val[i-1] + dp[i-1][w1-wt1[i-1]][w2-wt2[i-1]]...[wn-wtn[i-1]])`
    
    **Real-World Applications**:
    
    - Resource allocation in cloud computing (CPU, memory, network, etc.)
    - Project portfolio selection with multiple budget constraints
    - Manufacturing with multiple resource limitations
    - Logistics optimization with volume and weight constraints

=== "Example Problems"
    - **Multi-dimensional Knapsack Problem**: Items have multiple weight dimensions and multiple capacity constraints
      - Example: Choose projects with values, but limited by both budget and manpower
    
    - **Task Scheduling with Resource Constraints**: Schedule tasks to maximize value while respecting multiple resource constraints
      - Example: Jobs require CPU, memory, and network bandwidth, all of which are limited
    
    - **Class Registration Problem**: Students selecting courses with time slots, prerequisites, and credit limits
      - Shows how multiple constraints can interact in scheduling problems

=== "Visualization"
    For a 2D knapsack with weights [2,3,4,5] and weights2 [3,1,2,4] with values [3,4,5,6] and capacities W1=5, W2=6:
    
    ```text
    3D dp array visualized as slices (simplified):
    
    dp[0][w1][w2] = 0 for all w1,w2 (no items)
    
    For item 1 (w1=2, w2=3, val=3):
    dp[1][0-1][0-2] = 0
    dp[1][2-5][3-6] = 3
    
    For item 2 (w1=3, w2=1, val=4):
    dp[2][0-2][0] = 0
    dp[2][3-5][1-6] = max(previous, 4)
    dp[2][≥3][≥4] = max(previous, 7) (if we can fit both items)
    ...
    ```
    
    Final result: Maximum value possible given both constraints

=== "Implementation Template"
    ```python
    def multi_dimensional_knapsack(values, weights1, weights2, capacity1, capacity2):
        n = len(values)
        dp = [[[0 for _ in range(capacity2 + 1)] 
              for _ in range(capacity1 + 1)] 
             for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w1 in range(capacity1 + 1):
                for w2 in range(capacity2 + 1):
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
            for w1 in range(capacity1, weights1[i]-1, -1):
                for w2 in range(capacity2, weights2[i]-1, -1):
                    dp[w1][w2] = max(dp[w1][w2], values[i] + dp[w1-weights1[i]][w2-weights2[i]])
        
        return dp[capacity1][capacity2]
    ```

### 5. Interval DP Patterns

Interval DP patterns involve solving problems by considering different intervals or ranges within a sequence. The key characteristic is that solutions are built by combining solutions to smaller intervals.

- [Matrix Chain Multiplication](matrix-chain.md) - Find optimal way to perform a sequence of operations
- [Palindrome Partitioning](palindrome-partitioning.md) - Partition a sequence to optimize some property
- [Optimal Triangulation](optimal-triangulation.md) - Find optimal way to triangulate polygons

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

=== "Example Problems"
    - **Matrix Chain Multiplication**: Find the most efficient way to multiply a sequence of matrices
      - Problem: Given dimensions of a sequence of matrices, find the minimum number of scalar multiplications
      - Example: Matrices with dimensions [10×30, 30×5, 5×60] → Optimal order: ((A×B)×C) with cost 4500
    
    - **Burst Balloons**: Burst balloons to maximize collected coins
      - Problem: Each balloon has coins, and when burst you get coins[left] * coins[i] * coins[right]
      - Clever approach: Think of it as the last balloon to burst in each interval
    
    - **Minimum Cost Tree from Leaf Values**: Build a tree where internal nodes are products of their children
      - Interval approach: Determine the optimal partitioning of the array
      - Shows how interval DP can solve problems with tree structures
    
    - **Optimal Binary Search Tree**: Build a BST with minimum expected search time
      - Problem: Given keys and their search frequencies, arrange them to minimize total search cost
      - Real application: Optimizing decision trees and search structures

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
    
    ![Matrix Chain Multiplication Visualization](https://i.imgur.com/WcZVpnf.png)

=== "Implementation Template"
    ```python
    def matrix_chain_pattern(dimensions):
        n = len(dimensions) - 1  # Number of matrices
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        for length in range(2, n + 1):
            for i in range(1, n - length + 2):
                j = i + length - 1
                dp[i][j] = float('inf')
                for k in range(i, j):
                    cost = dp[i][k] + dp[k+1][j] + dimensions[i-1] * dimensions[k] * dimensions[j]
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[1][n]
    ```
    
    **With Solution Reconstruction**:
    
    ```python
    def matrix_chain_with_parenthesization(dimensions):
        n = len(dimensions) - 1
        dp = [[float('inf')] * (n + 1) for _ in range(n + 1)]
        split = [[0] * (n + 1) for _ in range(n + 1)]
        
        # Base case: single matrix
        for i in range(1, n + 1):
            dp[i][i] = 0
        
        # Fill table
        for length in range(2, n + 1):
            for i in range(1, n - length + 2):
                j = i + length - 1
                for k in range(i, j):
                    cost = dp[i][k] + dp[k+1][j] + dimensions[i-1] * dimensions[k] * dimensions[j]
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = k
        
        # Function to print optimal parenthesization
        def print_optimal_parens(s, i, j):
            if i == j:
                return f"A{i}"
            else:
                return f"({print_optimal_parens(s, i, s[i][j])} × {print_optimal_parens(s, s[i][j] + 1, j)})"
        
        return dp[1][n], print_optimal_parens(split, 1, n)
    ```

### 5.2 Palindrome Partitioning Pattern

=== "Overview"
    **Core Idea**: Partition a sequence (often a string) to optimize some property, typically by finding cut points that create optimal subproblems.
    
    **When to Use**:
    
    - When dividing a sequence into segments with certain properties
    - When the cost of a solution depends on how you partition the data
    - When working with palindromic structures or similar patterns
    - When optimizing cuts or break points in a sequence
    
    **Recurrence Relation**: `dp[i][j] = min(dp[i][k] + dp[k+1][j] + cost(i,j))` for all k from i to j-1
    
    **Real-World Applications**:
    
    - Text justification and line breaking algorithms
    - DNA sequence segmentation in bioinformatics
    - Data compression algorithms
    - File or data partitioning for distributed systems

=== "Example Problems"
    - **Palindrome Partitioning**: Partition a string so that each substring is a palindrome with minimum cuts
      - Problem: Find the minimum cuts needed to partition a string into palindromes
      - Example: "aab" → 1 cut to get ["a", "a", "b"] or ["aa", "b"]
    
    - **Burst Balloons**: (This problem fits both matrix chain and palindrome partitioning patterns)
      - Different perspective: Choosing the optimal balloon to burst at each step
    
    - **Minimum Cost to Merge Stones**: Merge piles of stones until there's only k piles
      - Problem: Each merge of k adjacent piles costs the sum of their values
      - Shows how interval DP can handle complex merging problems
    
    - **Word Break**: Determine if a string can be segmented into dictionary words
      - Variant: Breaking a string at optimal positions to form valid words
      - Application: Text segmentation in natural language processing

=== "Visualization"
    For Palindrome Partitioning of "aab":
    
    First, compute palindrome information:
    
    ```text
    isPalindrome table:
    
        | a | a | b |
    ----|---|---|---|
     a  | T | T | F |
    ----|---|---|---|
     a  | - | T | F |
    ----|---|---|---|
     b  | - | - | T |
    ```
    
    Then, compute minimum cuts:
    
    ```text
    dp array (minimum cuts for prefix ending at index i):
    [0, 0, 1]
     a  aa aab
    ```
    
    Result: 1 cut is needed.
    
    ![Palindrome Partitioning Visualization](https://i.imgur.com/cNOZrhw.png)

=== "Implementation Template"
    ```python
    def palindrome_partition_pattern(s):
        n = len(s)
        # First, precompute palindrome information
        is_palindrome = [[False] * n for _ in range(n)]
        
        # All single characters are palindromes
        for i in range(n):
            is_palindrome[i][i] = True
        
        # Check palindromes of length 2
        for i in range(n-1):
            if s[i] == s[i+1]:
                is_palindrome[i][i+1] = True
        
        # Check palindromes of length 3+
        for length in range(3, n+1):
            for i in range(n-length+1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i+1][j-1]:
                    is_palindrome[i][j] = True
        
        # Calculate minimum cuts
        dp = [float('inf')] * n
        
        for i in range(n):
            if is_palindrome[0][i]:
                dp[i] = 0  # No cuts needed if whole string is palindrome
            else:
                for j in range(i):
                    if is_palindrome[j+1][i]:
                        dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n-1]
    ```
    
    **Alternative Approach** (using linear dp array):
    
    ```python
    def palindrome_partition_optimized(s):
        n = len(s)
        # Calculate all palindromes
        is_palindrome = [[False] * n for _ in range(n)]
        
        for i in range(n):
            is_palindrome[i][i] = True
            
        for length in range(2, n+1):
            for i in range(n-length+1):
                j = i + length - 1
                if length == 2:
                    is_palindrome[i][j] = (s[i] == s[j])
                else:
                    is_palindrome[i][j] = (s[i] == s[j] and is_palindrome[i+1][j-1])
        
        # dp[i] = minimum cuts needed for s[0...i]
        dp = list(range(-1, n))  # Initialize with worst case: dp[i] = i cuts
        
        for i in range(n):
            for j in range(i+1):
                if is_palindrome[j][i]:
                    dp[i+1] = min(dp[i+1], dp[j] + 1)
                    
        return dp[n]
    ```
    n = len(s)
    # isPalindrome[i][j] = True if s[i:j+1] is palindrome
    isPalindrome = [[False] * n for _ in range(n)]
    
    # Fill isPalindrome table
    for i in range(n):
        isPalindrome[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                isPalindrome[i][j] = (s[i] == s[j])
            else:
                isPalindrome[i][j] = (s[i] == s[j] and isPalindrome[i+1][j-1])
    
    # Minimum cuts needed for a palindrome partitioning
    dp = [float('inf')] * n
    
    for i in range(n):
        if isPalindrome[0][i]:
            dp[i] = 0
        else:
            for j in range(i):
                if isPalindrome[j+1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)
    
    return dp[n-1]
```

### 5.3 Optimal Triangulation Pattern

**Core Idea**: Find the optimal way to triangulate a polygon or divide a sequence by adding internal connections.

**When to Use**:

- When dividing a polygon into triangles
- When optimizing hierarchical structures
- When the cost function depends on non-adjacent elements

**Example Problems**:

- **Minimum Score Triangulation of Polygon**: Triangulate a convex polygon to minimize the score
- **Optimal Binary Search Tree**: Construct a BST with minimum expected search cost

### 6. Decision Making Patterns

Decision making patterns involve problems where you need to make a sequence of choices to optimize some objective. These patterns are particularly useful in financial modeling, gaming strategies, and resource management.

- [Buy/Sell Stock](buy-sell-stock.md) - Decide when to buy and sell to maximize profit
- [Game Theory](game-theory.md) - Optimal decision making in competitive scenarios

**Core Idea**: Make sequential decisions (buy/sell/hold) to maximize profit or minimize loss.

**When to Use**:

- When making decisions in time series data
- When each state depends on previous decisions
- When working with stock trading or similar financial problems
- When there are constraints on the number of transactions or actions

**Recurrence Relations**: Vary based on constraints, generally:

- `dp[i][0]` = max profit when not holding stock after day i
- `dp[i][1]` = max profit when holding stock after day i
- Transitions: `dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])` (not holding stock)
- Transitions: `dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])` (holding stock)

**Real-World Applications**:

- Algorithmic trading strategies
- Market timing optimization
- Resource allocation over time
- Inventory management systems

**Example Problems**:

- **Best Time to Buy and Sell Stock I**: You can make at most one transaction
  - Problem: Find the maximum profit by buying and selling once
  - Solution: Track minimum price seen so far and maximize profit
  - Example: Prices [7,1,5,3,6,4] → Max profit: 5 (buy at 1, sell at 6)

- **Best Time to Buy and Sell Stock II**: You can make unlimited transactions
  - Problem: Find maximum profit with any number of transactions
  - Simple approach: Add up all price increases
  - Example: Prices [7,1,5,3,6,4] → Max profit: 7 (buy at 1, sell at 5, buy at 3, sell at 6)

- **Best Time to Buy and Sell Stock III**: At most two transactions allowed
  - Problem: Find maximum profit with at most two transactions
  - Approach: Track best single transaction before and after each position
  - Shows how to handle transaction limits

- **Best Time to Buy and Sell Stock with Cooldown**: After selling, you must wait one day
  - Illustrates how to handle additional state constraints
  - Requires tracking an extra state for the cooldown period

**Visualization**:

For Stock Prices [7,1,5,3,6,4] with unlimited transactions:

```text
   | 7 | 1 | 5 | 3 | 6 | 4 |
---|---|---|---|---|---|---|
Not holding | 0 | 0 | 4 | 4 | 7 | 7 |
Holding     | -7| -1| -1| 1 | 1 | 3 |
```

The maximum profit is 7 (final value in "Not holding" row).

**Implementation Template (unlimited transactions)**:

```python
def max_profit_pattern(prices):
    n = len(prices)
    # dp[i][0] = max profit when not holding stock after day i
    # dp[i][1] = max profit when holding stock after day i
    dp = [[0, 0] for _ in range(n)]
    
    dp[0][1] = -prices[0]  # Initial buy
    
    for i in range(1, n):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])  # Sell or hold
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])  # Hold or buy
    
    return dp[n-1][0]  # Final state without stock
```

**Space-Optimized Version**:

```python
def max_profit_optimized(prices):
    not_holding, holding = 0, -prices[0]
    
    for i in range(1, len(prices)):
        prev_not_holding = not_holding
        not_holding = max(not_holding, holding + prices[i])
        holding = max(holding, prev_not_holding - prices[i])
    
    return not_holding
```

### 6.2 Game Theory Pattern

**Core Idea**: Make optimal decisions in competitive scenarios where players alternate turns.

**When to Use**:

- When modeling games with perfect information
- When players take turns making optimal moves
- When the outcome depends on the sequence of choices
- When optimizing against an opponent who also plays optimally

**Recurrence Relation**: `dp[i][j] = max(values[i] - dp[i+1][j], values[j] - dp[i][j-1])`

**Real-World Applications**:

- Board game AI development
- Bidding strategies in auctions
- Resource competition models
- Negotiation strategy optimization

**Example Problems**:

- **Stone Game**: Two players take turns picking stones from either end of the pile
  - Problem: Determine if the first player can win
  - Insight: Relative advantage is what matters, not absolute score
  - Example: Piles [5,3,4,5] → First player wins by taking 5, then 5

- **Predict the Winner**: Similar to Stone Game but with arbitrary values
  - Problem: Can the first player win a game where players take turns selecting from ends?
  - Approach: Calculate the maximum advantage first player can achieve

- **Can I Win**: Players take turns selecting from available integers
  - Problem: Determine if first player can force a win in a number selection game
  - Shows how game theory can be applied to more abstract scenarios

**Visualization**:

For Stone Game with piles [3, 9, 1, 2]:

```text
Advantage dp table (diagonal represents single elements):

    | 3 | 9 | 1 | 2 |
----|---|---|---|---|
 3  | 3 | -6| 8 | 5 |
----|---|---|---|---|
 9  | - | 9 | 8 | 7 |
----|---|---|---|---|
 1  | - | - | 1 | -1|
----|---|---|---|---|
 2  | - | - | - | 2 |
```

The value 5 in dp[0][3] means first player has a 5-point advantage, so they win.

**Implementation Template**:

```python
def game_theory_pattern(values):
    n = len(values)
    # dp[i][j] = maximum advantage first player has over second player
    dp = [[0] * n for _ in range(n)]
    
    for i in range(n):
        dp[i][i] = values[i]  # Base case
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(
                values[i] - dp[i+1][j],  # Choose left
                values[j] - dp[i][j-1]   # Choose right
            )
    
    return dp[0][n-1] >= 0  # First player wins if advantage >= 0
```

**Minimax Variation**:

```python
def minimax_game(values):
    memo = {}
    
    def minimax(i, j, is_first_player):
        if (i, j, is_first_player) in memo:
            return memo[(i, j, is_first_player)]
        
        if i > j:
            return 0
        
        if is_first_player:
            result = max(
                values[i] + minimax(i+1, j, False),
                values[j] + minimax(i, j-1, False)
            )
        else:
            result = min(
                minimax(i+1, j, True),
                minimax(i, j-1, True)
            )
            
        memo[(i, j, is_first_player)] = result
        return result
    
    first_player_score = minimax(0, len(values)-1, True)
    total = sum(values)
    
    return first_player_score >= total - first_player_score
```

### 7. State Machine Patterns

State machine patterns model problems as a set of distinct states with transitions between them. These patterns are particularly useful for problems where the system moves through different configurations over time or through a sequence of decisions.

- [State Transition](state-transition.md) - Model problems with discrete states and transitions
- [Finite State Machine (FSM)](fsm.md) - Solve problems using formal state machines

**Core Idea**: Model problems as states and transitions with specific rules governing moves between states.

**When to Use**:

- When a problem has a fixed number of distinct states
- When transitions follow strict rules
- When the system evolves through specific phases
- When solving problems that resemble automata or state diagrams

**Example Problems**:

- **Paint House**: Paint houses with different colors such that no adjacent houses have the same color
  - Problem: Minimize cost of painting n houses with 3 colors, no adjacent houses same color
  - States: Current color for each house
  - Example: Costs [[17,2,17], [16,16,5], [14,3,19]] → Minimum cost: 10 (colors: 2,3,1)

- **Best Time to Buy and Sell Stock with Cooldown**: Stock trading with cooldown periods
  - States: Holding stock, not holding stock, in cooldown
  - Shows how state machines can model complex constraints

- **Minimum Cost For Tickets**: Buy different duration passes for daily travel needs
  - Problem: Given travel days and costs for 1-day, 7-day, and 30-day passes, find minimum cost
  - States: Different pass coverage status for each day
  - Example: Travel days [1,4,6,7,8,20], costs [2,7,15] → Minimum cost: 11

**Visualization**:

For the Paint House problem with costs:
```
House 1: [17, 2, 17] (Red, Blue, Green)
House 2: [16, 16, 5] (Red, Blue, Green)
House 3: [14, 3, 19] (Red, Blue, Green)
```

DP table (rows = houses, columns = colors):

```text
   | Red | Blue| Green|
---|-----|-----|------|
H1 | 17  | 2   | 17   |
---|-----|-----|------|
H2 | 18  | 33  | 7    |
---|-----|-----|------|
H3 | 17  | 10  | 22   |
```

The minimum cost is 10 (painting houses 1=Blue, 2=Green, 3=Blue).

**Implementation Template (Paint House)**:

```python
def paint_house_pattern(costs):
    n = len(costs)
    dp = [[0, 0, 0] for _ in range(n)]
    dp[0] = costs[0][:]
    
    for i in range(1, n):
        dp[i][0] = costs[i][0] + min(dp[i-1][1], dp[i-1][2])  # Red
        dp[i][1] = costs[i][1] + min(dp[i-1][0], dp[i-1][2])  # Blue
        dp[i][2] = costs[i][2] + min(dp[i-1][0], dp[i-1][1])  # Green
    
    return min(dp[n-1])
```

**Space-Optimized Version**:

```python
def paint_house_optimized(costs):
    if not costs:
        return 0
        
    red, blue, green = costs[0]
    
    for i in range(1, len(costs)):
        new_red = costs[i][0] + min(blue, green)
        new_blue = costs[i][1] + min(red, green)
        new_green = costs[i][2] + min(red, blue)
        
        red, blue, green = new_red, new_blue, new_green
    
    return min(red, blue, green)
```

### 7.2 Finite State Machine (FSM) Pattern

**Core Idea**: Model problems using a finite number of states and transitions with specific rules governing moves between states.

**When to Use**:

- When a problem has a fixed number of distinct states
- When transitions follow strict rules
- When the system evolves through specific phases
- When solving problems that resemble automata or state diagrams

**Example Problems**:

- **Student Attendance Record**: Count the number of valid attendance records with constraints
  - Problem: Count records of length n with < 2 absences and < 3 consecutive lates
  - States: Track counts of absences and consecutive lates

- **Regular Expression Matching**: Implement regex matching with '.' and '*'
  - Problem: Match patterns with special characters to strings
  - States: Represent different matching positions and pattern states

- **Stock Trading with Transaction Fee and Cooldown**:
  - Combines state transitions with costs and constraints
  - Shows how complex rules can be modeled with multiple states

### 8. Bitmasking DP Patterns

Bitmasking DP is a powerful technique that uses binary representation to efficiently track subsets of elements. This is particularly useful for problems involving small sets where you need to consider all possible subsets.

- [Subset State](subset-state.md) - Represent and solve problems involving all possible subsets
- [State Compression](state-compression.md) - Use bit manipulation for efficient state representation
- [Submask Enumeration](submask-enumeration.md) - Process all subsets of a given set efficiently

**Core Idea**: Use bitmasks to represent subsets of elements and track state, with each bit indicating whether an element is included or excluded.

**When to Use**:

- When dealing with small sets (typically n ≤ 20) where you need to consider all subsets
- When tracking which elements have been processed or selected
- When the problem involves permutations or combinations of elements
- When the state space is exponential but manageable with bit manipulation

**Recurrence Relation**: `dp[mask] = optimal(dp[submask])` for all valid submasks or transitions

**Real-World Applications**:

- Task assignment optimization
- Resource allocation with complex dependencies
- Network routing with multiple constraints
- Computational biology sequence alignment

**Example Problems**:

- **Traveling Salesman Problem (TSP)**: Find the shortest possible route that visits all cities exactly once
  - Problem: Given distances between n cities, find minimum distance tour
  - State: Use bits to represent which cities have been visited
  - Example: For 4 cities, 1101 means cities 0, 2, and 3 have been visited

- **Minimum Cost to Visit All Nodes**: Find minimum cost path that visits all nodes in a graph
  - Variation: May have different starting/ending constraints
  - Shows how TSP can be adapted to different graph problems

- **Maximum Score After Applying Operations**: Optimize score by choosing elements with constraints
  - Problem: Select elements to maximize score while respecting dependencies
  - Uses bitmasks to track which elements have been processed

**Visualization**:

For TSP with 4 cities and the following distance matrix:
```
   | 0 | 1 | 2 | 3 |
---|---|---|---|---|
 0 | 0 | 10| 15| 20|
---|---|---|---|---|
 1 | 10| 0 | 35| 25|
---|---|---|---|---|
 2 | 15| 35| 0 | 30|
---|---|---|---|---|
 3 | 20| 25| 30| 0 |
```

DP table (rows = masks, columns = ending city):
```
Mask | City 0 | City 1 | City 2 | City 3 |
-----|--------|--------|--------|--------|
0001 |   0    |   ∞    |   ∞    |   ∞    |
0011 |   ∞    |   10   |   ∞    |   ∞    |
0101 |   ∞    |   ∞    |   15   |   ∞    |
0111 |   ∞    |   45   |   25   |   ∞    |
1001 |   ∞    |   ∞    |   ∞    |   20   |
1011 |   ∞    |   30   |   ∞    |   30   |
1101 |   ∞    |   ∞    |   35   |   30   |
1111 |   60   |   65   |   55   |   65   |
```

The minimum tour cost is 55 + 15 = 70 (starting from city 0, ending at city 2, plus return to city 0).

**Implementation Template (Traveling Salesman)**:

```python
def tsp_pattern(distances):
    n = len(distances)
    # dp[mask][i] = min distance to visit all cities in mask and end at city i
    dp = [[float('inf')] * n for _ in range(1 << n)]
    
    # Base case: start at city 0
    dp[1][0] = 0  # 1 = binary 0001 = visited only city 0
    
    # For each subset of cities
    for mask in range(1, 1 << n):
        for end in range(n):
            # If end city is in the current subset
            if mask & (1 << end):
                # Previous subset without end city
                prev_mask = mask ^ (1 << end)
                
                if prev_mask == 0:
                    # Only possible if end is the starting city
                    continue
                
                # Try all possible previous cities
                for prev in range(n):
                    if prev_mask & (1 << prev):
                        dp[mask][end] = min(
                            dp[mask][end],
                            dp[prev_mask][prev] + distances[prev][end]
                        )
    
    # Return min distance to visit all cities and return to city 0
    all_cities = (1 << n) - 1  # All 1's
    return min(dp[all_cities][i] + distances[i][0] for i in range(1, n))
```

### 8.2 State Compression Pattern

**Core Idea**: Use bitmasks to compress the representation of complex states into integers, allowing efficient storage and manipulation.

**When to Use**:

- When states have binary characteristics (yes/no, on/off, included/excluded)
- When the state space would be too large with conventional representation
- When states can be represented as combinations of independent properties
- When problems involve matching or pairing elements

**Example Problems**:

- **Assign Tasks to Workers**: Assign n tasks to n workers to minimize total cost
  - Problem: Find optimal assignment where each worker gets exactly one task
  - Approach: Use bitmask to represent which tasks have been assigned

- **Maximum Students Taking Exam**: Place students in a classroom with broken seats
  - Problem: Maximize students while respecting seating constraints
  - Uses bitmasks to represent valid seating arrangements in each row

- **Matching of Points and Houses**: Match each point to a house to minimize total distance
  - Variation of assignment problem with geometric constraints
  - Shows how matching problems can be solved with bitmasks

**Implementation Example (Assignment Problem)**:

```python
def min_cost_assignment(costs):
    n = len(costs)
    # dp[mask] = min cost to assign tasks represented by mask
    dp = [float('inf')] * (1 << n)
    dp[0] = 0  # Base case: no tasks assigned
    
    for mask in range(1 << n):
        # Count number of tasks already assigned
        worker_idx = bin(mask).count('1')
        
        # Try to assign next task to current worker
        for task in range(n):
            if not (mask & (1 << task)):  # If task not assigned yet
                new_mask = mask | (1 << task)
                dp[new_mask] = min(dp[new_mask], dp[mask] + costs[worker_idx][task])
    
    return dp[(1 << n) - 1]  # All tasks assigned
```

### 8.3 Submask Enumeration Pattern

**Core Idea**: Process all valid submasks of a given bitmask efficiently to solve problems involving subsets.

**When to Use**:

- When needing to consider all possible subsets of a set
- When combining results from smaller subsets
- When working with inclusion-exclusion problems

**Example Problems**:

- **Sum of All Subset XOR Totals**: Calculate sum of XORs of all possible subsets
- **Partition to K Equal Sum Subsets**: Divide array into k subsets with equal sums

**Implementation Example (Submask Enumeration)**:

```python
def process_all_submasks(n):
    results = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Process mask itself first
        results[mask] = process_mask(mask)
        
        # Process all submasks - Gosper's hack for iterating through submasks
        submask = mask
        while submask > 0:
            submask = (submask - 1) & mask
            # Combine results from submask
            results[mask] = combine(results[mask], results[submask])
            if submask == 0:
                break
    
    return results[(1 << n) - 1]  # Result for the full set
```

### 9. Probability DP Patterns

Probability DP patterns deal with problems involving chance, expected values, and stochastic processes. These patterns are particularly useful in modeling uncertain outcomes and analyzing probabilistic systems.

- [Expected Value](expected-value.md) - Calculate expected outcomes in probabilistic scenarios
- [Markov Chain](markov-chain.md) - Use transition probabilities between states

**Core Idea**: Calculate expected values or probabilities based on probabilistic transitions between states.

**When to Use**:

- When calculating expected outcomes in probabilistic scenarios
- When dealing with random processes with known transition probabilities
- When analyzing games with chance elements
- When modeling real-world scenarios with uncertainty

**Recurrence Relation**: `dp[state] = ∑(p[i] * (dp[next_state] + cost))` for all possible transitions i

**Real-World Applications**:

- Financial risk assessment models
- Game AI for probabilistic games like poker
- Resource planning under uncertainty
- Medical diagnosis and treatment planning
- Monte Carlo simulations

**Example Problems**:

- **Soup Servings**: Probability of emptying soup bowls in certain orders
  - Problem: Given random serving amounts, find probability of specific outcomes
  - Approach: Calculate expected values for different remaining portions

- **New 21 Game**: Probability of winning a card drawing game with constraints
  - Problem: Drawing cards until sum ≥ K or > N, find probability of winning
  - Demonstrates how to calculate probabilities with stopping conditions

- **Knight Probability in Chessboard**: Probability knight remains on board after K moves
  - Problem: Calculate probability knight stays on board after k random moves
  - Example: 3×3 board, starting position (0,0), 2 moves → probability ≈ 0.0625

**Visualization**:

For Knight Probability on a 3×3 board after 1 move from center:

```text
Initial position (center of 3×3 board):
┌───┬───┬───┐
│   │   │   │
├───┼───┼───┤
│   │ N │   │
├───┼───┼───┤
│   │   │   │
└───┴───┴───┘

After considering all 8 possible moves:
- 4 moves stay on the board (probability 4/8)
- 4 moves fall off the board (probability 4/8)

Therefore, probability = 0.5
```

**Implementation Template (Knight Probability)**:

```python
def knight_probability_pattern(N, K, r, c):
    # dp[k][i][j] = probability of knight staying on board
    # after k moves when starting at position (i, j)
    dp = [[[0] * N for _ in range(N)] for _ in range(K + 1)]
    
    # Possible knight moves
    moves = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
    
    # Base case
    for i in range(N):
        for j in range(N):
            dp[0][i][j] = 1
    
    # Fill dp table
    for k in range(1, K + 1):
        for i in range(N):
            for j in range(N):
                for dx, dy in moves:
                    prev_i, prev_j = i - dx, j - dy
                    if 0 <= prev_i < N and 0 <= prev_j < N:
                        dp[k][i][j] += dp[k-1][prev_i][prev_j] / 8.0
    
    return dp[K][r][c]
```

### 9.2 Markov Chain Pattern

**Core Idea**: Model problems as Markov processes where future states depend only on the current state.

**When to Use**:

- When dealing with systems where transitions have fixed probabilities
- When calculating steady-state probabilities
- When modeling sequential decision processes with probabilistic outcomes

**Example Problems**:

- **Random Walk**: Calculate probability of ending at specific positions
- **Dice Roll Simulation**: Probability of specific sequences with constraints
- **Google PageRank**: Calculate importance scores for linked pages

**Implementation Example (Random Walk)**:

```python
def random_walk_probability(n, steps, start):
    # dp[step][position] = probability of being at position after step steps
    dp = [[0] * (2*n+1) for _ in range(steps+1)]
    
    # Base case: 100% probability of starting at the start position
    dp[0][start+n] = 1
    
    # For each step
    for step in range(1, steps+1):
        for pos in range(2*n+1):
            # Skip positions that can't be reached
            if pos == 0 or pos == 2*n:
                continue
                
            # Equal probability of moving left or right
            dp[step][pos] = dp[step-1][pos-1] * 0.5 + dp[step-1][pos+1] * 0.5
    
    # Return probability of being at position 0 after all steps
    return dp[steps][n]
```

### 9.3 Conditional Probability Pattern

**Core Idea**: Calculate probabilities based on conditions and constraints.

**When to Use**:

- When outcomes depend on specific conditions being met
- When working with conditional events
- When calculating probabilities with dependencies

**Example Problems**:

- **Out of Boundary Paths**: Probability of a ball moving out of grid bounds
- **Predict Winner with Random Moves**: Win probability with random game elements

### 10. String DP Patterns

String DP patterns focus on problems involving strings, substrings, and character sequences. These patterns are essential for text processing, bioinformatics, and many other applications involving sequence data.

- [Palindrome Problems](palindrome.md) - Find or construct palindromes in strings
- [String Matching](string-matching.md) - Find patterns or subsequences in strings
- [String Transformation](string-transformation.md) - Convert one string to another efficiently

## Learning Strategy

To master dynamic programming:

1. **Start with the fundamentals**: Understand the core concepts of DP before diving into specific patterns.
2. **Learn pattern by pattern**: Focus on one pattern at a time and solve multiple problems within that pattern.
3. **Practice implementation**: Code solutions for each pattern to reinforce understanding.
4. **Connect the patterns**: Recognize how patterns can be combined to solve more complex problems.
5. **Review and optimize**: After solving a problem, look for ways to improve your solution.

## Further Resources

- [Dynamic Programming Fundamentals](fundamentals.md) - Core concepts and approaches
- [DP Problem Collections](index.md) - Categorized problems by difficulty
- [Visual Explanations](https://visualgo.net/en) - Interactive visualizations of algorithms

### 10.1 Substring Pattern

**Core Idea**: Process substrings and build up solutions by considering different substring properties and relationships.

**When to Use**:

- When analyzing or manipulating substrings of a string
- When finding patterns or special subsequences in strings
- When calculating properties of all possible substrings
- When matching or comparing string features

**Recurrence Relation**: Varies, generally: `dp[i][j] = f(dp[i+1][j-1], s[i], s[j])`

**Real-World Applications**:

- Text processing and analysis
- DNA and protein sequence analysis
- Natural language processing
- Pattern matching in data streams
- Spelling correction algorithms

**Example Problems**:

- **Longest Palindromic Substring**: Find the longest substring that reads the same forward and backward
  - Problem: For string "babad", the longest palindromic substring is "bab" or "aba"
  - Approach: Check if s[i:j] is a palindrome by checking if s[i]=s[j] and s[i+1:j-1] is a palindrome
  - Example: For "racecar", the result is "racecar" (the entire string)

- **Count Different Palindromic Subsequences**: Count unique palindromic subsequences
  - Problem: More complex variation requiring distinct subsequence counting
  - Shows advanced application of palindrome detection combined with counting

- **Distinct Subsequences**: Count number of ways a subsequence can form a target string
  - Problem: Find number of subsequences of s that equal t
  - Example: s="rabbbit", t="rabbit" → 3 different ways to form "rabbit"

**Visualization**:

For the Longest Palindromic Substring in "babad":

```text
   | b | a | b | a | d |
---|---|---|---|---|---|
 b | T | F | T | F | F |
---|---|---|---|---|---|
 a | - | T | F | T | F |
---|---|---|---|---|---|
 b | - | - | T | F | F |
---|---|---|---|---|---|
 a | - | - | - | T | F |
---|---|---|---|---|---|
 d | - | - | - | - | T |
```

Where T means the substring is a palindrome. The longest palindromic substring is "bab" (from indices 0-2).

**Implementation Template (Longest Palindromic Substring)**:

```python
def longest_palindrome_pattern(s):
    n = len(s)
    # dp[i][j] = whether s[i:j+1] is a palindrome
    dp = [[False] * n for _ in range(n)]
    
    # All single characters are palindromes
    for i in range(n):
        dp[i][i] = True
    
    start, max_len = 0, 1
    
    # Check for palindromes of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2
    
    # Check for palindromes of length 3 or more
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length
    
    return s[start:start + max_len]
```

### 10.2 String Transformation Pattern

**Core Idea**: Transform one string into another through a series of operations while optimizing some metric.

**When to Use**:

- When converting between strings with specific operations
- When analyzing string similarity or distance
- When working with string manipulation problems

**Example Problems**:

- **Regular Expression Matching**: Check if a string matches a pattern
- **Wildcard Matching**: Match strings with wildcard characters
- **Word Break**: Determine if a string can be segmented into dictionary words

**Implementation Example (Word Break)**:

```python
def word_break(s, word_dict):
    word_set = set(word_dict)
    n = len(s)
    # dp[i] = whether s[:i] can be segmented into dictionary words
    dp = [False] * (n + 1)
    dp[0] = True  # Empty string can always be segmented
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]
```

## Summary and Practical Guidance

### Pattern Selection Guide

When facing a new problem, first identify its key characteristics:

| Problem Type | Key Features | Suggested Pattern | Example Problems |
|--------------|-------------|-------------------|------------------|
| Single sequence operations | Calculating value based on previous elements | Fibonacci, LIS | Climbing Stairs, Longest Increasing Subsequence |
| Two sequence comparison | Finding commonality or differences | LCS, Edit Distance | Longest Common Subsequence, Edit Distance |
| Grid traversal | Moving through a 2D grid | Unique Paths, Min/Max Path Sum | Robot Path Planning, Minimum Path Sum |
| Resource allocation with constraints | Selecting items with restrictions | Knapsack patterns | 0/1 Knapsack, Coin Change |
| Operations on intervals | Finding optimal splits or merges | Matrix Chain, Palindrome Partition | Matrix Multiplication, Burst Balloons |
| Sequential decision making | Optimizing over multiple steps | Buy/Sell Stock, Game Theory | Stock Trading, Stone Game |
| State transitions | Problems with distinct configurations | State Machine patterns | Paint House, Student Attendance |
| Set subset operations | Working with all possible subsets | Bitmasking patterns | Traveling Salesman, Task Assignment |
| Probabilistic outcomes | Calculating expected values | Expected Value pattern | Knight Probability, Random Walk |
| String processing | Working with substrings and patterns | String DP patterns | Longest Palindrome, Word Break |

### Approach to Solving DP Problems

1. **Identify the problem type** and match it to the appropriate pattern
2. **Define states clearly** - What information do you need to represent each subproblem?
3. **Formulate the recurrence relation** - How do smaller subproblems relate to larger ones?
4. **Determine base cases** - What are the simplest scenarios you can solve directly?
5. **Choose implementation approach**:
   - Top-down (memoization): Better for sparse state spaces, easier to reason about
   - Bottom-up (tabulation): Often more efficient, easier to optimize space

### Optimization Techniques

1. **Space Optimization**: Reduce memory usage by only keeping necessary states
   - Rolling arrays: Use prev, curr arrays instead of full table
   - State compression: Use minimal representation for states

2. **Pre-computation**: Calculate and store common values
   - Preprocess to avoid redundant calculations
   - Create lookup tables for frequently accessed values

3. **Efficient Transitions**: Minimize work in the inner loop
   - Use appropriate data structures for fast lookups
   - Avoid recomputing values that don't change

### Common Pitfalls and Solutions

| Pitfall | Solution |
|---------|----------|
| Overlapping cases | Ensure base cases are correct and handle edge cases |
| Infinite recursion | Verify that recurrence relation makes progress |
| Memory limit exceeded | Apply space optimization techniques |
| Time limit exceeded | Look for more efficient algorithms or optimizations |
| Off-by-one errors | Double-check index calculations and boundaries |

### Practice Strategy

1. **Master one pattern at a time** - Deeply understand each pattern before moving on
2. **Solve multiple problems** for each pattern to recognize variations
3. **Gradually increase difficulty** within each pattern
4. **Review and optimize** your solutions after solving them
5. **Mix problems from different patterns** to improve recognition skills

By understanding these dynamic programming patterns and when to apply them, you'll develop the ability to recognize and solve a wide range of algorithmic problems efficiently.
