# Dynamic Programming - Hard Problems

## 🎯 Learning Objectives

Master advanced DP techniques including interval DP, state machine DP, and complex optimization problems:

- Complex state transitions and multi-dimensional DP
- Interval and range DP patterns
- State machine and game theory DP
- Advanced optimization with multiple constraints
- Matrix chain multiplication and complex recurrences

---

## Problem 1: Regular Expression Matching

**Difficulty**: 🔴 Hard  
**Pattern**: 2D String DP with Wildcards  
**Time**: O(m×n), **Space**: O(m×n)

=== "Problem"

    Given an input string `s` and a pattern `p`, implement regular expression matching with support for '.' and '*' where:

    - '.' Matches any single character.
    - '*' Matches zero or more of the preceding element.

    The matching should cover the **entire** input string (not partial).

    **Example 1:**
    ```
    Input: s = "aa", p = "a"
    Output: false
    Explanation: "a" does not match the entire string "aa".
    ```

    **Example 2:**
    ```
    Input: s = "aa", p = "a*"
    Output: true
    Explanation: '*' means zero or more of the preceding element, 'a'.
    ```

    **Example 3:**
    ```
    Input: s = "ab", p = ".*"
    Output: true
    Explanation: ".*" means "zero or more (*) of any character (.)".
    ```

=== "Solution"

    ```python
    def isMatch(s, p):
        """
        2D DP solution for regex matching.
        
        Time: O(m×n), Space: O(m×n)
        """
        m, n = len(s), len(p)
        
        # dp[i][j] = True if s[:i] matches p[:j]
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        # Base case: empty string matches empty pattern
        dp[0][0] = True
        
        # Handle patterns like a*, a*b*, a*b*c* (can match empty string)
        for j in range(2, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-2]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j-1] == '*':
                    # Star case: check two possibilities
                    # 1. Match zero occurrences (ignore X*)
                    dp[i][j] = dp[i][j-2]
                    
                    # 2. Match one or more occurrences
                    if matches(s[i-1], p[j-2]):
                        dp[i][j] = dp[i][j] or dp[i-1][j]
                else:
                    # Regular character or dot
                    if matches(s[i-1], p[j-1]):
                        dp[i][j] = dp[i-1][j-1]
        
        return dp[m][n]
    
    def matches(s_char, p_char):
        """Check if characters match (considering '.' wildcard)"""
        return p_char == '.' or s_char == p_char
    
    def isMatch_recursive(s, p):
        """
        Recursive solution with memoization.
        
        Time: O(m×n), Space: O(m×n)
        """
        memo = {}
        
        def dp(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            # Base case: reached end of pattern
            if j == len(p):
                return i == len(s)
            
            # Check if current characters match
            first_match = i < len(s) and (p[j] == s[i] or p[j] == '.')
            
            # Handle star case
            if j + 1 < len(p) and p[j + 1] == '*':
                result = (dp(i, j + 2) or  # Match zero occurrences
                         (first_match and dp(i + 1, j)))  # Match one or more
            else:
                result = first_match and dp(i + 1, j + 1)
            
            memo[(i, j)] = result
            return result
        
        return dp(0, 0)
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Star Handling**: '*' can match zero or more of preceding character
    - **Two Cases for '*'**: Match zero (skip pattern) or match one/more (consume string)
    - **Base Cases**: Empty patterns and star patterns that can match empty string
    - **Character Matching**: Handle '.' wildcard and exact character matches
    
    **Algorithm Strategy:**
    For each position, either:
    1. Characters match directly (move both pointers)
    2. Star pattern: try both zero matches and one+ matches
    3. Dot matches any character

---

## Problem 2: Burst Balloons

**Difficulty**: 🔴 Hard  
**Pattern**: Interval DP  
**Time**: O(n³), **Space**: O(n²)

=== "Problem"

    You have `n` balloons, indexed from 0 to n - 1. Each balloon is painted with a number on it represented by an array `nums`. You are asked to burst all the balloons.

    If you burst balloon `i`, you will get `nums[i - 1] * nums[i] * nums[i + 1]` coins. If `i - 1` or `i + 1` goes out of bounds, then treat it as if there is a balloon with a 1 painted on it.

    Return the maximum coins you can collect by bursting the balloons wisely.

    **Example 1:**
    ```
    Input: nums = [3,1,5,8]
    Output: 167
    Explanation:
    nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
    coins = 3*1*5 + 3*5*8 + 1*3*8 + 1*8*1 = 15 + 120 + 24 + 8 = 167
    ```

=== "Solution"

    ```python
    def maxCoins(nums):
        """
        Interval DP solution - think backwards about last balloon to burst.
        
        Time: O(n³), Space: O(n²)
        """
        # Add boundary balloons with value 1
        balloons = [1] + nums + [1]
        n = len(balloons)
        
        # dp[i][j] = max coins from bursting balloons between i and j (exclusive)
        dp = [[0] * n for _ in range(n)]
        
        # Try all possible intervals
        for length in range(2, n):  # length of interval
            for left in range(n - length):
                right = left + length
                
                # Try each balloon k as the LAST one to burst in interval (left, right)
                for k in range(left + 1, right):
                    # Coins from bursting k last = left * k * right + 
                    # optimal coins from left subinterval + optimal coins from right subinterval
                    coins = balloons[left] * balloons[k] * balloons[right]
                    coins += dp[left][k] + dp[k][right]
                    
                    dp[left][right] = max(dp[left][right], coins)
        
        return dp[0][n - 1]
    
    def maxCoins_memoization(nums):
        """
        Top-down memoization approach.
        """
        balloons = [1] + nums + [1]
        memo = {}
        
        def dp(left, right):
            if left + 1 == right:
                return 0  # No balloons between left and right
            
            if (left, right) in memo:
                return memo[(left, right)]
            
            max_coins = 0
            # Try each balloon k as last to burst
            for k in range(left + 1, right):
                coins = balloons[left] * balloons[k] * balloons[right]
                coins += dp(left, k) + dp(k, right)
                max_coins = max(max_coins, coins)
            
            memo[(left, right)] = max_coins
            return max_coins
        
        return dp(0, len(balloons) - 1)
    
    def maxCoins_detailed(nums):
        """
        Version that also returns the optimal bursting order.
        """
        balloons = [1] + nums + [1]
        n = len(balloons)
        dp = [[0] * n for _ in range(n)]
        choice = [[0] * n for _ in range(n)]
        
        for length in range(2, n):
            for left in range(n - length):
                right = left + length
                
                for k in range(left + 1, right):
                    coins = balloons[left] * balloons[k] * balloons[right]
                    coins += dp[left][k] + dp[k][right]
                    
                    if coins > dp[left][right]:
                        dp[left][right] = coins
                        choice[left][right] = k
        
        # Reconstruct the order
        def get_order(left, right, order):
            if left + 1 >= right:
                return
            
            k = choice[left][right]
            get_order(left, k, order)
            get_order(k, right, order)
            order.append(nums[k - 1])  # Convert back to original indexing
        
        order = []
        get_order(0, n - 1, order)
        
        return dp[0][n - 1], order
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Interval DP**: Process intervals of increasing length
    - **Last Balloon Strategy**: Think about which balloon to burst LAST in each interval
    - **Boundary Handling**: Add balloons with value 1 at boundaries
    - **Subproblem Independence**: After choosing last balloon, subproblems are independent
    
    **Counter-intuitive Insight:**
    Instead of thinking "which balloon to burst first", think "which balloon to burst LAST" in each interval. This makes subproblems independent.

---

## Problem 3: Scramble String

**Difficulty**: 🔴 Hard  
**Pattern**: 3D String DP  
**Time**: O(n⁴), **Space**: O(n³)

=== "Problem"

    We can scramble a string s to get a string t using the following algorithm:

    1. If the length of the string is 1, stop.
    2. If the length of the string is > 1, do the following:
       - Split the string into two non-empty substrings at a random index.
       - Randomly decide to swap the two substrings or to keep them in the same order.
       - Apply this algorithm recursively on each substring.

    Given two strings s1 and s2 of the same length, return true if s2 is a scrambled string of s1, otherwise, return false.

    **Example 1:**
    ```
    Input: s1 = "great", s2 = "rgeat"
    Output: true
    Explanation: One possible scrambling of "great" is "rgeat".
    ```

=== "Solution"

    ```python
    def isScramble(s1, s2):
        """
        3D DP solution for scramble string.
        
        Time: O(n⁴), Space: O(n³)
        """
        if len(s1) != len(s2):
            return False
        
        n = len(s1)
        
        # dp[i][j][k] = True if s1[i:i+k] can be scrambled to s2[j:j+k]
        dp = [[[False] * (n + 1) for _ in range(n)] for _ in range(n)]
        
        # Base case: single characters
        for i in range(n):
            for j in range(n):
                dp[i][j][1] = s1[i] == s2[j]
        
        # Fill for all lengths
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                for j in range(n - length + 1):
                    # Try all possible split points
                    for k in range(1, length):
                        # Case 1: No swap
                        # s1[i:i+k] -> s2[j:j+k] and s1[i+k:i+length] -> s2[j+k:j+length]
                        if (dp[i][j][k] and dp[i + k][j + k][length - k]):
                            dp[i][j][length] = True
                            break
                        
                        # Case 2: Swap
                        # s1[i:i+k] -> s2[j+length-k:j+length] and s1[i+k:i+length] -> s2[j:j+length-k]
                        if (dp[i][j + length - k][k] and dp[i + k][j][length - k]):
                            dp[i][j][length] = True
                            break
        
        return dp[0][0][n]
    
    def isScramble_memoization(s1, s2):
        """
        Memoization approach with string keys.
        
        Time: O(n⁴), Space: O(n³)
        """
        memo = {}
        
        def helper(s1, s2):
            if (s1, s2) in memo:
                return memo[(s1, s2)]
            
            if s1 == s2:
                return True
            
            if len(s1) != len(s2) or sorted(s1) != sorted(s2):
                return False
            
            n = len(s1)
            
            # Try all possible split points
            for i in range(1, n):
                # Case 1: No swap
                if (helper(s1[:i], s2[:i]) and helper(s1[i:], s2[i:])):
                    memo[(s1, s2)] = True
                    return True
                
                # Case 2: Swap
                if (helper(s1[:i], s2[n-i:]) and helper(s1[i:], s2[:n-i])):
                    memo[(s1, s2)] = True
                    return True
            
            memo[(s1, s2)] = False
            return False
        
        return helper(s1, s2)
    
    def isScramble_optimized(s1, s2):
        """
        Optimized version with early termination.
        """
        if len(s1) != len(s2):
            return False
        
        if s1 == s2:
            return True
        
        # Quick check: must have same character frequencies
        if sorted(s1) != sorted(s2):
            return False
        
        memo = {}
        
        def dp(i1, i2, length):
            if (i1, i2, length) in memo:
                return memo[(i1, i2, length)]
            
            if length == 1:
                result = s1[i1] == s2[i2]
            else:
                result = False
                # Try all split points
                for k in range(1, length):
                    # No swap case
                    if dp(i1, i2, k) and dp(i1 + k, i2 + k, length - k):
                        result = True
                        break
                    
                    # Swap case
                    if dp(i1, i2 + length - k, k) and dp(i1 + k, i2, length - k):
                        result = True
                        break
            
            memo[(i1, i2, length)] = result
            return result
        
        return dp(0, 0, len(s1))
    ```

=== "Insights"

    **Key Concepts:**
    
    - **3D State Space**: Track starting positions and length
    - **Two Split Cases**: With and without swapping substrings
    - **Character Frequency**: Quick check for impossible cases
    - **Recursive Structure**: Problem breaks into independent subproblems
    
    **Optimization Strategies:**
    1. Early termination with character frequency check
    2. Memoization to avoid recomputing same subproblems
    3. String-based keys vs index-based keys trade-offs

---

## Problem 4: Distinct Subsequences

**Difficulty**: 🔴 Hard  
**Pattern**: 2D String DP with Counting  
**Time**: O(m×n), **Space**: O(n)

=== "Problem"

    Given two strings `s` and `t`, return the number of distinct subsequences of `s` which equals `t`.

    **Example 1:**
    ```
    Input: s = "rabbbit", t = "rabbit"
    Output: 3
    Explanation:
    As shown below, there are 3 ways you can generate "rabbit" from s:
    rabbbit -> rabbit
    rabbbit -> rabbit  
    rabbbit -> rabbit
    ```

=== "Solution"

    ```python
    def numDistinct(s, t):
        """
        2D DP solution for counting distinct subsequences.
        
        Time: O(m×n), Space: O(m×n)
        """
        m, n = len(s), len(t)
        
        # dp[i][j] = number of ways to form t[:j] from s[:i]
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base case: empty target can be formed in 1 way (by choosing nothing)
        for i in range(m + 1):
            dp[i][0] = 1
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Always have the option to not use s[i-1]
                dp[i][j] = dp[i-1][j]
                
                # If characters match, add the ways using s[i-1]
                if s[i-1] == t[j-1]:
                    dp[i][j] += dp[i-1][j-1]
        
        return dp[m][n]
    
    def numDistinct_optimized(s, t):
        """
        Space-optimized version using 1D array.
        
        Time: O(m×n), Space: O(n)
        """
        m, n = len(s), len(t)
        
        # dp[j] = number of ways to form t[:j] from current prefix of s
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty target
        
        for i in range(1, m + 1):
            # Process from right to left to avoid overwriting needed values
            for j in range(min(i, n), 0, -1):
                if s[i-1] == t[j-1]:
                    dp[j] += dp[j-1]
        
        return dp[n]
    
    def numDistinct_memoization(s, t):
        """
        Top-down memoization approach.
        """
        memo = {}
        
        def dp(i, j):
            if j == len(t):
                return 1  # Found complete subsequence
            if i == len(s):
                return 0  # Ran out of source characters
            
            if (i, j) in memo:
                return memo[(i, j)]
            
            # Option 1: Skip current character in s
            result = dp(i + 1, j)
            
            # Option 2: Use current character if it matches
            if s[i] == t[j]:
                result += dp(i + 1, j + 1)
            
            memo[(i, j)] = result
            return result
        
        return dp(0, 0)
    
    def numDistinct_with_paths(s, t):
        """
        Return count and all actual subsequences.
        """
        m, n = len(s), len(t)
        
        # For tracking paths
        paths = [[[] for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Base case
        for i in range(m + 1):
            paths[i][0] = [[]]  # One way to form empty string
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Copy paths that don't use s[i-1]
                paths[i][j] = [path[:] for path in paths[i-1][j]]
                
                # Add paths that use s[i-1] if characters match
                if s[i-1] == t[j-1]:
                    for path in paths[i-1][j-1]:
                        new_path = path + [i-1]  # Add index of matched character
                        paths[i][j].append(new_path)
        
        return len(paths[m][n]), paths[m][n]
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Subsequence Counting**: Count ways to form target from source
    - **Character Matching**: Use character when it matches, always have option to skip
    - **Space Optimization**: Process right-to-left in 1D array
    - **Base Cases**: Empty target has 1 way, empty source has 0 ways (unless target empty)
    
    **Recurrence Relation:**
    ```
    If s[i-1] == t[j-1]:
        dp[i][j] = dp[i-1][j] + dp[i-1][j-1]
    Else:
        dp[i][j] = dp[i-1][j]
    ```

---

## Problem 5: Interleaving String

**Difficulty**: 🔴 Hard  
**Pattern**: 2D String DP with Path Tracking  
**Time**: O(m×n), **Space**: O(m×n)

=== "Problem"

    Given strings `s1`, `s2`, and `s3`, find whether `s3` is formed by an **interleaving** of `s1` and `s2`.

    An **interleaving** of two strings `s` and `t` is a configuration where `s` and `t` are divided into `n` and `m` non-empty substrings respectively, and interleaved.

    **Example 1:**
    ```
    Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
    Output: true
    Explanation: One possible way: "aa" + "dbbc" + "bc" + "a" + "c"
    ```

=== "Solution"

    ```python
    def isInterleave(s1, s2, s3):
        """
        2D DP solution for interleaving string.
        
        Time: O(m×n), Space: O(m×n)
        """
        m, n, l = len(s1), len(s2), len(s3)
        
        # Quick check: lengths must add up
        if m + n != l:
            return False
        
        # dp[i][j] = True if s3[:i+j] can be formed by interleaving s1[:i] and s2[:j]
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        # Base case
        dp[0][0] = True
        
        # Fill first row (only using s2)
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
        
        # Fill first column (only using s1)
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
        
        # Fill rest of table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Can form s3[:i+j] if either:
                # 1. Can form s3[:i+j-1] from s1[:i] and s2[:j-1], and s2[j-1] == s3[i+j-1]
                # 2. Can form s3[:i+j-1] from s1[:i-1] and s2[:j], and s1[i-1] == s3[i+j-1]
                dp[i][j] = ((dp[i][j-1] and s2[j-1] == s3[i+j-1]) or
                           (dp[i-1][j] and s1[i-1] == s3[i+j-1]))
        
        return dp[m][n]
    
    def isInterleave_optimized(s1, s2, s3):
        """
        Space-optimized version using 1D array.
        
        Time: O(m×n), Space: O(n)
        """
        m, n, l = len(s1), len(s2), len(s3)
        
        if m + n != l:
            return False
        
        # Use smaller string for space optimization
        if m > n:
            return isInterleave_optimized(s2, s1, s3)
        
        dp = [False] * (n + 1)
        dp[0] = True
        
        # Initialize first row
        for j in range(1, n + 1):
            dp[j] = dp[j-1] and s2[j-1] == s3[j-1]
        
        # Fill row by row
        for i in range(1, m + 1):
            dp[0] = dp[0] and s1[i-1] == s3[i-1]
            
            for j in range(1, n + 1):
                dp[j] = ((dp[j-1] and s2[j-1] == s3[i+j-1]) or
                        (dp[j] and s1[i-1] == s3[i+j-1]))
        
        return dp[n]
    
    def isInterleave_with_path(s1, s2, s3):
        """
        Return whether interleaving is possible and one valid interleaving.
        """
        m, n, l = len(s1), len(s2), len(s3)
        
        if m + n != l:
            return False, ""
        
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        parent = [[None] * (n + 1) for _ in range(m + 1)]
        
        dp[0][0] = True
        
        # Fill first row
        for j in range(1, n + 1):
            if dp[0][j-1] and s2[j-1] == s3[j-1]:
                dp[0][j] = True
                parent[0][j] = (0, j-1, 's2')
        
        # Fill first column
        for i in range(1, m + 1):
            if dp[i-1][0] and s1[i-1] == s3[i-1]:
                dp[i][0] = True
                parent[i][0] = (i-1, 0, 's1')
        
        # Fill rest of table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if dp[i][j-1] and s2[j-1] == s3[i+j-1]:
                    dp[i][j] = True
                    parent[i][j] = (i, j-1, 's2')
                elif dp[i-1][j] and s1[i-1] == s3[i+j-1]:
                    dp[i][j] = True
                    parent[i][j] = (i-1, j, 's1')
        
        if not dp[m][n]:
            return False, ""
        
        # Reconstruct path
        path = []
        i, j = m, n
        while parent[i][j] is not None:
            pi, pj, source = parent[i][j]
            if source == 's1':
                path.append(f"s1[{i-1}]")
            else:
                path.append(f"s2[{j-1}]")
            i, j = pi, pj
        
        path.reverse()
        return True, " -> ".join(path)
    ```

=== "Insights"

    **Key Concepts:**
    
    - **2D State Space**: Track positions in both input strings
    - **Character Consumption**: Each step consumes exactly one character
    - **Two Choices**: At each position, choose from s1 or s2
    - **Length Constraint**: Total length must equal sum of input lengths
    
    **Algorithm Logic:**
    At position (i,j), we can reach it by:
    1. Using s1[i-1] (if it matches s3[i+j-1])
    2. Using s2[j-1] (if it matches s3[i+j-1])

---

## 📝 Summary

### Advanced DP Patterns Mastered

| **Pattern** | **Key Technique** | **Example Problems** |
|-------------|-------------------|---------------------|
| **Interval DP** | Process intervals of increasing length | Burst Balloons, Matrix Chain |
| **String Matching DP** | Complex character matching rules | Regex Matching, Scramble String |
| **Counting DP** | Count number of valid arrangements | Distinct Subsequences |
| **3D DP** | Multiple dimension state tracking | Scramble String |
| **Path Reconstruction** | Track optimal choices for solution | Interleaving String |

### Key Problem-Solving Strategies

1. **Think Backwards**: Sometimes easier to think about last action (Burst Balloons)
2. **Add Boundaries**: Simplify edge cases with dummy elements (Burst Balloons)
3. **Character Frequency**: Quick impossibility checks (Scramble String)
4. **Multiple Cases**: Handle different scenarios systematically (Regex Matching)
5. **Space Optimization**: Reduce dimensions when possible

### Time Complexity Patterns

- **String DP**: Usually O(m×n) for two strings
- **Interval DP**: Often O(n³) for trying all split points
- **3D DP**: O(n⁴) when considering all positions and lengths
- **Counting DP**: Same as decision DP but accumulates counts

### Common Optimization Techniques

1. **Memoization**: Cache expensive recursive calls
2. **Space Reduction**: Use rolling arrays or 1D when possible
3. **Early Termination**: Quick checks for impossible cases
4. **Reverse Iteration**: Avoid overwriting needed values

---

## 🎯 Mastery Checklist

✅ **Interval DP**: Understand processing intervals of increasing size  
✅ **Complex String DP**: Handle wildcards and complex matching rules  
✅ **3D State Spaces**: Track multiple dimensions efficiently  
✅ **Counting vs Decision**: Adapt algorithms for counting problems  
✅ **Path Reconstruction**: Build actual solutions, not just optimal values  

---

## 🏆 Congratulations!

You've mastered the most challenging DP problems! These techniques appear in:

- **Compiler Design**: Parsing and optimization
- **Bioinformatics**: Sequence alignment and analysis  
- **Game Theory**: Optimal strategy computation
- **Operations Research**: Resource allocation optimization
- **AI/ML**: Dynamic programming in reinforcement learning

### 📚 What's Next

- **[Advanced Algorithms](../advanced/index.md)** - Network flows, advanced graph algorithms
- **[System Design](../../../system-design/index.md)** - Apply DP at scale
- **Competitive Programming** - Practice more complex variations
- **Research Papers** - Explore cutting-edge DP applications

---

*You're now equipped to tackle any DP problem, no matter how complex!*
