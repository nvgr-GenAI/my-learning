# Palindrome Partitioning Pattern

## Introduction

The Palindrome Partitioning pattern is a specialized interval dynamic programming pattern that deals with partitioning a sequence (typically a string) into segments that satisfy certain properties, often related to palindromes.

=== "Overview"
    **Core Idea**: Partition a sequence (often a string) to optimize some property, typically by finding cut points that create optimal subproblems.
    
    **When to Use**:
    
    - When dividing a sequence into segments with certain properties
    - When the cost of a solution depends on how you partition the data
    - When working with palindromic structures or similar patterns
    - When optimizing cuts or break points in a sequence
    
    **Recurrence Relation**: `dp[i] = min(dp[j] + cost(j+1,i))` for all j < i
    
    **Real-World Applications**:
    
    - Text justification and line breaking algorithms
    - DNA sequence segmentation in bioinformatics
    - Data compression algorithms
    - File or data partitioning for distributed systems
    - Natural language processing for word segmentation

=== "Example Problems"
    - **Palindrome Partitioning (Minimum Cuts)**: Partition a string so that each substring is a palindrome with minimum cuts
      - Problem: Find the minimum cuts needed to partition a string into palindromes
      - Example: "aab" → 1 cut to get ["aa", "b"] (minimum cuts)
    
    - **Palindrome Partitioning (All Partitions)**: Find all possible ways to partition a string into palindromes
      - Problem: Generate all possible palindrome partitionings of a string
      - Example: "aab" → [["a","a","b"], ["aa","b"]]
    
    - **Text Justification**: Break text into lines with minimum "badness"
      - Problem: Partition text to minimize the sum of squares of empty spaces
      - Application: Document formatting systems and typesetting
    
    - **Word Break**: Determine if a string can be segmented into dictionary words
      - Problem: Given a string and a dictionary, can the string be split into valid words?
      - Example: "leetcode" with dictionary ["leet", "code"] → True ("leet" + "code")
    
    - **Word Break II**: Return all possible word break results
      - Variation: Instead of yes/no, return all valid partitionings
      - Example: "catsanddog" with dictionary ["cat", "cats", "sand", "and", "dog"] → ["cat sand dog", "cats and dog"]

=== "Visualization"
    For Palindrome Partitioning with minimum cuts for "aab":
    
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
    
    For "a": 0 cuts (already a palindrome)
    For "aa": 0 cuts (already a palindrome)
    For "aab": 1 cut (partition into "aa"|"b")
    ```
    
    Result: 1 cut is needed.
    
    ![Palindrome Partitioning Visualization](https://i.imgur.com/cNOZrhw.png)

=== "Implementation"
    **Palindrome Partitioning (Minimum Cuts)**:
    
    ```python
    def min_cut(s):
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
    
    **Palindrome Partitioning (All Partitions)**:
    
    ```python
    def partition(s):
        n = len(s)
        # Precompute palindromes
        is_palindrome = [[False] * n for _ in range(n)]
        
        for i in range(n):
            is_palindrome[i][i] = True
        
        for i in range(n-1):
            if s[i] == s[i+1]:
                is_palindrome[i][i+1] = True
        
        for length in range(3, n+1):
            for i in range(n-length+1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i+1][j-1]:
                    is_palindrome[i][j] = True
        
        # Backtracking to find all partitions
        result = []
        
        def backtrack(start, path):
            if start >= n:
                result.append(path.copy())
                return
            
            for end in range(start, n):
                if is_palindrome[start][end]:
                    path.append(s[start:end+1])
                    backtrack(end + 1, path)
                    path.pop()
        
        backtrack(0, [])
        return result
    ```
    
    **Word Break**:
    
    ```python
    def word_break(s, wordDict):
        n = len(s)
        word_set = set(wordDict)
        dp = [False] * (n + 1)
        dp[0] = True  # Empty string can always be segmented
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[n]
    ```

=== "Tips and Insights"
    - **Preprocessing**: For palindrome problems, precompute all palindrome information first
    - **DP State Definition**:
      - For minimum cuts: `dp[i]` = minimum cuts needed for substring s[0...i]
      - For word break: `dp[i]` = whether s[0...i] can be segmented
    - **Optimization Techniques**:
      - Preprocessing palindromes reduces repeated work
      - Can use memoization for recursive approaches
      - Consider starting from the end for some problems
    - **Time Complexity Comparison**:
      - Naive recursive approach: O(2^n)
      - With DP: O(n²) for minimum cuts, O(n²) for word break
      - With backtracking: Exponential for all partitions
    - **Memory Optimization**:
      - For minimum cuts, we only need a 1D array of size n
      - For palindrome checks, we need a 2D array of size n×n
    - **Common Variations**:
      - Finding vs. counting vs. enumerating all partitions
      - Different cost functions for cuts
      - Different constraints on segments (palindromes, dictionary words, etc.)
    - **Related Problems**:
      - String segmentation
      - Text justification
      - Optimal splitting of sequences
    - **Edge Cases to Consider**:
      - Empty string (usually trivially handled)
      - Single character (always a palindrome)
      - No valid partitioning exists (for word break)

## Pattern Recognition

The Palindrome Partitioning pattern appears when:

1. **Dividing a sequence** into valid segments
2. **Optimization over all possible partitions**
3. **Working with palindromic properties** of strings
4. **Minimizing cuts** or cost of partitioning
