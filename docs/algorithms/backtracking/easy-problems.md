# Backtracking - Easy Problems

## 🎯 Learning Objectives

Build a solid foundation in backtracking by mastering these fundamental concepts:

- Basic backtracking template and recursion patterns
- Generate all permutations, combinations, and subsets
- Simple constraint satisfaction problems
- Tree traversal with path tracking
- Early pruning and optimization techniques

---

## Problem 1: Generate All Subsets

**Difficulty**: 🟢 Easy  
**Pattern**: Generate All Combinations  
**Time**: O(2^n), **Space**: O(n)

=== "Problem"

    Given an integer array `nums` of **unique** elements, return all possible subsets (the power set).

    The solution set **must not** contain duplicate subsets. Return the solution in **any order**.

    **Example 1:**
    ```
    Input: nums = [1,2,3]
    Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
    ```

    **Example 2:**
    ```
    Input: nums = [0]
    Output: [[],[0]]
    ```

=== "Solution"

    ```python
    def subsets(nums):
        """
        Generate all possible subsets using backtracking.
        
        Time Complexity: O(2^n) - each element can be included or excluded
        Space Complexity: O(n) - recursion depth
        """
        result = []
        
        def backtrack(start, path):
            # Add current subset to result (make a copy)
            result.append(path[:])
            
            # Try adding each remaining element
            for i in range(start, len(nums)):
                path.append(nums[i])
                backtrack(i + 1, path)  # Move to next element
                path.pop()  # Backtrack
        
        backtrack(0, [])
        return result
    
    # Alternative iterative approach
    def subsets_iterative(nums):
        """Iterative approach using bit manipulation"""
        n = len(nums)
        result = []
        
        # Generate all 2^n possible subsets
        for i in range(1 << n):
            subset = []
            for j in range(n):
                if i & (1 << j):
                    subset.append(nums[j])
            result.append(subset)
        
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Include/Exclude Decision**: Each element can be included or excluded
    - **Path Tracking**: Maintain current subset being built
    - **Backtracking**: Undo choices to explore other possibilities
    - **Base Case**: Add current path to result at each step
    
    **Template Pattern:**
    ```python
    def backtrack(start, path):
        result.append(path[:])  # Process current state
        for i in range(start, len(choices)):
            path.append(choices[i])  # Make choice
            backtrack(i + 1, path)   # Recurse
            path.pop()               # Backtrack
    ```

---

## Problem 2: Generate All Permutations

**Difficulty**: 🟢 Easy  
**Pattern**: Generate All Arrangements  
**Time**: O(n! × n), **Space**: O(n)

=== "Problem"

    Given an array `nums` of distinct integers, return all the possible permutations. You can return the answer in **any order**.

    **Example 1:**
    ```
    Input: nums = [1,2,3]
    Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    ```

    **Example 2:**
    ```
    Input: nums = [0,1]
    Output: [[0,1],[1,0]]
    ```

=== "Solution"

    ```python
    def permute(nums):
        """
        Generate all permutations using backtracking.
        
        Time: O(n! × n) - n! permutations, O(n) to copy each
        Space: O(n) - recursion depth
        """
        result = []
        
        def backtrack(path):
            # Base case: permutation is complete
            if len(path) == len(nums):
                result.append(path[:])
                return
            
            # Try each unused number
            for num in nums:
                if num not in path:
                    path.append(num)
                    backtrack(path)
                    path.pop()
        
        backtrack([])
        return result
    
    # Optimized version using index swapping
    def permute_optimal(nums):
        """More efficient using in-place swapping"""
        result = []
        
        def backtrack(start):
            if start == len(nums):
                result.append(nums[:])
                return
            
            for i in range(start, len(nums)):
                # Swap current element to start position
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)
                # Backtrack: restore original order
                nums[start], nums[i] = nums[i], nums[start]
        
        backtrack(0)
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Used Elements**: Track which elements are already in current permutation
    - **Completion Check**: When path length equals input length
    - **Swapping Optimization**: Avoid extra space for tracking used elements
    - **Order Matters**: Unlike combinations, [1,2] ≠ [2,1]
    
    **Optimization Techniques:**
    - Use boolean array instead of `in` check for O(1) lookup
    - Swap elements in-place to avoid extra space
    - Early termination when impossible to complete

---

## Problem 3: Combinations

**Difficulty**: 🟢 Easy  
**Pattern**: Choose K from N  
**Time**: O(C(n,k) × k), **Space**: O(k)

=== "Problem"

    Given two integers `n` and `k`, return all possible combinations of `k` numbers chosen from the range `[1, n]`.

    You may return the answer in **any order**.

    **Example 1:**
    ```
    Input: n = 4, k = 2
    Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
    ```

    **Example 2:**
    ```
    Input: n = 1, k = 1
    Output: [[1]]
    ```

=== "Solution"

    ```python
    def combine(n, k):
        """
        Generate all combinations of k numbers from 1 to n.
        
        Time: O(C(n,k) × k) - C(n,k) combinations, O(k) to copy each
        Space: O(k) - recursion depth and path size
        """
        result = []
        
        def backtrack(start, path):
            # Base case: combination is complete
            if len(path) == k:
                result.append(path[:])
                return
            
            # Early pruning: not enough numbers left
            remaining_needed = k - len(path)
            remaining_available = n - start + 1
            if remaining_needed > remaining_available:
                return
            
            # Try each number from start to n
            for i in range(start, n + 1):
                path.append(i)
                backtrack(i + 1, path)
                path.pop()
        
        backtrack(1, [])
        return result
    
    # Alternative with better pruning
    def combine_pruned(n, k):
        """Enhanced version with aggressive pruning"""
        result = []
        
        def backtrack(start, path):
            if len(path) == k:
                result.append(path[:])
                return
            
            # Only iterate if we can still form a valid combination
            for i in range(start, n - (k - len(path)) + 2):
                path.append(i)
                backtrack(i + 1, path)
                path.pop()
        
        backtrack(1, [])
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Start Index**: Avoid duplicates by only considering numbers ≥ start
    - **Fixed Size**: Unlike subsets, combinations have fixed size k
    - **Early Pruning**: Stop when not enough numbers remain
    - **Mathematical Bound**: Need at least (k - current_size) more numbers
    
    **Pruning Strategy:**
    ```
    remaining_needed = k - len(path)
    remaining_available = n - start + 1
    if remaining_needed > remaining_available: return
    ```

---

## Problem 4: Letter Combinations of Phone Number

**Difficulty**: 🟢 Easy  
**Pattern**: Multiple Choice Backtracking  
**Time**: O(4^n), **Space**: O(n)

=== "Problem"

    Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent. Return the answer in **any order**.

    A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

    **Example 1:**
    ```
    Input: digits = "23"
    Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
    ```

    **Example 2:**
    ```
    Input: digits = ""
    Output: []
    ```

=== "Solution"

    ```python
    def letterCombinations(digits):
        """
        Generate all letter combinations for phone number digits.
        
        Time: O(4^n) where n is length of digits (worst case all 7,9)
        Space: O(n) for recursion depth
        """
        if not digits:
            return []
        
        # Digit to letters mapping
        phone_map = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }
        
        result = []
        
        def backtrack(index, path):
            # Base case: processed all digits
            if index == len(digits):
                result.append(path)
                return
            
            # Get current digit and its possible letters
            digit = digits[index]
            letters = phone_map[digit]
            
            # Try each letter for current digit
            for letter in letters:
                backtrack(index + 1, path + letter)
        
        backtrack(0, "")
        return result
    
    # Alternative using list for path
    def letterCombinations_list(digits):
        """Using list for path building (might be more efficient)"""
        if not digits:
            return []
        
        phone_map = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }
        
        result = []
        
        def backtrack(index, path):
            if index == len(digits):
                result.append(''.join(path))
                return
            
            digit = digits[index]
            for letter in phone_map[digit]:
                path.append(letter)
                backtrack(index + 1, path)
                path.pop()
        
        backtrack(0, [])
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Multi-Choice**: Each digit has multiple letter options
    - **Fixed Sequence**: Process digits in order (left to right)
    - **String Building**: Build result string character by character
    - **No Backtracking Needed**: Since we're building strings, no undo required
    
    **Complexity Analysis:**
    - Digits 2,3,4,5,6,8 have 3 letters each
    - Digits 7,9 have 4 letters each
    - Worst case: all digits are 7 or 9 → O(4^n)
    - Average case: mixed digits → O(3^n to 4^n)

---

## Problem 5: Binary Tree Paths

**Difficulty**: 🟢 Easy  
**Pattern**: Tree Path Tracking  
**Time**: O(n), **Space**: O(h)

=== "Problem"

    Given the `root` of a binary tree, return all root-to-leaf paths in **any order**.

    A **leaf** is a node with no children.

    **Example 1:**
    ```
    Input: root = [1,2,3,null,5]
         1
       /   \
      2     3
       \
        5
    Output: ["1->2->5","1->3"]
    ```

    **Example 2:**
    ```
    Input: root = [1]
    Output: ["1"]
    ```

=== "Solution"

    ```python
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    def binaryTreePaths(root):
        """
        Find all root-to-leaf paths in binary tree.
        
        Time: O(n) - visit each node once
        Space: O(h) - recursion depth equals tree height
        """
        if not root:
            return []
        
        result = []
        
        def backtrack(node, path):
            # Add current node to path
            path.append(str(node.val))
            
            # If leaf node, add complete path to result
            if not node.left and not node.right:
                result.append("->".join(path))
            else:
                # Recurse on children
                if node.left:
                    backtrack(node.left, path)
                if node.right:
                    backtrack(node.right, path)
            
            # Backtrack: remove current node from path
            path.pop()
        
        backtrack(root, [])
        return result
    
    # Alternative without explicit backtracking
    def binaryTreePaths_no_backtrack(root):
        """Version that builds path string directly"""
        if not root:
            return []
        
        result = []
        
        def dfs(node, path):
            if not node.left and not node.right:
                result.append(path)
                return
            
            if node.left:
                dfs(node.left, path + "->" + str(node.left.val))
            if node.right:
                dfs(node.right, path + "->" + str(node.right.val))
        
        dfs(root, str(root.val))
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Path Tracking**: Maintain current path from root to current node
    - **Leaf Detection**: Node with no left and right children
    - **String Building**: Convert path to required string format
    - **Tree Backtracking**: Remove node when returning from recursion
    
    **Two Approaches:**
    1. **Explicit Backtracking**: Modify shared path list
    2. **String Passing**: Build new string for each recursive call
    
    **Edge Cases:**
    - Empty tree (root is null)
    - Single node tree (root is only node)
    - Skewed tree (essentially a linked list)

---

## Problem 6: Combination Sum

**Difficulty**: 🟢 Easy  
**Pattern**: Unlimited Choice Backtracking  
**Time**: O(n^(target/min)), **Space**: O(target/min)

=== "Problem"

    Given an array of **distinct** integers `candidates` and a `target` integer, return a list of all **unique combinations** of `candidates` where the chosen numbers sum to `target`. You may return the combinations in **any order**.

    The **same** number may be chosen from `candidates` an **unlimited number of times**. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

    **Example 1:**
    ```
    Input: candidates = [2,3,6,7], target = 7
    Output: [[2,2,3],[7]]
    ```

    **Example 2:**
    ```
    Input: candidates = [2,3,5], target = 8
    Output: [[2,2,2,2],[2,3,3],[3,5]]
    ```

=== "Solution"

    ```python
    def combinationSum(candidates, target):
        """
        Find all unique combinations that sum to target.
        
        Time: O(n^(target/min)) where min is smallest candidate
        Space: O(target/min) for recursion depth
        """
        result = []
        
        def backtrack(start, path, remaining):
            # Base case: found valid combination
            if remaining == 0:
                result.append(path[:])
                return
            
            # Early pruning: remaining becomes negative
            if remaining < 0:
                return
            
            # Try each candidate starting from start index
            for i in range(start, len(candidates)):
                candidate = candidates[i]
                
                # Skip if candidate is larger than remaining
                if candidate > remaining:
                    continue
                
                path.append(candidate)
                # Can reuse same candidate (start=i, not i+1)
                backtrack(i, path, remaining - candidate)
                path.pop()
        
        backtrack(0, [], target)
        return result
    
    # Optimized version with sorting
    def combinationSum_optimized(candidates, target):
        """Optimized with sorting for better pruning"""
        candidates.sort()  # Sort for early termination
        result = []
        
        def backtrack(start, path, remaining):
            if remaining == 0:
                result.append(path[:])
                return
            
            for i in range(start, len(candidates)):
                candidate = candidates[i]
                
                # Early termination: since sorted, all remaining are larger
                if candidate > remaining:
                    break
                
                path.append(candidate)
                backtrack(i, path, remaining - candidate)
                path.pop()
        
        backtrack(0, [], target)
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Unlimited Reuse**: Same number can be used multiple times
    - **Avoid Duplicates**: Use start index to maintain order
    - **Early Pruning**: Stop when remaining becomes negative
    - **Sorting Optimization**: Enables early termination
    
    **Critical Insight:**
    ```
    backtrack(i, path, remaining - candidate)  # i, not i+1
    ```
    This allows reusing the same candidate multiple times.
    
    **Pruning Strategies:**
    1. Skip candidates larger than remaining target
    2. Sort candidates for early termination
    3. Use remaining sum instead of current sum

---

## Problem 7: Generate Parentheses

**Difficulty**: 🟢 Easy  
**Pattern**: Constraint-Based Generation  
**Time**: O(4^n / √n), **Space**: O(n)

=== "Problem"

    Given `n` pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

    **Example 1:**
    ```
    Input: n = 3
    Output: ["((()))","(()())","(())()","()(())","()()()"]
    ```

    **Example 2:**
    ```
    Input: n = 1
    Output: ["()"]
    ```

=== "Solution"

    ```python
    def generateParenthesis(n):
        """
        Generate all valid parentheses combinations.
        
        Time: O(4^n / √n) - Catalan number
        Space: O(n) - recursion depth
        """
        result = []
        
        def backtrack(path, open_count, close_count):
            # Base case: used all n pairs
            if len(path) == 2 * n:
                result.append(path)
                return
            
            # Add opening parenthesis if we can
            if open_count < n:
                backtrack(path + "(", open_count + 1, close_count)
            
            # Add closing parenthesis if valid
            if close_count < open_count:
                backtrack(path + ")", open_count, close_count + 1)
        
        backtrack("", 0, 0)
        return result
    
    # Alternative using list for path building
    def generateParenthesis_list(n):
        """Using list for potentially better performance"""
        result = []
        
        def backtrack(path, open_count, close_count):
            if len(path) == 2 * n:
                result.append(''.join(path))
                return
            
            if open_count < n:
                path.append('(')
                backtrack(path, open_count + 1, close_count)
                path.pop()
            
            if close_count < open_count:
                path.append(')')
                backtrack(path, open_count, close_count + 1)
                path.pop()
        
        backtrack([], 0, 0)
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Valid Constraints**: 
      - At most n opening parentheses
      - Closing only when there's unmatched opening
      - Total length is exactly 2n
    
    - **State Tracking**: Count open and close parentheses used
    - **Pruning**: Only add valid characters at each step
    
    **Validity Rules:**
    1. Can add '(' if open_count < n
    2. Can add ')' if close_count < open_count
    3. Never add ')' if no unmatched '(' exists
    
    **Time Complexity**: This is the n-th Catalan number: O(4^n / √n)

---

## Problem 8: Word Search

**Difficulty**: 🟢 Easy  
**Pattern**: 2D Grid Backtracking  
**Time**: O(n × m × 4^L), **Space**: O(L)

=== "Problem"

    Given an `m x n` grid of characters `board` and a string `word`, return `true` if `word` exists in the grid.

    The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

    **Example 1:**
    ```
    Input: board = [["A","B","C","E"],
                   ["S","F","C","S"],
                   ["A","D","E","E"]], word = "ABCCED"
    Output: true
    ```

    **Example 2:**
    ```
    Input: board = [["A","B","C","E"],
                   ["S","F","C","S"],
                   ["A","D","E","E"]], word = "SEE"
    Output: true
    ```

=== "Solution"

    ```python
    def exist(board, word):
        """
        Search for word in 2D grid using backtracking.
        
        Time: O(n × m × 4^L) where L is word length
        Space: O(L) for recursion depth
        """
        if not board or not board[0] or not word:
            return False
        
        rows, cols = len(board), len(board[0])
        
        def backtrack(row, col, index):
            # Base case: found complete word
            if index == len(word):
                return True
            
            # Check bounds and character match
            if (row < 0 or row >= rows or col < 0 or col >= cols or
                board[row][col] != word[index]):
                return False
            
            # Mark cell as visited
            temp = board[row][col]
            board[row][col] = '#'
            
            # Explore all 4 directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            found = False
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if backtrack(new_row, new_col, index + 1):
                    found = True
                    break
            
            # Restore cell (backtrack)
            board[row][col] = temp
            return found
        
        # Try starting from each cell
        for i in range(rows):
            for j in range(cols):
                if backtrack(i, j, 0):
                    return True
        
        return False
    
    # Alternative without modifying board
    def exist_with_visited_set(board, word):
        """Version using visited set instead of modifying board"""
        if not board or not board[0] or not word:
            return False
        
        rows, cols = len(board), len(board[0])
        
        def backtrack(row, col, index, visited):
            if index == len(word):
                return True
            
            if (row < 0 or row >= rows or col < 0 or col >= cols or
                (row, col) in visited or board[row][col] != word[index]):
                return False
            
            visited.add((row, col))
            
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dr, dc in directions:
                if backtrack(row + dr, col + dc, index + 1, visited):
                    visited.remove((row, col))
                    return True
            
            visited.remove((row, col))
            return False
        
        for i in range(rows):
            for j in range(cols):
                if backtrack(i, j, 0, set()):
                    return True
        
        return False
    ```

=== "Insights"

    **Key Concepts:**
    
    - **2D Exploration**: Move in 4 directions (up, down, left, right)
    - **Visited Marking**: Prevent revisiting cells in current path
    - **Backtracking**: Restore state when backtracking
    - **Multiple Starting Points**: Try every cell as potential start
    
    **State Management:**
    1. **Modify Board**: Mark visited with special character
    2. **Visited Set**: Track coordinates in separate set
    
    **Optimization Tips:**
    - Early termination when character doesn't match
    - Character frequency optimization for large boards
    - Try starting with less frequent characters

---

## Problem 9: Palindrome Partitioning

**Difficulty**: 🟢 Easy  
**Pattern**: String Partitioning  
**Time**: O(n × 2^n), **Space**: O(n)

=== "Problem"

    Given a string `s`, partition `s` such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of `s`.

    **Example 1:**
    ```
    Input: s = "aab"
    Output: [["a","a","b"],["aa","b"]]
    ```

    **Example 2:**
    ```
    Input: s = "raceacar"
    Output: [["r","a","c","e","a","c","a","r"],["r","a","ce","c","a","r"],["r","ace","ca","r"],["raceacar"]]
    ```

=== "Solution"

    ```python
    def partition(s):
        """
        Find all palindrome partitions of string.
        
        Time: O(n × 2^n) - 2^n partitions, O(n) to check palindrome
        Space: O(n) - recursion depth
        """
        result = []
        
        def is_palindrome(string):
            """Check if string is palindrome"""
            return string == string[::-1]
        
        def backtrack(start, path):
            # Base case: processed entire string
            if start == len(s):
                result.append(path[:])
                return
            
            # Try all possible endings for current substring
            for end in range(start + 1, len(s) + 1):
                substring = s[start:end]
                
                # Only proceed if current substring is palindrome
                if is_palindrome(substring):
                    path.append(substring)
                    backtrack(end, path)
                    path.pop()
        
        backtrack(0, [])
        return result
    
    # Optimized version with palindrome preprocessing
    def partition_optimized(s):
        """Optimized with precomputed palindrome check"""
        n = len(s)
        
        # Precompute palindrome check table
        is_palindrome = [[False] * n for _ in range(n)]
        
        # Every single character is palindrome
        for i in range(n):
            is_palindrome[i][i] = True
        
        # Check for 2-character palindromes
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                is_palindrome[i][i + 1] = True
        
        # Check for palindromes of length 3 and more
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                    is_palindrome[i][j] = True
        
        result = []
        
        def backtrack(start, path):
            if start == n:
                result.append(path[:])
                return
            
            for end in range(start, n):
                if is_palindrome[start][end]:
                    path.append(s[start:end + 1])
                    backtrack(end + 1, path)
                    path.pop()
        
        backtrack(0, [])
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **String Partitioning**: Split string into non-overlapping substrings
    - **Palindrome Check**: Validate each potential substring
    - **Dynamic Programming**: Precompute palindrome checks for efficiency
    - **Substring Generation**: Try all possible split points
    
    **Optimization Strategies:**
    1. **Precompute Palindromes**: Use DP table for O(1) lookup
    2. **Early Pruning**: Skip non-palindrome substrings immediately
    3. **Memoization**: Cache results for repeated subproblems
    
    **Palindrome Check Optimization:**
    ```
    is_palindrome[i][j] = (s[i] == s[j]) && is_palindrome[i+1][j-1]
    ```

---

## Problem 10: Letter Case Permutation

**Difficulty**: 🟢 Easy  
**Pattern**: Character Choice Backtracking  
**Time**: O(2^n), **Space**: O(n)

=== "Problem"

    Given a string `s`, you can transform every letter individually to be lowercase or uppercase to create another string.

    Return a list of all possible strings we could create. Return the output in **any order**.

    **Example 1:**
    ```
    Input: s = "a1b2"
    Output: ["a1b2","a1B2","A1b2","A1B2"]
    ```

    **Example 2:**
    ```
    Input: s = "3z4"
    Output: ["3z4","3Z4"]
    ```

=== "Solution"

    ```python
    def letterCasePermutation(s):
        """
        Generate all case permutations of string.
        
        Time: O(2^n) where n is number of letters
        Space: O(n) for recursion depth
        """
        result = []
        
        def backtrack(index, path):
            # Base case: processed all characters
            if index == len(s):
                result.append(''.join(path))
                return
            
            char = s[index]
            
            if char.isalpha():
                # For letters, try both cases
                path.append(char.lower())
                backtrack(index + 1, path)
                path.pop()
                
                path.append(char.upper())
                backtrack(index + 1, path)
                path.pop()
            else:
                # For digits, only one choice
                path.append(char)
                backtrack(index + 1, path)
                path.pop()
        
        backtrack(0, [])
        return result
    
    # Alternative without explicit backtracking
    def letterCasePermutation_iterative(s):
        """Iterative approach using bit manipulation concept"""
        result = ['']
        
        for char in s:
            if char.isalpha():
                # For each existing string, add both cases
                result = [string + char.lower() for string in result] + \
                        [string + char.upper() for string in result]
            else:
                # For digits, just append to all existing strings
                result = [string + char for string in result]
        
        return result
    
    # BFS-style approach
    def letterCasePermutation_bfs(s):
        """BFS approach building strings level by level"""
        from collections import deque
        
        queue = deque([''])
        
        for char in s:
            size = len(queue)
            for _ in range(size):
                current = queue.popleft()
                
                if char.isalpha():
                    queue.append(current + char.lower())
                    queue.append(current + char.upper())
                else:
                    queue.append(current + char)
        
        return list(queue)
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Character Classification**: Letters have 2 choices, digits have 1
    - **Binary Decision**: For each letter, choose lowercase or uppercase
    - **String Building**: Construct result character by character
    - **Multiple Approaches**: Recursive, iterative, or BFS
    
    **Complexity Analysis:**
    - If there are k letters in string of length n
    - Time: O(2^k × n) - 2^k combinations, O(n) to build each
    - Space: O(2^k × n) - store all combinations
    
    **Pattern Recognition:**
    This is similar to subsets problem where each letter gives binary choice.

---

## Problem 11: Restore IP Addresses

**Difficulty**: 🟢 Easy  
**Pattern**: String Segmentation with Validation  
**Time**: O(1), **Space**: O(1)

=== "Problem"

    A **valid IP address** consists of exactly four integers separated by single dots. Each integer is between `0` and `255` (**inclusive**) and cannot have leading zeros.

    Given a string `s` containing only digits, return all possible valid IP addresses that can be formed by inserting dots into `s`. You are **not** allowed to reorder or remove any digits in `s`.

    **Example 1:**
    ```
    Input: s = "25525511135"
    Output: ["255.255.11.135","255.255.111.35"]
    ```

    **Example 2:**
    ```
    Input: s = "0000"
    Output: ["0.0.0.0"]
    ```

=== "Solution"

    ```python
    def restoreIpAddresses(s):
        """
        Generate all valid IP addresses from digit string.
        
        Time: O(1) - at most 3^4 = 81 combinations to check
        Space: O(1) - constant space for recursion
        """
        result = []
        n = len(s)
        
        # Early pruning: invalid length
        if n < 4 or n > 12:
            return result
        
        def is_valid_segment(segment):
            """Check if segment is valid IP part"""
            if not segment:
                return False
            
            # Check for leading zeros (except "0" itself)
            if len(segment) > 1 and segment[0] == '0':
                return False
            
            # Check range [0, 255]
            num = int(segment)
            return 0 <= num <= 255
        
        def backtrack(start, path):
            # Base case: need exactly 4 segments
            if len(path) == 4:
                if start == n:  # Used all characters
                    result.append('.'.join(path))
                return
            
            # Early pruning: too many/few characters remaining
            remaining_segments = 4 - len(path)
            remaining_chars = n - start
            
            if remaining_chars < remaining_segments or remaining_chars > remaining_segments * 3:
                return
            
            # Try segments of length 1, 2, 3
            for length in range(1, 4):
                if start + length > n:
                    break
                
                segment = s[start:start + length]
                
                if is_valid_segment(segment):
                    path.append(segment)
                    backtrack(start + length, path)
                    path.pop()
        
        backtrack(0, [])
        return result
    
    # Alternative implementation with different pruning
    def restoreIpAddresses_alt(s):
        """Alternative with explicit segment bounds"""
        result = []
        
        def backtrack(start, segments):
            if len(segments) == 4:
                if start == len(s):
                    result.append('.'.join(segments))
                return
            
            # Try different segment lengths
            for i in range(start, min(start + 3, len(s))):
                segment = s[start:i + 1]
                
                # Validate segment
                if (len(segment) == 1 or 
                    (segment[0] != '0' and int(segment) <= 255)):
                    segments.append(segment)
                    backtrack(i + 1, segments)
                    segments.pop()
        
        backtrack(0, [])
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Fixed Structure**: Exactly 4 segments separated by dots
    - **Segment Validation**: Each part must be 0-255 without leading zeros
    - **Length Constraints**: String length must be 4-12 characters
    - **Aggressive Pruning**: Use mathematical bounds to eliminate impossible cases
    
    **Validation Rules:**
    1. No leading zeros (except "0" itself)
    2. Value must be 0 ≤ value ≤ 255
    3. Exactly 4 segments required
    4. Use all characters exactly once
    
    **Pruning Strategy:**
    ```
    remaining_segments = 4 - current_segments
    remaining_chars = total_length - current_position
    
    # Too few chars: can't fill remaining segments
    if remaining_chars < remaining_segments: return
    
    # Too many chars: can't fit in remaining segments
    if remaining_chars > remaining_segments * 3: return
    ```

---

## Problem 12: Beautiful Arrangement

**Difficulty**: 🟢 Easy  
**Pattern**: Position-Based Permutation  
**Time**: O(k), **Space**: O(n)

=== "Problem"

    Suppose you have `n` integers labeled `1` through `n`. A permutation of those `n` integers `perm` (**1-indexed**) is considered a **beautiful arrangement** if for every `i` (1 ≤ i ≤ n), **either**:

    - `perm[i]` is divisible by `i`, or
    - `i` is divisible by `perm[i]`.

    Given an integer `n`, return the number of the **beautiful arrangements** that you can construct.

    **Example 1:**
    ```
    Input: n = 2
    Output: 2
    Explanation: [1,2] and [2,1] are beautiful arrangements.
    ```

    **Example 2:**
    ```
    Input: n = 1
    Output: 1
    ```

=== "Solution"

    ```python
    def countArrangement(n):
        """
        Count beautiful arrangements using backtracking.
        
        Time: O(k) where k is much less than n! due to constraints
        Space: O(n) for recursion depth and visited array
        """
        def backtrack(position, used):
            # Base case: filled all positions
            if position > n:
                return 1
            
            count = 0
            # Try each unused number at current position
            for num in range(1, n + 1):
                if not used[num] and (num % position == 0 or position % num == 0):
                    used[num] = True
                    count += backtrack(position + 1, used)
                    used[num] = False
            
            return count
        
        used = [False] * (n + 1)  # 1-indexed
        return backtrack(1, used)
    
    # Optimized version working backwards
    def countArrangement_optimized(n):
        """Optimized by filling positions from n to 1"""
        def backtrack(position, used):
            if position == 0:
                return 1
            
            count = 0
            for num in range(1, n + 1):
                if not used[num] and (num % position == 0 or position % num == 0):
                    used[num] = True
                    count += backtrack(position - 1, used)
                    used[num] = False
            
            return count
        
        used = [False] * (n + 1)
        return backtrack(n, used)
    
    # Version with memoization
    def countArrangement_memo(n):
        """With memoization for overlapping subproblems"""
        memo = {}
        
        def backtrack(position, used_mask):
            if position > n:
                return 1
            
            if (position, used_mask) in memo:
                return memo[(position, used_mask)]
            
            count = 0
            for num in range(1, n + 1):
                if not (used_mask & (1 << num)) and (num % position == 0 or position % num == 0):
                    new_mask = used_mask | (1 << num)
                    count += backtrack(position + 1, new_mask)
            
            memo[(position, used_mask)] = count
            return count
        
        return backtrack(1, 0)
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Constraint Checking**: Both divisibility conditions for beautiful arrangement
    - **Position vs Value**: Consider which number goes in which position
    - **State Compression**: Use bitmask to represent used numbers
    - **Memoization**: Cache results based on position and used set
    
    **Beautiful Condition:**
    ```
    (num % position == 0) OR (position % num == 0)
    ```
    
    **Optimization Techniques:**
    1. **Work Backwards**: Fill from position n to 1 for better pruning
    2. **Bit Masking**: Use integer to represent used number set
    3. **Memoization**: Cache based on (position, used_mask) state
    
    **Why Time Complexity is O(k) not O(n!):**
    The divisibility constraints significantly reduce the search space, making k much smaller than n!.

---

## Problem 13: N-Queens Easy Version (Count Solutions)

**Difficulty**: 🟢 Easy  
**Pattern**: Constraint Satisfaction  
**Time**: O(n!), **Space**: O(n)

=== "Problem"

    The **n-queens** puzzle is the problem of placing `n` queens on an `n x n` chessboard such that no two queens attack each other.

    Given an integer `n`, return the number of distinct solutions to the **n-queens puzzle**.

    **Example 1:**
    ```
    Input: n = 4
    Output: 2
    Explanation: There are two distinct solutions to the 4-queens puzzle.
    ```

    **Example 2:**
    ```
    Input: n = 1
    Output: 1
    ```

=== "Solution"

    ```python
    def totalNQueens(n):
        """
        Count solutions to N-Queens problem.
        
        Time: O(n!) in worst case, but much less due to pruning
        Space: O(n) for recursion depth and constraint tracking
        """
        def backtrack(row, cols, diag1, diag2):
            # Base case: placed all queens
            if row == n:
                return 1
            
            count = 0
            # Try placing queen in each column of current row
            for col in range(n):
                # Check if position is safe
                diagonal1 = row - col  # Top-left to bottom-right
                diagonal2 = row + col  # Top-right to bottom-left
                
                if col in cols or diagonal1 in diag1 or diagonal2 in diag2:
                    continue  # Position is under attack
                
                # Place queen (mark constraints)
                cols.add(col)
                diag1.add(diagonal1)
                diag2.add(diagonal2)
                
                # Recurse to next row
                count += backtrack(row + 1, cols, diag1, diag2)
                
                # Remove queen (backtrack)
                cols.remove(col)
                diag1.remove(diagonal1)
                diag2.remove(diagonal2)
            
            return count
        
        return backtrack(0, set(), set(), set())
    
    # Alternative using arrays instead of sets
    def totalNQueens_arrays(n):
        """Using boolean arrays for constraint tracking"""
        cols = [False] * n
        diag1 = [False] * (2 * n - 1)  # row - col + (n-1)
        diag2 = [False] * (2 * n - 1)  # row + col
        
        def backtrack(row):
            if row == n:
                return 1
            
            count = 0
            for col in range(n):
                d1_idx = row - col + n - 1
                d2_idx = row + col
                
                if cols[col] or diag1[d1_idx] or diag2[d2_idx]:
                    continue
                
                # Place queen
                cols[col] = diag1[d1_idx] = diag2[d2_idx] = True
                count += backtrack(row + 1)
                
                # Remove queen
                cols[col] = diag1[d1_idx] = diag2[d2_idx] = False
            
            return count
        
        return backtrack(0)
    
    # Bit manipulation version for maximum efficiency
    def totalNQueens_bits(n):
        """Ultra-optimized using bit manipulation"""
        def backtrack(row, cols, diag1, diag2):
            if row == n:
                return 1
            
            # Available positions (not under attack)
            available = ((1 << n) - 1) & ~(cols | diag1 | diag2)
            count = 0
            
            while available:
                # Get rightmost available position
                pos = available & -available
                available ^= pos  # Remove this position
                
                count += backtrack(row + 1, 
                                 cols | pos,
                                 (diag1 | pos) << 1,
                                 (diag2 | pos) >> 1)
            
            return count
        
        return backtrack(0, 0, 0, 0)
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Constraint Propagation**: Track columns and diagonals under attack
    - **Row-by-Row Placement**: Place one queen per row to avoid row conflicts
    - **Diagonal Indexing**: 
      - Main diagonal: `row - col` (constant for each diagonal)
      - Anti-diagonal: `row + col` (constant for each diagonal)
    - **Bit Manipulation**: Use bits for extremely fast constraint checking
    
    **Attack Patterns:**
    - **Same Column**: Same col value
    - **Main Diagonal**: Same (row - col) value  
    - **Anti-diagonal**: Same (row + col) value
    
    **Optimization Levels:**
    1. **Sets**: Clean and readable
    2. **Arrays**: Faster than sets
    3. **Bit Manipulation**: Maximum performance
    
    **Diagonal Indexing Formula:**
    ```
    diag1_index = row - col + (n - 1)  # Offset to make non-negative
    diag2_index = row + col
    ```

---

## Problem 14: Gray Code

**Difficulty**: 🟢 Easy  
**Pattern**: Sequence Generation  
**Time**: O(2^n), **Space**: O(2^n)

=== "Problem"

    An **n-bit gray code sequence** is a sequence of `2^n` integers where:

    - Every integer is in the **inclusive** range `[0, 2^n - 1]`,
    - The first integer is `0`,
    - An integer appears **no more than once** in the sequence,
    - The binary representation of every pair of **adjacent** integers differs by **exactly one bit**, and
    - The binary representation of the **first** and **last** integers differs by **exactly one bit**.

    Given an integer `n`, return _any valid **n-bit gray code sequence**_.

    **Example 1:**
    ```
    Input: n = 2
    Output: [0,1,3,2]
    Explanation: 00 - 01 - 11 - 10
    ```

=== "Solution"

    ```python
    def grayCode(n):
        """
        Generate n-bit Gray code sequence.
        
        Time: O(2^n) - generate 2^n numbers
        Space: O(2^n) - store all numbers
        """
        # Mathematical approach using Gray code formula
        result = []
        for i in range(1 << n):  # 2^n numbers
            # Gray code formula: i XOR (i >> 1)
            gray = i ^ (i >> 1)
            result.append(gray)
        return result
    
    # Recursive approach building from smaller cases
    def grayCode_recursive(n):
        """Build Gray code recursively"""
        if n == 0:
            return [0]
        
        # Get Gray code for n-1 bits
        prev_gray = grayCode_recursive(n - 1)
        
        # First half: same as n-1 case
        result = prev_gray[:]
        
        # Second half: reverse order with MSB set
        msb = 1 << (n - 1)
        for i in range(len(prev_gray) - 1, -1, -1):
            result.append(msb | prev_gray[i])
        
        return result
    
    # Iterative building approach
    def grayCode_iterative(n):
        """Build Gray code iteratively"""
        result = [0]
        
        for i in range(n):
            # Add numbers in reverse order with i-th bit set
            size = len(result)
            for j in range(size - 1, -1, -1):
                result.append(result[j] | (1 << i))
        
        return result
    
    # Backtracking approach (educational, not optimal)
    def grayCode_backtrack(n):
        """Backtracking approach for understanding"""
        if n == 0:
            return [0]
        
        target_size = 1 << n
        used = set()
        result = []
        
        def backtrack(current):
            if len(result) == target_size:
                # Check if last connects to first
                return bin(result[0]).count('1') - bin(result[-1]).count('1') == 0
            
            # Try all numbers that differ by exactly 1 bit
            for i in range(n):
                next_num = current ^ (1 << i)
                
                if next_num not in used:
                    used.add(next_num)
                    result.append(next_num)
                    
                    if backtrack(next_num):
                        return True
                    
                    result.pop()
                    used.remove(next_num)
            
            return False
        
        used.add(0)
        result.append(0)
        backtrack(0)
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Gray Code Formula**: `gray[i] = i XOR (i >> 1)`
    - **Recursive Construction**: Build from smaller Gray codes
    - **Bit Manipulation**: Use XOR and bit shifts efficiently
    - **Hamilton Path**: Gray code is Hamiltonian path on n-cube
    
    **Construction Methods:**
    
    1. **Mathematical Formula**: Direct computation using bit operations
    2. **Recursive Building**: 
       ```
       G(n) = [G(n-1), reverse(G(n-1)) with MSB set]
       ```
    3. **Iterative Building**: Add one bit at a time
    4. **Backtracking**: Educational approach (inefficient)
    
    **Gray Code Properties:**
    - Adjacent numbers differ by exactly 1 bit
    - Forms a cycle (first and last also differ by 1 bit)
    - Total of 2^n unique numbers in range [0, 2^n - 1]
    
    **Applications:**
    - Digital circuits (minimizing switching)
    - Error correction codes
    - Puzzles and combinatorial problems

---

## Problem 15: Factor Combinations

**Difficulty**: 🟢 Easy  
**Pattern**: Number Factorization  
**Time**: O(√n), **Space**: O(log n)

=== "Problem"

    Numbers can be regarded as the product of their factors.

    - For example, `8 = 2 x 2 x 2 = 2 x 4`.

    Given an integer `n`, return _all possible combinations of its factors_.

    You may return the answer in **any order**.

    **Note** that the factors should be in the range `[2, n - 1]`.

    **Example 1:**
    ```
    Input: n = 1
    Output: []
    ```

    **Example 2:**
    ```
    Input: n = 12
    Output: [[2,6],[2,2,3],[3,4]]
    ```

=== "Solution"

    ```python
    def getFactors(n):
        """
        Find all factor combinations of n.
        
        Time: O(√n) for each recursive call
        Space: O(log n) for recursion depth
        """
        result = []
        
        def backtrack(num, start, path):
            # Base case: if num becomes 1, we have a valid factorization
            if num == 1:
                if len(path) > 1:  # Must have at least 2 factors
                    result.append(path[:])
                return
            
            # Try factors from start to sqrt(num)
            for i in range(start, int(num**0.5) + 1):
                if num % i == 0:
                    path.append(i)
                    backtrack(num // i, i, path)
                    path.pop()
            
            # If num > start, it can be a factor itself
            if num >= start:
                path.append(num)
                backtrack(1, num, path)
                path.pop()
        
        backtrack(n, 2, [])
        return result
    
    # Alternative cleaner implementation
    def getFactors_clean(n):
        """Cleaner implementation with helper function"""
        def factorize(num, start):
            """Return all factorizations of num starting from start"""
            if num == 1:
                return [[]]
            
            result = []
            
            # Try each potential factor
            for factor in range(start, int(num**0.5) + 1):
                if num % factor == 0:
                    quotient = num // factor
                    
                    # Get all factorizations of quotient
                    for sub_factors in factorize(quotient, factor):
                        result.append([factor] + sub_factors)
            
            # The number itself can be a single factor
            if num >= start:
                result.append([num])
            
            return result
        
        factors = factorize(n, 2)
        # Filter out single-element factorizations (the number itself)
        return [f for f in factors if len(f) > 1]
    
    # Iterative approach using stack
    def getFactors_iterative(n):
        """Iterative approach using explicit stack"""
        if n <= 3:
            return []
        
        result = []
        stack = [(n, 2, [])]  # (remaining, start_factor, current_path)
        
        while stack:
            num, start, path = stack.pop()
            
            # Try all factors from start to sqrt(num)
            for i in range(start, int(num**0.5) + 1):
                if num % i == 0:
                    quotient = num // i
                    new_path = path + [i]
                    
                    if quotient < i:
                        continue
                    elif quotient == i:
                        result.append(new_path + [i])
                    else:
                        result.append(new_path + [quotient])
                        if quotient > i:
                            stack.append((quotient, i, new_path))
        
        return result
    ```

=== "Insights"

    **Key Concepts:**
    
    - **Factor Pairs**: For each factor `i`, consider `n/i` as complement
    - **Avoid Duplicates**: Use start index to maintain order
    - **Square Root Bound**: Only check factors up to √n
    - **Recursive Decomposition**: Break problem into smaller subproblems
    
    **Algorithm Strategy:**
    1. Try each potential factor from 2 to √n
    2. For valid factor i, recursively factorize n/i
    3. Use start index to ensure factors are in non-decreasing order
    4. Handle the case where remaining number is itself a factor
    
    **Edge Cases:**
    - n = 1: No factorizations (empty result)
    - Prime numbers: Only factorization is [n] (excluded)
    - Perfect squares: Handle i = √n case carefully
    
    **Optimization Tips:**
    - Early termination when factor > √remaining
    - Use integer division and modulo efficiently
    - Consider memoization for repeated factorizations

---

## 📝 Summary

### Core Backtracking Patterns

| **Pattern Type** | **Key Characteristics** | **Example Problems** |
|------------------|------------------------|---------------------|
| **Generation** | Build all possible combinations | Subsets, Permutations, Combinations |
| **Constraint Satisfaction** | Find solutions meeting constraints | N-Queens, Sudoku, Word Search |
| **Path Finding** | Explore paths with backtracking | Binary Tree Paths, Maze Solving |
| **Partitioning** | Split input into valid parts | Palindrome Partitioning, IP Addresses |
| **Choice Making** | Make binary/multiple choices | Letter Case, Phone Numbers |

### Universal Backtracking Template

```python
def backtrack(state, path, choices):
    # Base case: solution found or impossible
    if is_complete(state):
        process_solution(path)
        return
    
    if is_invalid(state):
        return
    
    # Try each choice
    for choice in get_valid_choices(choices, state):
        # Make choice
        path.append(choice)
        new_state = update_state(state, choice)
        
        # Recurse
        backtrack(new_state, path, remaining_choices)
        
        # Backtrack (undo choice)
        path.pop()
        restore_state(state, choice)
```

### Optimization Techniques

1. **Early Pruning**: Eliminate impossible branches early
2. **Constraint Propagation**: Use constraints to reduce search space
3. **Ordering**: Try most promising choices first
4. **Memoization**: Cache results for overlapping subproblems
5. **Bit Manipulation**: Use bits for efficient state representation

### Time Complexity Patterns

- **Subsets**: O(2^n) - binary choice for each element
- **Permutations**: O(n!) - factorial arrangements
- **Combinations**: O(C(n,k)) - binomial coefficient
- **Constrained Problems**: Often much better than worst case due to pruning

### Problem-Solving Strategy

1. **Identify the Choices**: What decisions need to be made at each step?
2. **Define the State**: What information needs to be tracked?
3. **Establish Base Cases**: When is a solution complete/invalid?
4. **Design Pruning**: How can impossible branches be eliminated early?
5. **Implement Backtracking**: Make choice → recurse → undo choice

---

## 🎯 Next Steps

- **[Medium Backtracking Problems](medium-problems.md)** - More complex constraint satisfaction
- **[Advanced Techniques](../advanced/index.md)** - Constraint propagation, branch and bound
- **[Dynamic Programming](../dp/index.md)** - When backtracking meets memoization

Master these easy problems before moving to medium difficulty. The patterns you learn here form the foundation for all advanced backtracking algorithms!
