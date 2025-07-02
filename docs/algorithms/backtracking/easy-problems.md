# Backtracking - Easy Problems

## Problem Categories

### 1. Basic Generation Problems
- Subsets
- Permutations
- Combinations
- Phone number letter combinations

### 2. Simple Search Problems
- Binary tree paths
- Path sum
- Sum of root to leaf numbers

---

## 1. Generate All Subsets

**Problem**: Given an integer array `nums` of unique elements, return all possible subsets.

**Example**:
```
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**Solution**:
```python
def subsets(nums):
    """
    Generate all possible subsets using backtracking.
    
    Time Complexity: O(2^n) - each element can be included or excluded
    Space Complexity: O(n) - recursion depth
    """
    result = []
    
    def backtrack(start, path):
        # Add current subset to result
        result.append(path[:])
        
        # Try adding each remaining element
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)  # Move to next element
            path.pop()  # Backtrack
    
    backtrack(0, [])
    return result

# Test
print(subsets([1, 2, 3]))
# Output: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
```

**Key Points**:
- Include current subset before trying extensions
- Use `start` parameter to avoid duplicates
- Backtrack by removing last added element

---

## 2. Generate All Permutations

**Problem**: Given an array `nums` of distinct integers, return all possible permutations.

**Example**:
```
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**Solution**:
```python
def permute(nums):
    """
    Generate all permutations using backtracking.
    
    Time Complexity: O(n! * n) - n! permutations, each takes O(n) to construct
    Space Complexity: O(n) - recursion depth
    """
    result = []
    
    def backtrack(path, remaining):
        # Base case: no more elements to add
        if not remaining:
            result.append(path[:])
            return
        
        # Try each remaining element
        for i in range(len(remaining)):
            # Choose element
            path.append(remaining[i])
            # Recurse with remaining elements
            backtrack(path, remaining[:i] + remaining[i+1:])
            # Backtrack
            path.pop()
    
    backtrack([], nums)
    return result

# Alternative using visited array
def permute_v2(nums):
    result = []
    
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            
            # Choose
            path.append(nums[i])
            used[i] = True
            
            # Recurse
            backtrack(path, used)
            
            # Backtrack
            path.pop()
            used[i] = False
    
    backtrack([], [False] * len(nums))
    return result
```

**Key Points**:
- Base case: when path length equals input length
- Track used elements to avoid repetition
- Two approaches: slicing array or using visited array

---

## 3. Combinations

**Problem**: Given two integers `n` and `k`, return all possible combinations of `k` numbers out of the range `[1, n]`.

**Example**:
```
Input: n = 4, k = 2
Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
```

**Solution**:
```python
def combine(n, k):
    """
    Generate all combinations of k numbers from 1 to n.
    
    Time Complexity: O(C(n,k) * k) - C(n,k) combinations, each takes O(k) to construct
    Space Complexity: O(k) - recursion depth
    """
    result = []
    
    def backtrack(start, path):
        # Base case: combination is complete
        if len(path) == k:
            result.append(path[:])
            return
        
        # Try each number from start to n
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)  # Next number must be larger
            path.pop()
    
    backtrack(1, [])
    return result

# Optimized version with pruning
def combine_optimized(n, k):
    result = []
    
    def backtrack(start, path):
        # Pruning: if remaining slots > remaining numbers, skip
        if len(path) + (n - start + 1) < k:
            return
        
        if len(path) == k:
            result.append(path[:])
            return
        
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(1, [])
    return result
```

**Key Points**:
- Use `start` parameter to ensure increasing order
- Pruning optimization to skip impossible branches
- Base case checks if enough elements selected

---

## 4. Letter Combinations of Phone Number

**Problem**: Given a string containing digits from 2-9, return all possible letter combinations that the number could represent.

**Example**:
```
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

**Solution**:
```python
def letterCombinations(digits):
    """
    Generate all letter combinations for phone number digits.
    
    Time Complexity: O(3^n * 4^m) where n is digits with 3 letters, m is digits with 4 letters
    Space Complexity: O(n) - recursion depth
    """
    if not digits:
        return []
    
    # Mapping of digits to letters
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
        
        # Get current digit and its letters
        current_digit = digits[index]
        letters = phone_map[current_digit]
        
        # Try each letter
        for letter in letters:
            backtrack(index + 1, path + letter)
    
    backtrack(0, "")
    return result

# Test
print(letterCombinations("23"))
# Output: ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
```

**Key Points**:
- Use index to track current digit position
- Build string incrementally
- No explicit backtracking needed since strings are immutable

---

## 5. Binary Tree Paths

**Problem**: Given the root of a binary tree, return all root-to-leaf paths.

**Example**:
```
Input: root = [1,2,3,null,5]
Output: ["1->2->5","1->3"]
```

**Solution**:
```python
def binaryTreePaths(root):
    """
    Find all root-to-leaf paths using backtracking.
    
    Time Complexity: O(n * h) where n is nodes, h is height
    Space Complexity: O(h) - recursion depth
    """
    if not root:
        return []
    
    result = []
    
    def backtrack(node, path):
        # Add current node to path
        path.append(str(node.val))
        
        # If leaf node, add path to result
        if not node.left and not node.right:
            result.append("->".join(path))
        else:
            # Recurse on children
            if node.left:
                backtrack(node.left, path)
            if node.right:
                backtrack(node.right, path)
        
        # Backtrack
        path.pop()
    
    backtrack(root, [])
    return result

# Alternative without explicit backtracking
def binaryTreePaths_v2(root):
    if not root:
        return []
    
    def dfs(node, path):
        if not node.left and not node.right:
            return [path + str(node.val)]
        
        paths = []
        if node.left:
            paths.extend(dfs(node.left, path + str(node.val) + "->"))
        if node.right:
            paths.extend(dfs(node.right, path + str(node.val) + "->"))
        return paths
    
    return dfs(root, "")
```

**Key Points**:
- Check for leaf nodes (no children)
- Build path as string with arrow separator
- Backtrack by removing last added node

---

## 6. Combination Sum

**Problem**: Given an array of distinct integers `candidates` and a target integer `target`, return all unique combinations where the candidates sum to `target`. The same number may be chosen multiple times.

**Example**:
```
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
```

**Solution**:
```python
def combinationSum(candidates, target):
    """
    Find all combinations that sum to target.
    
    Time Complexity: O(2^target) - in worst case
    Space Complexity: O(target) - recursion depth
    """
    result = []
    
    def backtrack(start, path, remaining):
        # Base case: found valid combination
        if remaining == 0:
            result.append(path[:])
            return
        
        # Pruning: if remaining is negative, skip
        if remaining < 0:
            return
        
        # Try each candidate from start index
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            # Can reuse same number, so pass i (not i+1)
            backtrack(i, path, remaining - candidates[i])
            path.pop()
    
    backtrack(0, [], target)
    return result

# Test
print(combinationSum([2, 3, 6, 7], 7))
# Output: [[2, 2, 3], [7]]
```

**Key Points**:
- Allow reusing same number by passing `i` instead of `i+1`
- Prune early when remaining becomes negative
- Use `start` to avoid duplicate combinations

---

## Practice Tips

### 1. Template Recognition
- Identify if problem requires generating all possibilities
- Look for constraints that can be used for pruning
- Determine base case for recursion termination

### 2. Common Patterns
```python
# Generation pattern
def generate_all(choices):
    result = []
    
    def backtrack(path, remaining_choices):
        if is_complete(path):
            result.append(path[:])
            return
        
        for choice in remaining_choices:
            path.append(choice)
            backtrack(path, updated_choices)
            path.pop()
    
    backtrack([], choices)
    return result
```

### 3. Optimization Techniques
- **Early Pruning**: Check constraints before recursing
- **Sorting**: Order choices to enable better pruning
- **Memoization**: Cache results for overlapping subproblems

### 4. Debugging Strategies
- Print path at each recursive call
- Verify base cases handle edge cases
- Check that backtracking properly undoes changes
- Test with small inputs first

### 5. Time Complexity Analysis
- Count total number of recursive calls
- Consider cost of operations in each call
- Factor in pruning effectiveness

These easy problems establish the foundation for more complex backtracking scenarios. Master these patterns before moving to medium-level problems!
