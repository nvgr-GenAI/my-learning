# Backtracking - Hard Problems

## Problem Categories

### 1. Advanced Constraint Satisfaction
- Regex matching
- Wildcard matching
- Expression evaluation

### 2. Complex Combinatorial Problems
- Partition to k equal sum subsets
- 24 Game
- Cryptarithmetic puzzles

### 3. Advanced Search Problems
- Remove invalid parentheses
- Word ladder with transformations
- Minimum genetic mutation

---

## 1. Regular Expression Matching

**Problem**: Given an input string `s` and a pattern `p`, implement regular expression matching with support for `'.'` and `'*'`.

**Example**:
```
Input: s = "aa", p = "a*"
Output: true
```

**Solution**:
```python
def isMatch(s, p):
    """
    Regular expression matching with backtracking.
    
    Time Complexity: O(2^(m+n)) in worst case with many '*'
    Space Complexity: O(m+n) - recursion depth
    """
    def backtrack(i, j):
        # Base case: reached end of pattern
        if j == len(p):
            return i == len(s)
        
        # Check if current characters match
        first_match = i < len(s) and (p[j] == s[i] or p[j] == '.')
        
        # Handle '*' wildcard
        if j + 1 < len(p) and p[j + 1] == '*':
            # Two choices: skip pattern or match one character
            return (backtrack(i, j + 2) or  # Skip pattern (0 matches)
                   (first_match and backtrack(i + 1, j)))  # Match and continue
        else:
            # Regular character match
            return first_match and backtrack(i + 1, j + 1)
    
    return backtrack(0, 0)

# Memoized version for optimization
def isMatch_memoized(s, p):
    memo = {}
    
    def dp(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        
        if j == len(p):
            result = i == len(s)
        else:
            first_match = i < len(s) and (p[j] == s[i] or p[j] == '.')
            
            if j + 1 < len(p) and p[j + 1] == '*':
                result = (dp(i, j + 2) or 
                         (first_match and dp(i + 1, j)))
            else:
                result = first_match and dp(i + 1, j + 1)
        
        memo[(i, j)] = result
        return result
    
    return dp(0, 0)

# Test cases
print(isMatch("aa", "a"))      # False
print(isMatch("aa", "a*"))     # True
print(isMatch("ab", ".*"))     # True
print(isMatch("aab", "c*a*b")) # True
```

**Key Points**:
- Handle `'*'` by trying both skip and match options
- Use memoization to avoid recomputing subproblems
- Base case: when pattern is exhausted

---

## 2. Remove Invalid Parentheses

**Problem**: Given a string `s` that contains parentheses and letters, remove the minimum number of invalid parentheses to make the input string valid.

**Example**:
```
Input: s = "()())"
Output: ["()()", "(())"]
```

**Solution**:
```python
def removeInvalidParentheses(s):
    """
    Remove minimum invalid parentheses using backtracking.
    
    Time Complexity: O(2^n) - each character can be kept or removed
    Space Complexity: O(n) - recursion depth
    """
    def is_valid(string):
        count = 0
        for char in string:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
                if count < 0:
                    return False
        return count == 0
    
    def calculate_invalid_counts(s):
        left = right = 0
        for char in s:
            if char == '(':
                left += 1
            elif char == ')':
                if left > 0:
                    left -= 1
                else:
                    right += 1
        return left, right
    
    result = set()
    left_rem, right_rem = calculate_invalid_counts(s)
    
    def backtrack(index, path, left_rem, right_rem, open_count):
        # Base case: processed all characters
        if index == len(s):
            if left_rem == 0 and right_rem == 0 and open_count == 0:
                result.add(path)
            return
        
        char = s[index]
        
        # Option 1: Remove current character (if it's a parenthesis to remove)
        if char == '(' and left_rem > 0:
            backtrack(index + 1, path, left_rem - 1, right_rem, open_count)
        if char == ')' and right_rem > 0:
            backtrack(index + 1, path, left_rem, right_rem - 1, open_count)
        
        # Option 2: Keep current character
        if char == '(':
            backtrack(index + 1, path + char, left_rem, right_rem, open_count + 1)
        elif char == ')':
            if open_count > 0:  # Only if we have unmatched '('
                backtrack(index + 1, path + char, left_rem, right_rem, open_count - 1)
        else:  # Regular character
            backtrack(index + 1, path + char, left_rem, right_rem, open_count)
    
    backtrack(0, "", left_rem, right_rem, 0)
    return list(result)

# Optimized BFS approach
def removeInvalidParentheses_bfs(s):
    from collections import deque
    
    def is_valid(s):
        count = 0
        for char in s:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
                if count < 0:
                    return False
        return count == 0
    
    if is_valid(s):
        return [s]
    
    queue = deque([s])
    visited = {s}
    result = []
    found = False
    
    while queue and not found:
        size = len(queue)
        
        for _ in range(size):
            current = queue.popleft()
            
            for i in range(len(current)):
                if current[i] in '()':
                    next_string = current[:i] + current[i+1:]
                    
                    if next_string not in visited:
                        visited.add(next_string)
                        
                        if is_valid(next_string):
                            result.append(next_string)
                            found = True
                        else:
                            queue.append(next_string)
    
    return result
```

**Key Points**:
- Calculate minimum removals needed first
- Use backtracking to try all removal combinations
- BFS alternative finds minimum removals level by level

---

## 3. Expression Add Operators

**Problem**: Given a string `num` and an integer `target`, insert binary operators `+`, `-`, `*` between digits to make the expression equal to `target`.

**Example**:
```
Input: num = "123", target = 6
Output: ["1*2*3","1+2+3"]
```

**Solution**:
```python
def addOperators(num, target):
    """
    Find all expressions that evaluate to target using backtracking.
    
    Time Complexity: O(4^n) - 4 choices for each position
    Space Complexity: O(n) - recursion depth and path length
    """
    result = []
    
    def backtrack(index, path, value, prev):
        # Base case: processed all digits
        if index == len(num):
            if value == target:
                result.append(path)
            return
        
        # Try all possible number formations from current position
        for i in range(index, len(num)):
            num_str = num[index:i+1]
            
            # Skip numbers with leading zeros (except single "0")
            if len(num_str) > 1 and num_str[0] == '0':
                break
            
            num_val = int(num_str)
            
            if index == 0:
                # First number, no operator needed
                backtrack(i + 1, num_str, num_val, num_val)
            else:
                # Try addition
                backtrack(i + 1, path + '+' + num_str, value + num_val, num_val)
                
                # Try subtraction
                backtrack(i + 1, path + '-' + num_str, value - num_val, -num_val)
                
                # Try multiplication (tricky due to operator precedence)
                backtrack(i + 1, path + '*' + num_str, 
                         value - prev + prev * num_val, prev * num_val)
    
    backtrack(0, "", 0, 0)
    return result

# Example with detailed explanation of multiplication handling
def addOperators_detailed(num, target):
    """
    The key insight for multiplication:
    If we have "1+2*3", we need to handle precedence.
    
    When we see '*':
    1. Remove the previous addition: value - prev
    2. Add the multiplication: + prev * current
    
    Example: "1+2*3"
    - After "1+2": value=3, prev=2
    - When seeing "*3": value = 3-2+2*3 = 1+6 = 7
    """
    result = []
    
    def backtrack(index, path, value, prev):
        if index == len(num):
            if value == target:
                result.append(path)
            return
        
        for i in range(index, len(num)):
            num_str = num[index:i+1]
            
            # Handle leading zeros
            if len(num_str) > 1 and num_str[0] == '0':
                break
            
            num_val = int(num_str)
            
            if index == 0:
                backtrack(i + 1, num_str, num_val, num_val)
            else:
                # Addition: value = value + num_val, prev = num_val
                backtrack(i + 1, path + '+' + num_str, value + num_val, num_val)
                
                # Subtraction: value = value - num_val, prev = -num_val
                backtrack(i + 1, path + '-' + num_str, value - num_val, -num_val)
                
                # Multiplication: handle precedence
                # New value = value - prev + (prev * num_val)
                # New prev = prev * num_val
                backtrack(i + 1, path + '*' + num_str, 
                         value - prev + prev * num_val, prev * num_val)
    
    backtrack(0, "", 0, 0)
    return result

# Test
print(addOperators("123", 6))
# Output: ['1*2*3', '1+2+3']
print(addOperators("232", 8))
# Output: ['2*3+2', '2+3*2']
```

**Key Points**:
- Handle operator precedence correctly for multiplication
- Track previous operand to handle multiplication precedence
- Skip numbers with leading zeros

---

## 4. Partition to K Equal Sum Subsets

**Problem**: Given an integer array `nums` and an integer `k`, return `true` if it's possible to divide this array into `k` non-empty subsets whose sums are all equal.

**Example**:
```
Input: nums = [4,3,2,3,5,2,1], k = 4
Output: true
```

**Solution**:
```python
def canPartitionKSubsets(nums, k):
    """
    Partition array into k equal sum subsets using backtracking.
    
    Time Complexity: O(k^n) - each element can go to any of k subsets
    Space Complexity: O(n + k) - recursion depth and subset tracking
    """
    total = sum(nums)
    if total % k != 0:
        return False
    
    target = total // k
    nums.sort(reverse=True)  # Sort descending for better pruning
    
    # Early termination: if any number is larger than target
    if nums[0] > target:
        return False
    
    def backtrack(index, subsets):
        # Base case: all numbers assigned
        if index == len(nums):
            return True
        
        num = nums[index]
        
        # Try putting current number in each subset
        for i in range(k):
            # Pruning: skip if adding would exceed target
            if subsets[i] + num <= target:
                subsets[i] += num
                
                if backtrack(index + 1, subsets):
                    return True
                
                # Backtrack
                subsets[i] -= num
            
            # Pruning: if current subset is empty, no point trying other empty subsets
            if subsets[i] == 0:
                break
        
        return False
    
    return backtrack(0, [0] * k)

# Alternative approach: fill one subset at a time
def canPartitionKSubsets_v2(nums, k):
    total = sum(nums)
    if total % k != 0:
        return False
    
    target = total // k
    nums.sort(reverse=True)
    used = [False] * len(nums)
    
    def fill_subset(start, current_sum, filled_subsets):
        # Base case: all subsets filled
        if filled_subsets == k:
            return True
        
        # Current subset complete, start next one
        if current_sum == target:
            return fill_subset(0, 0, filled_subsets + 1)
        
        # Try adding each unused number to current subset
        for i in range(start, len(nums)):
            if used[i] or current_sum + nums[i] > target:
                continue
            
            used[i] = True
            if fill_subset(i + 1, current_sum + nums[i], filled_subsets):
                return True
            used[i] = False
            
            # Pruning: if we can't use this number, skip all equal numbers
            while i + 1 < len(nums) and nums[i] == nums[i + 1]:
                i += 1
        
        return False
    
    return fill_subset(0, 0, 0)

# Optimized with bitmask memoization
def canPartitionKSubsets_memo(nums, k):
    total = sum(nums)
    if total % k != 0:
        return False
    
    target = total // k
    memo = {}
    
    def dp(mask, current_sum):
        if mask in memo:
            return memo[mask]
        
        # All numbers used
        if mask == (1 << len(nums)) - 1:
            return True
        
        # Current subset complete
        if current_sum == target:
            result = dp(mask, 0)
        else:
            result = False
            for i in range(len(nums)):
                # If number not used and fits in current subset
                if not (mask & (1 << i)) and current_sum + nums[i] <= target:
                    if dp(mask | (1 << i), current_sum + nums[i]):
                        result = True
                        break
        
        memo[mask] = result
        return result
    
    return dp(0, 0)
```

**Key Points**:
- Sort numbers in descending order for better pruning
- Skip trying empty subsets after first empty one
- Alternative: fill one subset completely before starting next
- Memoization with bitmask for exponential optimization

---

## 5. Wildcard Matching

**Problem**: Given an input string `s` and a pattern `p`, implement wildcard pattern matching with support for `'?'` and `'*'`.

**Example**:
```
Input: s = "adceb", p = "*a*b*"
Output: true
```

**Solution**:
```python
def isMatch(s, p):
    """
    Wildcard matching using backtracking with memoization.
    
    Time Complexity: O(m*n) with memoization
    Space Complexity: O(m*n) - memoization table
    """
    memo = {}
    
    def backtrack(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        
        # Base cases
        if j == len(p):
            result = i == len(s)
        elif i == len(s):
            # Pattern must be all '*' to match empty string
            result = all(c == '*' for c in p[j:])
        elif p[j] == '*':
            # '*' can match empty string or any sequence
            result = (backtrack(i, j + 1) or      # Match empty
                     backtrack(i + 1, j))        # Match one char and continue
        elif p[j] == '?' or p[j] == s[i]:
            # '?' matches any single char, or exact match
            result = backtrack(i + 1, j + 1)
        else:
            # No match
            result = False
        
        memo[(i, j)] = result
        return result
    
    return backtrack(0, 0)

# Iterative DP solution
def isMatch_dp(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Base case: empty pattern matches empty string
    dp[0][0] = True
    
    # Handle patterns starting with '*'
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]

# Space-optimized version
def isMatch_optimized(s, p):
    m, n = len(s), len(p)
    
    # Use two rows instead of full DP table
    prev = [False] * (n + 1)
    curr = [False] * (n + 1)
    
    prev[0] = True
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            prev[j] = prev[j - 1]
    
    for i in range(1, m + 1):
        curr[0] = False
        
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                curr[j] = prev[j] or curr[j - 1]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = False
        
        prev, curr = curr, prev
    
    return prev[n]
```

**Key Points**:
- `'*'` can match empty string or any sequence of characters
- `'?'` matches exactly one character
- Use memoization to avoid recomputing subproblems
- DP solution has better space complexity

---

## Advanced Backtracking Techniques

### 1. Constraint Propagation
```python
def backtrack_with_constraint_propagation(state):
    # Propagate constraints after each choice
    def propagate_constraints(state, choice):
        # Update domain of related variables
        affected_vars = get_affected_variables(choice)
        for var in affected_vars:
            remove_invalid_values(var, choice)
    
    if is_complete(state):
        return True
    
    var = select_variable(state)
    for value in get_domain(var):
        if is_consistent(var, value, state):
            make_assignment(var, value, state)
            propagate_constraints(state, (var, value))
            
            if backtrack_with_constraint_propagation(state):
                return True
            
            undo_assignment(var, value, state)
    
    return False
```

### 2. Forward Checking
```python
def backtrack_with_forward_checking(state, domains):
    if is_complete(state):
        return True
    
    var = select_unassigned_variable(state)
    
    for value in domains[var]:
        if is_consistent(var, value, state):
            # Save current domains
            saved_domains = {v: d.copy() for v, d in domains.items()}
            
            make_assignment(var, value, state)
            
            # Forward check: remove inconsistent values from future variables
            if forward_check(var, value, domains, state):
                if backtrack_with_forward_checking(state, domains):
                    return True
            
            # Restore domains and assignment
            domains = saved_domains
            undo_assignment(var, value, state)
    
    return False
```

### 3. Arc Consistency
```python
def maintain_arc_consistency(domains, constraints):
    queue = list(constraints)
    
    while queue:
        (xi, xj) = queue.pop(0)
        
        if revise(domains, xi, xj):
            if not domains[xi]:  # Domain became empty
                return False
            
            # Add all arcs (xk, xi) for xk != xi, xj
            for xk in get_neighbors(xi):
                if xk != xj:
                    queue.append((xk, xi))
    
    return True

def revise(domains, xi, xj):
    revised = False
    to_remove = []
    
    for x in domains[xi]:
        if not any(is_consistent(x, y, xi, xj) for y in domains[xj]):
            to_remove.append(x)
            revised = True
    
    for x in to_remove:
        domains[xi].remove(x)
    
    return revised
```

These hard problems demonstrate advanced backtracking techniques including sophisticated pruning strategies, constraint propagation, and optimization methods essential for solving complex combinatorial problems efficiently.
