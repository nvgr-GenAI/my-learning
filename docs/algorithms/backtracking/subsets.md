# Subset Generation

This section covers backtracking algorithms for generating subsets and partitions.

## Core Concepts

### Subsets (Power Set)

- All possible combinations of elements from a set
- For a set of n elements, there are 2^n subsets
- Includes empty set and the set itself

### Partitions

- Ways to divide a set into non-empty subsets
- Each element appears in exactly one subset
- Order of subsets doesn't matter

---

## Problems

### 1. Generate All Subsets

**Problem**: Generate all possible subsets of an array.

```python
def subsets(nums):
    """Generate all subsets of nums array."""
    result = []
    
    def backtrack(start, current_subset):
        # Add current subset to result
        result.append(current_subset[:])
        
        # Try adding each remaining element
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()
    
    backtrack(0, [])
    return result

# Alternative iterative approach
def subsets_iterative(nums):
    """Generate subsets using bit manipulation."""
    result = []
    n = len(nums)
    
    # Iterate through all possible combinations (2^n)
    for i in range(1 << n):
        subset = []
        for j in range(n):
            # Check if j-th bit is set
            if i & (1 << j):
                subset.append(nums[j])
        result.append(subset)
    
    return result
```

### 2. Subsets with Duplicates

**Problem**: Generate unique subsets when array contains duplicates.

```python
def subsets_with_dup(nums):
    """Generate unique subsets with duplicates."""
    result = []
    nums.sort()  # Sort to handle duplicates
    
    def backtrack(start, current_subset):
        result.append(current_subset[:])
        
        for i in range(start, len(nums)):
            # Skip duplicates: if current element equals previous
            # and we're not at the start position, skip
            if i > start and nums[i] == nums[i-1]:
                continue
                
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()
    
    backtrack(0, [])
    return result
```

### 3. Subset Sum

**Problem**: Find all subsets that sum to a target value.

```python
def subset_sum(nums, target):
    """Find all subsets that sum to target."""
    result = []
    
    def backtrack(start, current_subset, current_sum):
        if current_sum == target:
            result.append(current_subset[:])
            return
        
        if current_sum > target:
            return  # Prune: no point continuing
        
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset, current_sum + nums[i])
            current_subset.pop()
    
    backtrack(0, [], 0)
    return result
```

### 4. Partition Equal Subset Sum

**Problem**: Check if array can be partitioned into two equal sum subsets.

```python
def can_partition(nums):
    """Check if array can be partitioned into equal sum subsets."""
    total_sum = sum(nums)
    
    # If total sum is odd, cannot partition equally
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    
    def backtrack(start, current_sum):
        if current_sum == target:
            return True
        if current_sum > target:
            return False
        
        for i in range(start, len(nums)):
            if backtrack(i + 1, current_sum + nums[i]):
                return True
        
        return False
    
    return backtrack(0, 0)

# Optimized DP approach
def can_partition_dp(nums):
    """Dynamic programming solution."""
    total_sum = sum(nums)
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    return dp[target]
```

### 5. Palindromic Partitioning

**Problem**: Partition string so that every substring is a palindrome.

```python
def partition_palindromes(s):
    """Find all palindromic partitions of string."""
    result = []
    
    def is_palindrome(string):
        return string == string[::-1]
    
    def backtrack(start, current_partition):
        if start == len(s):
            result.append(current_partition[:])
            return
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                current_partition.append(substring)
                backtrack(end, current_partition)
                current_partition.pop()
    
    backtrack(0, [])
    return result

# Optimized with memoization
def partition_palindromes_memo(s):
    """Optimized with palindrome memoization."""
    n = len(s)
    
    # Precompute palindrome check
    is_palindrome = [[False] * n for _ in range(n)]
    
    # Single characters are palindromes
    for i in range(n):
        is_palindrome[i][i] = True
    
    # Check for palindromes of length 2
    for i in range(n - 1):
        is_palindrome[i][i + 1] = (s[i] == s[i + 1])
    
    # Check for longer palindromes
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_palindrome[i][j] = (s[i] == s[j] and is_palindrome[i + 1][j - 1])
    
    result = []
    
    def backtrack(start, current_partition):
        if start == len(s):
            result.append(current_partition[:])
            return
        
        for end in range(start, len(s)):
            if is_palindrome[start][end]:
                current_partition.append(s[start:end + 1])
                backtrack(end + 1, current_partition)
                current_partition.pop()
    
    backtrack(0, [])
    return result
```

## Advanced Techniques

### Bit Manipulation for Subsets

```python
def subsets_bitmask(nums):
    """Generate subsets using bitmask approach."""
    n = len(nums)
    result = []
    
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    
    return result
```

### Pruning Strategies

1. **Early Termination**: Stop when sum exceeds target
2. **Duplicate Skipping**: Sort array and skip duplicates
3. **Constraint Checking**: Validate constraints before recursing

### Memory Optimization

- Use iterative approaches for large datasets
- Implement generators for streaming results
- Use bit manipulation for space efficiency

## Time Complexities

- **All Subsets**: O(2^n × n)
- **Subset Sum**: O(2^n) worst case, better with pruning
- **Palindromic Partition**: O(2^n × n) with memoization

## Applications

- Combinatorial optimization
- Resource allocation problems
- Set partitioning in databases
- Feature selection in machine learning
- Cryptographic applications
