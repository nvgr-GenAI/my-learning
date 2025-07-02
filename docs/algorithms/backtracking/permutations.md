# Permutations & Combinations

This section covers backtracking algorithms for generating permutations and combinations.

## Core Concepts

### Permutations
- Order matters
- All possible arrangements of elements
- P(n,r) = n!/(n-r)!

### Combinations  
- Order doesn't matter
- Selecting elements from a set
- C(n,r) = n!/(r!(n-r)!)

---

## Problems

### 1. Generate All Permutations

**Problem**: Generate all possible permutations of an array.

```python
def permutations(nums):
    """Generate all permutations of nums array."""
    result = []
    
    def backtrack(current_perm):
        if len(current_perm) == len(nums):
            result.append(current_perm[:])
            return
        
        for i in range(len(nums)):
            if nums[i] not in current_perm:
                current_perm.append(nums[i])
                backtrack(current_perm)
                current_perm.pop()
    
    backtrack([])
    return result
```

### 2. Permutations with Duplicates

**Problem**: Generate permutations when array contains duplicates.

```python
def permutations_unique(nums):
    """Generate unique permutations with duplicates."""
    result = []
    nums.sort()  # Sort to handle duplicates
    used = [False] * len(nums)
    
    def backtrack(current_perm):
        if len(current_perm) == len(nums):
            result.append(current_perm[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            # Skip duplicates: if current element equals previous
            # and previous is not used, skip current
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue
                
            used[i] = True
            current_perm.append(nums[i])
            backtrack(current_perm)
            current_perm.pop()
            used[i] = False
    
    backtrack([])
    return result
```

### 3. Combinations

**Problem**: Generate all combinations of k elements from n elements.

```python
def combinations(n, k):
    """Generate all combinations of k elements from 1 to n."""
    result = []
    
    def backtrack(start, current_comb):
        if len(current_comb) == k:
            result.append(current_comb[:])
            return
        
        for i in range(start, n + 1):
            current_comb.append(i)
            backtrack(i + 1, current_comb)
            current_comb.pop()
    
    backtrack(1, [])
    return result
```

### 4. Next Permutation

**Problem**: Find the next lexicographically greater permutation.

```python
def next_permutation(nums):
    """Find next lexicographically greater permutation in-place."""
    # Find the largest index i such that nums[i] < nums[i+1]
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i == -1:
        # nums is the last permutation
        nums.reverse()
        return
    
    # Find the largest index j such that nums[i] < nums[j]
    j = len(nums) - 1
    while nums[j] <= nums[i]:
        j -= 1
    
    # Swap nums[i] and nums[j]
    nums[i], nums[j] = nums[j], nums[i]
    
    # Reverse the suffix starting at nums[i+1]
    nums[i + 1:] = reversed(nums[i + 1:])
```

## Advanced Patterns

### Lexicographic Ordering
- Generate permutations in lexicographic order
- Use for optimization and early termination

### Pruning Techniques
- Skip equivalent branches
- Use sorting and duplicate detection

### Memory Optimization
- Generate permutations iteratively
- Use generators for large datasets

## Time Complexities

- **Permutations**: O(n! × n)
- **Combinations**: O(C(n,k) × k)
- **Next Permutation**: O(n)

## Applications

- Cryptography and security
- Scheduling and resource allocation
- Game theory and strategy
- Data analysis and statistics
