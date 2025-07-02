# Permutations & Combinations 🔢

## 🎯 Overview

Combinatorics is a branch of mathematics dealing with counting, arrangement, and selection of objects. It's essential for solving algorithmic problems involving counting possibilities, probability, and optimization.

## 📋 Core Concepts

### Permutations

- **Definition**: Arrangement of objects where order matters
- **Formula**: P(n,r) = n! / (n-r)!
- **All permutations**: n!

### Combinations

- **Definition**: Selection of objects where order doesn't matter
- **Formula**: C(n,r) = n! / (r! × (n-r)!)
- **Also written as**: nCr or (n choose r)

### Binomial Coefficients

- **Pascal's Triangle**: C(n,r) = C(n-1,r-1) + C(n-1,r)
- **Properties**: C(n,0) = C(n,n) = 1
- **Symmetry**: C(n,r) = C(n,n-r)

## 🔧 Implementations

### Basic Factorial and Combinations

```python
def factorial(n):
    """Calculate n! iteratively"""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def factorial_recursive(n):
    """Calculate n! recursively"""
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

def permutations(n, r):
    """Calculate P(n,r) = n! / (n-r)!"""
    if r > n or r < 0:
        return 0
    if r == 0:
        return 1
    
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    return result

def combinations(n, r):
    """Calculate C(n,r) = n! / (r! * (n-r)!)"""
    if r > n or r < 0:
        return 0
    if r == 0 or r == n:
        return 1
    
    # Optimize by choosing smaller r
    r = min(r, n - r)
    
    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)
    return result

# Example usage
print(f"P(5,3) = {permutations(5, 3)}")  # 60
print(f"C(5,3) = {combinations(5, 3)}")  # 10
```

### Pascal's Triangle

```python
def pascal_triangle(n):
    """Generate Pascal's triangle up to row n"""
    triangle = []
    
    for i in range(n + 1):
        row = [1] * (i + 1)
        for j in range(1, i):
            row[j] = triangle[i-1][j-1] + triangle[i-1][j]
        triangle.append(row)
    
    return triangle

def pascal_row(n):
    """Generate nth row of Pascal's triangle efficiently"""
    row = [1]
    for i in range(1, n + 1):
        row.append(row[i-1] * (n - i + 1) // i)
    return row

# Example
triangle = pascal_triangle(5)
for i, row in enumerate(triangle):
    print(f"Row {i}: {row}")
```

### Modular Arithmetic for Large Numbers

```python
MOD = 10**9 + 7

def mod_inverse(a, mod=MOD):
    """Calculate modular inverse using Fermat's little theorem"""
    return pow(a, mod - 2, mod)

def factorial_mod(n, mod=MOD):
    """Calculate n! mod p"""
    result = 1
    for i in range(1, n + 1):
        result = (result * i) % mod
    return result

def combinations_mod(n, r, mod=MOD):
    """Calculate C(n,r) mod p"""
    if r > n or r < 0:
        return 0
    if r == 0 or r == n:
        return 1
    
    # Pre-calculate factorials
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = (fact[i-1] * i) % mod
    
    # C(n,r) = n! / (r! * (n-r)!)
    numerator = fact[n]
    denominator = (fact[r] * fact[n-r]) % mod
    
    return (numerator * mod_inverse(denominator, mod)) % mod

# Precomputed factorials for efficiency
class CombinatoricsPrecomputed:
    def __init__(self, n, mod=MOD):
        self.mod = mod
        self.fact = [1] * (n + 1)
        self.inv_fact = [1] * (n + 1)
        
        # Calculate factorials
        for i in range(1, n + 1):
            self.fact[i] = (self.fact[i-1] * i) % mod
        
        # Calculate inverse factorials
        self.inv_fact[n] = mod_inverse(self.fact[n], mod)
        for i in range(n - 1, -1, -1):
            self.inv_fact[i] = (self.inv_fact[i + 1] * (i + 1)) % mod
    
    def C(self, n, r):
        """Calculate C(n,r) in O(1)"""
        if r > n or r < 0:
            return 0
        return (self.fact[n] * self.inv_fact[r] % self.mod 
                * self.inv_fact[n-r]) % self.mod
    
    def P(self, n, r):
        """Calculate P(n,r) in O(1)"""
        if r > n or r < 0:
            return 0
        return (self.fact[n] * self.inv_fact[n-r]) % self.mod
```

### Generating Permutations and Combinations

```python
def generate_permutations(arr):
    """Generate all permutations of array"""
    if len(arr) <= 1:
        return [arr]
    
    result = []
    for i in range(len(arr)):
        element = arr[i]
        remaining = arr[:i] + arr[i+1:]
        
        for perm in generate_permutations(remaining):
            result.append([element] + perm)
    
    return result

def generate_combinations(arr, r):
    """Generate all combinations of r elements"""
    if r == 0:
        return [[]]
    if len(arr) < r:
        return []
    
    result = []
    
    # Include first element
    first = arr[0]
    remaining = arr[1:]
    
    for combo in generate_combinations(remaining, r - 1):
        result.append([first] + combo)
    
    # Exclude first element
    result.extend(generate_combinations(remaining, r))
    
    return result

# Using itertools (more efficient)
from itertools import permutations, combinations, combinations_with_replacement

def efficient_permutations(arr, r=None):
    """Generate permutations using itertools"""
    if r is None:
        r = len(arr)
    return list(permutations(arr, r))

def efficient_combinations(arr, r):
    """Generate combinations using itertools"""
    return list(combinations(arr, r))

# Example usage
arr = [1, 2, 3, 4]
print("Permutations P(4,2):", efficient_permutations(arr, 2))
print("Combinations C(4,2):", efficient_combinations(arr, 2))
```

## 🚀 Advanced Applications

### Derangements

```python
def derangements(n):
    """Count derangements (permutations with no fixed points)"""
    if n == 0:
        return 1
    if n == 1:
        return 0
    if n == 2:
        return 1
    
    # D(n) = (n-1) * (D(n-1) + D(n-2))
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 0
    dp[2] = 1
    
    for i in range(3, n + 1):
        dp[i] = (i - 1) * (dp[i-1] + dp[i-2])
    
    return dp[n]
```

### Catalan Numbers

```python
def catalan_number(n):
    """Calculate nth Catalan number"""
    if n <= 1:
        return 1
    
    # C(n) = C(2n,n) / (n+1)
    return combinations(2 * n, n) // (n + 1)

def catalan_dp(n):
    """Calculate nth Catalan number using DP"""
    if n <= 1:
        return 1
    
    catalan = [0] * (n + 1)
    catalan[0] = catalan[1] = 1
    
    for i in range(2, n + 1):
        for j in range(i):
            catalan[i] += catalan[j] * catalan[i-1-j]
    
    return catalan[n]
```

### Stirling Numbers

```python
def stirling_second(n, k):
    """Stirling numbers of the second kind S(n,k)"""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0
    
    # S(n,k) = k*S(n-1,k) + S(n-1,k-1)
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    
    for i in range(1, n + 1):
        for j in range(1, min(i + 1, k + 1)):
            dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
    
    return dp[n][k]
```

## 💡 Common Problem Patterns

### Pattern 1: Counting Arrangements

```python
def unique_permutations_with_duplicates(s):
    """Count unique permutations of string with duplicates"""
    from collections import Counter
    
    counter = Counter(s)
    n = len(s)
    
    # Total arrangements = n! / (count1! * count2! * ...)
    result = factorial(n)
    for count in counter.values():
        result //= factorial(count)
    
    return result

# Example: "AABBC" has 5!/(2!*2!*1!) = 30 unique permutations
```

### Pattern 2: Selection with Constraints

```python
def ways_to_distribute_items(items, groups, min_per_group=0):
    """Distribute items into groups with minimum constraint"""
    # This is a stars and bars problem with constraints
    if items < groups * min_per_group:
        return 0
    
    # After giving minimum to each group
    remaining = items - groups * min_per_group
    
    # Ways to distribute remaining items
    return combinations(remaining + groups - 1, groups - 1)
```

### Pattern 3: Inclusion-Exclusion Principle

```python
def count_with_inclusion_exclusion(sets):
    """Count elements in union using inclusion-exclusion"""
    from itertools import combinations
    
    total = 0
    n = len(sets)
    
    for r in range(1, n + 1):
        sign = (-1) ** (r - 1)
        for combo in combinations(range(n), r):
            intersection_size = len(set.intersection(*[sets[i] for i in combo]))
            total += sign * intersection_size
    
    return total
```

## 🎲 Interview Problems

### Problem 1: Unique Paths

```python
def unique_paths(m, n):
    """Number of unique paths in m×n grid"""
    # This is C(m+n-2, m-1) or C(m+n-2, n-1)
    return combinations(m + n - 2, m - 1)

def unique_paths_with_obstacles(grid):
    """Unique paths with obstacles using DP"""
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    
    # Initialize first cell
    dp[0][0] = 1 if grid[0][0] == 0 else 0
    
    # Fill first row and column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] if grid[i][0] == 0 else 0
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] if grid[0][j] == 0 else 0
    
    # Fill rest of the grid
    for i in range(1, m):
        for j in range(1, n):
            if grid[i][j] == 0:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]
```

### Problem 2: Letter Combinations of Phone Number

```python
def letter_combinations(digits):
    """Generate all letter combinations from phone number"""
    if not digits:
        return []
    
    phone = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    def backtrack(index, path):
        if index == len(digits):
            combinations.append(path)
            return
        
        for letter in phone[digits[index]]:
            backtrack(index + 1, path + letter)
    
    combinations = []
    backtrack(0, "")
    return combinations

# Total combinations = product of lengths of each digit's letters
def count_letter_combinations(digits):
    phone = {
        '2': 3, '3': 3, '4': 3, '5': 3,
        '6': 3, '7': 4, '8': 3, '9': 4
    }
    
    result = 1
    for digit in digits:
        result *= phone.get(digit, 0)
    return result
```

### Problem 3: Subsets and Power Set

```python
def generate_subsets(nums):
    """Generate all subsets (power set)"""
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

def count_subsets():
    """Count of subsets for n elements"""
    return lambda n: 2**n

# Subsets with specific sum
def subset_sum_count(nums, target):
    """Count subsets with given sum"""
    def dp_approach():
        dp = [0] * (target + 1)
        dp[0] = 1
        
        for num in nums:
            for j in range(target, num - 1, -1):
                dp[j] += dp[j - num]
        
        return dp[target]
    
    return dp_approach()
```

## 🔍 Time Complexities

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| C(n,r) calculation | O(r) | O(1) |
| Pascal's triangle | O(n²) | O(n²) |
| All permutations | O(n! × n) | O(n! × n) |
| All combinations | O(C(n,r)) | O(C(n,r)) |
| Factorial | O(n) | O(1) |
| Precomputed C(n,r) | O(n) setup, O(1) query | O(n) |

## 🎯 Key Takeaways

1. **Choose the right formula**: Permutations for order, combinations for selection
2. **Use modular arithmetic**: For large numbers to prevent overflow
3. **Precompute when possible**: For multiple queries
4. **Optimize combinations**: Use symmetry C(n,r) = C(n,n-r)
5. **Pascal's triangle**: Efficient for multiple binomial coefficients
6. **Dynamic programming**: For complex counting problems
7. **Inclusion-exclusion**: For counting with constraints

## 📚 Practice Problems

1. **Easy**: Climbing Stairs, Unique Paths, Letter Case Permutation
2. **Medium**: Permutations II, Combinations, Generate Parentheses
3. **Hard**: N-Queens, Sudoku Solver, Word Pattern II

The key to mastering combinatorics is understanding when to use permutations vs combinations and how to handle large numbers efficiently with modular arithmetic.
