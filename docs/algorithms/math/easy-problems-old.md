# Mathematical Algorithms - Easy Problems

## 🎯 Learning Objectives

Master fundamental mathematical concepts in programming:

- Basic number theory operations
- Simple combinatorics and probability
- Elementary geometry calculations
- Digit manipulation and arithmetic
- Mathematical patterns and sequences

---

## Problem 1: Palindrome Number

**Difficulty**: 🟢 Easy  
**Pattern**: Digit Manipulation  
**Time**: O(log n), **Space**: O(1)

### Problem Description

Given an integer `x`, return `true` if `x` is palindrome integer.

An integer is a palindrome when it reads the same backward as forward.

**Examples:**
```
Input: x = 121
Output: true

Input: x = -121
Output: false
```

### Solution

```python
def isPalindrome(x):
    """
    Reverse half the number and compare with remaining half
    """
    if x < 0 or (x % 10 == 0 and x != 0):
        return False
    
    reversed_half = 0
    while x > reversed_half:
        reversed_half = reversed_half * 10 + x % 10
        x //= 10
    
    # For even digits: x == reversed_half
    # For odd digits: x == reversed_half // 10
    return x == reversed_half or x == reversed_half // 10

# Alternative: String-based approach
def isPalindromeString(x):
    """
    Convert to string and check if it's equal to its reverse
    """
    if x < 0:
        return False
    
    s = str(x)
    return s == s[::-1]

# Test
print(isPalindrome(121))   # True
print(isPalindrome(-121))  # False
print(isPalindrome(10))    # False
```

### 🔍 Key Insights

- **Negative numbers**: Always false (contains '-' which doesn't reverse)
- **Trailing zeros**: Only 0 itself is palindrome among numbers ending in 0
- **Half reversal**: More efficient than reversing entire number
- **Odd vs even digits**: Handle middle digit in odd-length numbers

---

## Problem 2: Reverse Integer

**Difficulty**: 🟢 Easy  
**Pattern**: Digit Manipulation  
**Time**: O(log n), **Space**: O(1)

### Problem Description

Given a signed 32-bit integer `x`, return `x` with its digits reversed. If reversing `x` causes the value to go outside the signed 32-bit integer range `[-2^31, 2^31 - 1]`, then return `0`.

**Examples:**
```
Input: x = 123
Output: 321

Input: x = -123
Output: -321

Input: x = 120
Output: 21
```

### Solution

```python
def reverse(x):
    """
    Reverse digits while checking for overflow
    """
    INT_MAX = 2**31 - 1  # 2147483647
    INT_MIN = -2**31     # -2147483648
    
    sign = 1 if x >= 0 else -1
    x = abs(x)
    
    result = 0
    while x:
        digit = x % 10
        
        # Check for overflow before updating result
        if result > (INT_MAX - digit) // 10:
            return 0
        
        result = result * 10 + digit
        x //= 10
    
    return sign * result

# Test
print(reverse(123))   # 321
print(reverse(-123))  # -321
print(reverse(120))   # 21
```

---

## Problem 3: Roman to Integer

**Difficulty**: 🟢 Easy  
**Pattern**: String Processing + Rules  
**Time**: O(n), **Space**: O(1)

### Problem Description

Convert Roman numerals to integers. Roman numerals use these symbols:

- I = 1, V = 5, X = 10, L = 50, C = 100, D = 500, M = 1000
- Subtraction rules: IV = 4, IX = 9, XL = 40, XC = 90, CD = 400, CM = 900

### Solution

```python
def romanToInt(s):
    """
    Process from right to left, subtract if smaller value before larger
    """
    roman_map = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    result = 0
    prev_value = 0
    
    for char in reversed(s):
        value = roman_map[char]
        
        if value < prev_value:
            result -= value  # Subtraction case (IV, IX, etc.)
        else:
            result += value  # Normal addition
        
        prev_value = value
    
    return result

# Alternative: Left-to-right approach
def romanToIntLTR(s):
    """
    Process from left to right, handle special cases
    """
    roman_map = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    result = 0
    i = 0
    
    while i < len(s):
        # Check for subtraction cases
        if i + 1 < len(s) and roman_map[s[i]] < roman_map[s[i + 1]]:
            result += roman_map[s[i + 1]] - roman_map[s[i]]
            i += 2
        else:
            result += roman_map[s[i]]
            i += 1
    
    return result

# Test
print(romanToInt("III"))     # 3
print(romanToInt("LVIII"))   # 58
print(romanToInt("MCMXC"))   # 1990
```

---

## Problem 4: Happy Number

**Difficulty**: 🟢 Easy  
**Pattern**: Cycle Detection  
**Time**: O(log n), **Space**: O(log n)

### Problem Description

A happy number is a number defined by the following process:

1. Starting with any positive integer, replace the number by the sum of the squares of its digits.
2. Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
3. Those numbers for which this process ends in 1 are happy.

### Solution

```python
def isHappy(n):
    """
    Use set to detect cycles in the sequence
    """
    def get_sum_of_squares(num):
        total = 0
        while num:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total
    
    seen = set()
    
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_sum_of_squares(n)
    
    return n == 1

# Alternative: Floyd's Cycle Detection (Tortoise and Hare)
def isHappyFloyd(n):
    """
    Use two pointers to detect cycle without extra space
    """
    def get_sum_of_squares(num):
        total = 0
        while num:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total
    
    slow = n
    fast = n
    
    while True:
        slow = get_sum_of_squares(slow)
        fast = get_sum_of_squares(get_sum_of_squares(fast))
        
        if fast == 1:
            return True
        
        if slow == fast:  # Cycle detected
            return False

# Test
print(isHappy(19))  # True (19 → 82 → 68 → 100 → 1)
print(isHappy(2))   # False
```

---

## Problem 5: Power of Two

**Difficulty**: 🟢 Easy  
**Pattern**: Bit Manipulation  
**Time**: O(1), **Space**: O(1)

### Problem Description

Given an integer `n`, return `true` if it is a power of two. Otherwise, return `false`.

### Solution

```python
def isPowerOfTwo(n):
    """
    Use bit manipulation: power of 2 has exactly one bit set
    """
    return n > 0 and (n & (n - 1)) == 0

# Alternative approaches
def isPowerOfTwoLoop(n):
    """
    Keep dividing by 2 until we get 1 or an odd number
    """
    if n <= 0:
        return False
    
    while n % 2 == 0:
        n //= 2
    
    return n == 1

def isPowerOfTwoRecursive(n):
    """
    Recursive approach
    """
    if n <= 0:
        return False
    if n == 1:
        return True
    if n % 2 != 0:
        return False
    
    return isPowerOfTwoRecursive(n // 2)

# Test
print(isPowerOfTwo(1))   # True (2^0)
print(isPowerOfTwo(16))  # True (2^4)
print(isPowerOfTwo(3))   # False
```

### 🔍 Key Insight

**Bit manipulation trick**: For powers of 2, `n & (n-1) == 0`

- `n = 8 = 1000₂`, `n-1 = 7 = 0111₂`, `8 & 7 = 0000₂ = 0`
- `n = 6 = 0110₂`, `n-1 = 5 = 0101₂`, `6 & 5 = 0100₂ ≠ 0`

---

## Problem 6: Factorial Trailing Zeroes

**Difficulty**: 🟢 Easy  
**Pattern**: Mathematical Analysis  
**Time**: O(log n), **Space**: O(1)

### Problem Description

Given an integer `n`, return the number of trailing zeroes in `n!`.

### Solution

```python
def trailingZeroes(n):
    """
    Count factors of 5 (since 2s are always more abundant)
    Trailing zeros = min(count of 2s, count of 5s) in n!
    """
    count = 0
    
    # Count multiples of 5, 25, 125, etc.
    while n >= 5:
        n //= 5
        count += n
    
    return count

# Alternative: More explicit approach
def trailingZeroesExplicit(n):
    """
    Explicitly count factors of 5 at each power
    """
    count = 0
    power_of_5 = 5
    
    while power_of_5 <= n:
        count += n // power_of_5
        power_of_5 *= 5
    
    return count

# Test
print(trailingZeroes(3))   # 0 (3! = 6)
print(trailingZeroes(5))   # 1 (5! = 120)
print(trailingZeroes(10))  # 2 (10! = 3628800)
```

### 🔍 Key Insights

- **Trailing zeros**: Created by factors of 10 = 2 × 5
- **Factors of 2**: Always more abundant than factors of 5
- **Count factors of 5**: n/5 + n/25 + n/125 + ...
- **Why this works**: Every 5th number contributes one factor of 5, every 25th number contributes additional factor, etc.

---

## Problem 7: Excel Sheet Column Number

**Difficulty**: 🟢 Easy  
**Pattern**: Base Conversion  
**Time**: O(n), **Space**: O(1)

### Problem Description

Given a string `columnTitle` that represents the column title as appear in an Excel sheet, return its corresponding column number.

**Examples:**
```
A -> 1
B -> 2
...
Z -> 26
AA -> 27
AB -> 28
```

### Solution

```python
def titleToNumber(columnTitle):
    """
    Convert from base-26 (A=1, B=2, ..., Z=26) to decimal
    """
    result = 0
    
    for char in columnTitle:
        result = result * 26 + (ord(char) - ord('A') + 1)
    
    return result

# Alternative: Right-to-left processing
def titleToNumberRTL(columnTitle):
    """
    Process from right to left with explicit powers
    """
    result = 0
    power = 1
    
    for char in reversed(columnTitle):
        result += (ord(char) - ord('A') + 1) * power
        power *= 26
    
    return result

# Test
print(titleToNumber("A"))    # 1
print(titleToNumber("AB"))   # 28
print(titleToNumber("ZY"))   # 701
```

---

## Problem 8: Count Primes

**Difficulty**: 🟢 Easy  
**Pattern**: Sieve of Eratosthenes  
**Time**: O(n log log n), **Space**: O(n)

### Problem Description

Count the number of prime numbers less than a non-negative integer, `n`.

### Solution

```python
def countPrimes(n):
    """
    Use Sieve of Eratosthenes to find all primes less than n
    """
    if n <= 2:
        return 0
    
    # Initialize boolean array
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False  # 0 and 1 are not prime
    
    # Sieve of Eratosthenes
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark all multiples of i as not prime
            for j in range(i * i, n, i):
                is_prime[j] = False
    
    return sum(is_prime)

# Optimized version
def countPrimesOptimized(n):
    """
    Optimized sieve with better memory usage
    """
    if n <= 2:
        return 0
    
    # Only track odd numbers (except 2)
    is_prime = [True] * ((n + 1) // 2)
    
    # Handle number 2 separately
    count = 1  # Count for 2
    
    # Check odd numbers from 3
    for i in range(3, int(n**0.5) + 1, 2):
        if is_prime[i // 2]:  # i is prime
            # Mark odd multiples of i
            for j in range(i * i, n, 2 * i):
                is_prime[j // 2] = False
    
    # Count remaining primes
    for i in range(1, len(is_prime)):
        if is_prime[i]:
            count += 1
    
    return count

# Test
print(countPrimes(10))  # 4 (primes: 2, 3, 5, 7)
print(countPrimes(0))   # 0
print(countPrimes(1))   # 0
```

---

## 📝 Summary

### Mathematical Patterns Mastered

1. **Digit Manipulation** - Reverse, palindrome, digit operations
2. **Number Theory** - Primes, factorials, powers, GCD/LCM
3. **Base Conversion** - Roman numerals, Excel columns, binary
4. **Cycle Detection** - Happy numbers, repeated sequences
5. **Bit Manipulation** - Power of 2, single bit operations
6. **Combinatorics** - Counting, permutations, factorials

### Essential Techniques

| **Technique** | **Use Case** | **Time** | **Example** |
|---------------|--------------|----------|-------------|
| **Digit Extraction** | Process individual digits | O(log n) | Palindrome, Reverse |
| **Sieve of Eratosthenes** | Find all primes up to n | O(n log log n) | Count Primes |
| **Bit Manipulation** | Check powers, single bits | O(1) | Power of Two |
| **Cycle Detection** | Find repeating patterns | O(n) | Happy Number |
| **Mathematical Analysis** | Count factors, patterns | O(log n) | Trailing Zeroes |

### Key Mathematical Insights

- **Palindromes**: Only need to check half the digits
- **Trailing Zeroes**: Count factors of 5 in factorial
- **Power of Two**: Exactly one bit set in binary representation
- **Prime Counting**: Sieve is much faster than checking each number individually
- **Cycle Detection**: Use Floyd's algorithm to save space

### Common Pitfalls

1. **Integer Overflow** - Always check bounds in reverse operations
2. **Edge Cases** - Handle 0, negative numbers, single digits
3. **Base Conversion** - Remember A=1, not A=0 in Excel columns
4. **Efficiency** - Use mathematical insights instead of brute force

---

Ready for more challenging mathematical problems? Move on to **[Medium Math Problems](medium-problems.md)** to tackle modular arithmetic, advanced number theory, and optimization problems!

### 📚 What's Next

- **[Number Theory](number-theory.md)** - Deep dive into mathematical concepts
- **[Geometry](geometry.md)** - Coordinate geometry and computational geometry
- **[Medium Problems](medium-problems.md)** - More complex mathematical challenges
- **[Dynamic Programming](../dp/index.md)** - Mathematical optimization with memoization
