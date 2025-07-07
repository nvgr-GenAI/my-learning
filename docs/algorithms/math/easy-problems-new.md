# Mathematical Algorithms - Easy Problems

## 🎯 Learning Objectives

Master fundamental mathematical concepts in programming:

- Basic number theory operations
- Simple combinatorics and probability
- Elementary geometry calculations
- Digit manipulation and arithmetic
- Mathematical patterns and sequences

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Palindrome Number | Digit Manipulation | Easy | O(log n) | O(1) |
    | 2 | Reverse Integer | Digit Manipulation | Easy | O(log n) | O(1) |
    | 3 | Roman to Integer | String/Math | Easy | O(n) | O(1) |
    | 4 | Integer to Roman | String/Math | Easy | O(1) | O(1) |
    | 5 | Power of Two | Bit Manipulation | Easy | O(1) | O(1) |
    | 6 | Power of Three | Math/Recursion | Easy | O(log n) | O(1) |
    | 7 | Fibonacci Number | DP/Math | Easy | O(n) | O(1) |
    | 8 | Climbing Stairs | DP/Math | Easy | O(n) | O(1) |
    | 9 | Plus One | Array/Math | Easy | O(n) | O(1) |
    | 10 | Add Binary | String/Math | Easy | O(max(m,n)) | O(max(m,n)) |
    | 11 | Sqrt(x) | Binary Search/Math | Easy | O(log n) | O(1) |
    | 12 | Valid Perfect Square | Binary Search/Math | Easy | O(log n) | O(1) |
    | 13 | Happy Number | Math/Cycle Detection | Easy | O(log n) | O(1) |
    | 14 | Excel Sheet Column Number | Math/Base Conversion | Easy | O(n) | O(1) |
    | 15 | Count Primes | Sieve of Eratosthenes | Easy | O(n log log n) | O(n) |

=== "🎯 Core Math Patterns"

    **🔢 Number Theory:**
    - Prime factorization and divisibility
    - Greatest common divisor (GCD) and least common multiple (LCM)
    - Modular arithmetic and properties
    
    **🧮 Digit Manipulation:**
    - Extracting and reconstructing digits
    - Reversing numbers and overflow handling
    - Base conversion and representation
    
    **🔄 Mathematical Sequences:**
    - Fibonacci and recursive sequences
    - Arithmetic and geometric progressions
    - Pattern recognition in sequences

=== "📚 Algorithm Templates"

    **Digit Manipulation Template:**
    ```python
    def process_digits(n):
        result = 0
        while n > 0:
            digit = n % 10
            # Process digit here
            result = result * 10 + digit  # If building new number
            n //= 10
        return result
    ```
    
    **GCD Template (Euclidean Algorithm):**
    ```python
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    ```
    
    **Prime Checking Template:**
    ```python
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    ```

---

## Problem 1: Palindrome Number

**Difficulty**: 🟢 Easy  
**Pattern**: Digit Manipulation  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    Given an integer `x`, return `true` if `x` is palindrome integer.

    An integer is a palindrome when it reads the same backward as forward.

    **Examples:**
    ```
    Input: x = 121
    Output: true

    Input: x = -121
    Output: false
    
    Input: x = 10
    Output: false
    ```

=== "Solution"

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

=== "Insights"

    **Key Insights:**
    - **Negative numbers**: Always false (contains '-' which doesn't reverse)
    - **Trailing zeros**: Only 0 itself is palindrome among numbers ending in 0
    - **Half reversal**: More efficient than reversing entire number
    - **Odd vs even digits**: Handle middle digit in odd-length numbers

---

## Problem 2: Reverse Integer

**Difficulty**: 🟢 Easy  
**Pattern**: Digit Manipulation  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

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

=== "Solution"

    ```python
    def reverse(x):
        """
        Reverse integer with overflow checking
        """
        INT_MAX = 2**31 - 1
        INT_MIN = -2**31
        
        sign = -1 if x < 0 else 1
        x = abs(x)
        
        result = 0
        while x:
            digit = x % 10
            x //= 10
            
            # Check for overflow before adding digit
            if result > INT_MAX // 10:
                return 0
            if result == INT_MAX // 10 and digit > INT_MAX % 10:
                return 0
            
            result = result * 10 + digit
        
        return sign * result

    # Test
    print(reverse(123))    # 321
    print(reverse(-123))   # -321
    print(reverse(120))    # 21
    ```

=== "Insights"

    **Key Insights:**
    - **Overflow detection**: Check before multiplying by 10
    - **Sign handling**: Work with absolute value, apply sign at end
    - **Trailing zeros**: Automatically handled by integer division
    - **Edge cases**: INT_MAX and INT_MIN boundaries

---

## Problem 3: Roman to Integer

**Difficulty**: 🟢 Easy  
**Pattern**: String/Math  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

    Given a roman numeral, convert it to an integer.

    **Examples:**
    ```
    Input: s = "III"
    Output: 3

    Input: s = "LVIII"
    Output: 58 (L = 50, V = 5, III = 3)

    Input: s = "MCMXC"
    Output: 1994 (M = 1000, CM = 900, XC = 90)
    ```

=== "Solution"

    ```python
    def romanToInt(s):
        """
        Convert Roman numeral to integer using subtraction rule
        """
        values = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }
        
        result = 0
        prev_value = 0
        
        for char in reversed(s):
            value = values[char]
            if value < prev_value:
                result -= value
            else:
                result += value
            prev_value = value
        
        return result

    # Alternative: Left-to-right approach
    def romanToIntLTR(s):
        """
        Convert Roman numeral using left-to-right parsing
        """
        values = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }
        
        result = 0
        i = 0
        
        while i < len(s):
            if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
                result += values[s[i + 1]] - values[s[i]]
                i += 2
            else:
                result += values[s[i]]
                i += 1
        
        return result

    # Test
    print(romanToInt("III"))     # 3
    print(romanToInt("LVIII"))   # 58
    print(romanToInt("MCMXC"))   # 1994
    ```

=== "Insights"

    **Key Insights:**
    - **Subtraction rule**: When smaller numeral precedes larger, subtract it
    - **Right-to-left**: Easier to track when to subtract vs add
    - **Mapping**: Use dictionary for O(1) symbol lookup
    - **Edge cases**: Handle subtractive combinations (IV, IX, XL, XC, CD, CM)

---

## Problem 4: Power of Two

**Difficulty**: 🟢 Easy  
**Pattern**: Bit Manipulation  
**Time**: O(1), **Space**: O(1)

=== "Problem"

    Given an integer `n`, return `true` if it is a power of two. Otherwise, return `false`.

    An integer `n` is a power of two, if there exists an integer `x` such that `n == 2^x`.

    **Examples:**
    ```
    Input: n = 1
    Output: true (2^0 = 1)

    Input: n = 16
    Output: true (2^4 = 16)

    Input: n = 3
    Output: false
    ```

=== "Solution"

    ```python
    def isPowerOfTwo(n):
        """
        Use bit manipulation: power of 2 has exactly one bit set
        """
        return n > 0 and (n & (n - 1)) == 0

    # Alternative: Mathematical approach
    def isPowerOfTwoMath(n):
        """
        Check if n is positive and divides largest power of 2
        """
        return n > 0 and (1 << 30) % n == 0

    # Alternative: Iterative approach
    def isPowerOfTwoIterative(n):
        """
        Keep dividing by 2 until we get 1 or odd number
        """
        if n <= 0:
            return False
        
        while n % 2 == 0:
            n //= 2
        
        return n == 1

    # Test
    print(isPowerOfTwo(1))   # True
    print(isPowerOfTwo(16))  # True
    print(isPowerOfTwo(3))   # False
    ```

=== "Insights"

    **Key Insights:**
    - **Bit manipulation**: `n & (n-1) == 0` for powers of 2
    - **Single bit**: Powers of 2 have exactly one bit set
    - **Edge cases**: Handle zero and negative numbers
    - **Mathematical property**: All powers of 2 divide 2^30

---

## Problem 5: Fibonacci Number

**Difficulty**: 🟢 Easy  
**Pattern**: DP/Math  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1.

    Given `n`, calculate F(n).

    **Examples:**
    ```
    Input: n = 2
    Output: 1 (F(2) = F(1) + F(0) = 1 + 0 = 1)

    Input: n = 3
    Output: 2 (F(3) = F(2) + F(1) = 1 + 1 = 2)

    Input: n = 4
    Output: 3 (F(4) = F(3) + F(2) = 2 + 1 = 3)
    ```

=== "Solution"

    ```python
    def fib(n):
        """
        Iterative approach with O(1) space
        """
        if n <= 1:
            return n
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        
        return b

    # Alternative: Matrix exponentiation O(log n)
    def fibMatrix(n):
        """
        Use matrix exponentiation for O(log n) time
        """
        if n <= 1:
            return n
        
        def matrix_mult(A, B):
            return [[A[0][0] * B[0][0] + A[0][1] * B[1][0],
                     A[0][0] * B[0][1] + A[0][1] * B[1][1]],
                    [A[1][0] * B[0][0] + A[1][1] * B[1][0],
                     A[1][0] * B[0][1] + A[1][1] * B[1][1]]]
        
        def matrix_power(matrix, power):
            if power == 1:
                return matrix
            if power % 2 == 0:
                half = matrix_power(matrix, power // 2)
                return matrix_mult(half, half)
            else:
                return matrix_mult(matrix, matrix_power(matrix, power - 1))
        
        base = [[1, 1], [1, 0]]
        result = matrix_power(base, n)
        return result[0][1]

    # Alternative: Golden ratio formula
    def fibGolden(n):
        """
        Use golden ratio formula (Binet's formula)
        """
        if n <= 1:
            return n
        
        phi = (1 + 5**0.5) / 2
        return int((phi**n - (-phi)**(-n)) / 5**0.5)

    # Test
    print(fib(10))  # 55
    print(fibMatrix(10))  # 55
    print(fibGolden(10))  # 55
    ```

=== "Insights"

    **Key Insights:**
    - **Iterative approach**: Most practical with O(1) space
    - **Matrix exponentiation**: Fastest for large n with O(log n) time
    - **Golden ratio**: Mathematical elegance but precision issues
    - **Base cases**: F(0) = 0, F(1) = 1

---

## Problem 6: Count Primes

**Difficulty**: 🟢 Easy  
**Pattern**: Sieve of Eratosthenes  
**Time**: O(n log log n), **Space**: O(n)

=== "Problem"

    Given an integer `n`, return the number of prime numbers that are less than `n`.

    **Examples:**
    ```
    Input: n = 10
    Output: 4 (2, 3, 5, 7 are primes less than 10)

    Input: n = 0
    Output: 0

    Input: n = 1
    Output: 0
    ```

=== "Solution"

    ```python
    def countPrimes(n):
        """
        Sieve of Eratosthenes - most efficient for counting primes
        """
        if n <= 2:
            return 0
        
        # Initialize all numbers as prime
        is_prime = [True] * n
        is_prime[0] = is_prime[1] = False
        
        # Sieve process
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                # Mark multiples of i as non-prime
                for j in range(i * i, n, i):
                    is_prime[j] = False
        
        return sum(is_prime)

    # Alternative: Optimized sieve
    def countPrimesOptimized(n):
        """
        Optimized sieve with odd numbers only
        """
        if n <= 2:
            return 0
        if n <= 3:
            return 1
        
        # Only consider odd numbers (except 2)
        is_prime = [True] * (n // 2)
        
        for i in range(3, int(n**0.5) + 1, 2):
            if is_prime[i // 2]:
                # Mark multiples of i as non-prime
                for j in range(i * i, n, 2 * i):
                    is_prime[j // 2] = False
        
        return 1 + sum(is_prime)  # +1 for prime number 2

    # Test
    print(countPrimes(10))  # 4
    print(countPrimesOptimized(10))  # 4
    ```

=== "Insights"

    **Key Insights:**
    - **Sieve efficiency**: Mark multiples instead of checking each number
    - **Optimization**: Only check up to √n for factors
    - **Space optimization**: Only store odd numbers (except 2)
    - **Starting point**: Mark multiples starting from i²

---

## 📝 Summary

### Math Problem Patterns Mastered

1. **Digit Manipulation** - Reversing, palindromes, extraction
2. **Number Theory** - Primes, GCD, modular arithmetic
3. **Base Conversion** - Roman numerals, binary representation
4. **Bit Manipulation** - Powers of 2, efficient operations
5. **Sequences** - Fibonacci, arithmetic progressions
6. **Overflow Handling** - 32-bit integer constraints

### Key Algorithms

| **Algorithm** | **Time** | **Space** | **Use Cases** |
|---------------|----------|-----------|---------------|
| **Sieve of Eratosthenes** | O(n log log n) | O(n) | Prime counting/generation |
| **Euclidean Algorithm** | O(log min(a,b)) | O(1) | GCD computation |
| **Matrix Exponentiation** | O(log n) | O(1) | Fast fibonacci/recurrence |
| **Digit Manipulation** | O(log n) | O(1) | Number processing |

### Problem-Solving Tips

1. **Edge Cases**: Always consider 0, negative numbers, overflow
2. **Optimization**: Use bit manipulation for powers of 2
3. **Mathematical Properties**: Leverage number theory for efficiency
4. **Space-Time Tradeoffs**: Choose between iterative vs memoized approaches

---

## 🏆 Congratulations!

You've mastered fundamental mathematical algorithms! These skills are essential for:

- **Algorithm Competitions** - Number theory problems
- **Cryptography** - Prime generation, modular arithmetic
- **Computer Graphics** - Geometric calculations
- **Data Analysis** - Statistical computations
- **System Design** - Hash functions, load balancing

### 📚 What's Next

- **[Medium Problems](medium-problems.md)** - Advanced mathematical algorithms
- **[Hard Problems](hard-problems.md)** - Complex number theory and optimization
- **[Number Theory](../advanced/number-theory.md)** - Deep mathematical concepts
- **[Competitive Programming](../advanced/competitive-programming.md)** - Contest-level problems

*Continue building your mathematical foundation for advanced algorithmic problem-solving!*
