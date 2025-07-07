# Mathematical Algorithms - Fundamentals

## 🎯 Overview

Mathematical algorithms form the backbone of computational problem-solving, providing efficient solutions to numerical, geometric, and logical challenges. This section covers essential mathematical concepts and their algorithmic implementations.

=== "📋 Core Mathematical Concepts"

    ## **Number Theory**
    
    | Concept | Description | Applications |
    |---------|-------------|--------------|
    | **Prime Numbers** | Numbers divisible only by 1 and themselves | Cryptography, hashing |
    | **GCD/LCM** | Greatest Common Divisor / Least Common Multiple | Fraction reduction, scheduling |
    | **Modular Arithmetic** | Arithmetic with remainder operations | Cryptography, hash functions |
    | **Euler's Totient** | Count of integers ≤ n that are coprime to n | Number theory, encryption |
    | **Chinese Remainder Theorem** | System of congruences solution | Distributed computing |

    ## **Combinatorics**
    
    | Concept | Formula | Use Cases |
    |---------|---------|-----------|
    | **Permutations** | P(n,r) = n!/(n-r)! | Arrangements, sequences |
    | **Combinations** | C(n,r) = n!/(r!(n-r)!) | Selections, probability |
    | **Factorial** | n! = n × (n-1) × ... × 1 | Counting, arrangements |
    | **Catalan Numbers** | C(n) = (2n)!/(n!(n+1)!) | Binary trees, parentheses |
    | **Fibonacci** | F(n) = F(n-1) + F(n-2) | Growth patterns, optimization |

=== "🔢 Essential Algorithms"

    ## **Prime Number Algorithms**
    
    ```python
    # Sieve of Eratosthenes
    def sieve_of_eratosthenes(n):
        primes = [True] * (n + 1)
        primes[0] = primes[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if primes[i]:
                for j in range(i*i, n + 1, i):
                    primes[j] = False
        
        return [i for i in range(2, n + 1) if primes[i]]
    
    # Primality Test
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    ```
    
    ## **GCD and LCM**
    
    ```python
    def gcd(a, b):
        """Euclidean algorithm for GCD"""
        while b:
            a, b = b, a % b
        return a
    
    def lcm(a, b):
        """LCM using GCD"""
        return abs(a * b) // gcd(a, b)
    
    def extended_gcd(a, b):
        """Extended Euclidean algorithm"""
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    ```
    
    ## **Modular Arithmetic**
    
    ```python
    def mod_pow(base, exp, mod):
        """Fast modular exponentiation"""
        result = 1
        base = base % mod
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            exp = exp >> 1
            base = (base * base) % mod
        return result
    
    def mod_inverse(a, mod):
        """Modular multiplicative inverse"""
        gcd, x, _ = extended_gcd(a, mod)
        if gcd != 1:
            return None  # Inverse doesn't exist
        return (x % mod + mod) % mod
    ```

=== "🧮 Combinatorial Algorithms"

    ## **Factorials and Permutations**
    
    ```python
    def factorial(n):
        """Calculate n! efficiently"""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    def permutation(n, r):
        """Calculate P(n,r) = n!/(n-r)!"""
        if r > n or r < 0:
            return 0
        result = 1
        for i in range(n, n - r, -1):
            result *= i
        return result
    
    def combination(n, r):
        """Calculate C(n,r) = n!/(r!(n-r)!)"""
        if r > n or r < 0:
            return 0
        if r > n - r:
            r = n - r
        result = 1
        for i in range(r):
            result = result * (n - i) // (i + 1)
        return result
    ```
    
    ## **Fibonacci Sequence**
    
    ```python
    def fibonacci(n):
        """Calculate nth Fibonacci number"""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def fibonacci_matrix(n):
        """Fast Fibonacci using matrix exponentiation"""
        if n <= 1:
            return n
        
        def matrix_multiply(A, B):
            return [[A[0][0] * B[0][0] + A[0][1] * B[1][0],
                     A[0][0] * B[0][1] + A[0][1] * B[1][1]],
                    [A[1][0] * B[0][0] + A[1][1] * B[1][0],
                     A[1][0] * B[0][1] + A[1][1] * B[1][1]]]
        
        def matrix_power(matrix, power):
            if power == 1:
                return matrix
            if power % 2 == 0:
                half = matrix_power(matrix, power // 2)
                return matrix_multiply(half, half)
            else:
                return matrix_multiply(matrix, matrix_power(matrix, power - 1))
        
        base = [[1, 1], [1, 0]]
        result = matrix_power(base, n)
        return result[0][1]
    ```

=== "📐 Geometric Algorithms"

    ## **Basic Geometry**
    
    ```python
    import math
    
    def distance(p1, p2):
        """Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def area_triangle(p1, p2, p3):
        """Area of triangle using cross product"""
        return abs((p1[0] * (p2[1] - p3[1]) + 
                   p2[0] * (p3[1] - p1[1]) + 
                   p3[0] * (p1[1] - p2[1])) / 2)
    
    def are_collinear(p1, p2, p3):
        """Check if three points are collinear"""
        return area_triangle(p1, p2, p3) == 0
    
    def line_intersection(p1, p2, p3, p4):
        """Find intersection of two lines"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # Lines are parallel
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    ```
    
    ## **Coordinate Geometry**
    
    ```python
    def slope(p1, p2):
        """Calculate slope between two points"""
        if p1[0] == p2[0]:
            return float('inf')  # Vertical line
        return (p2[1] - p1[1]) / (p2[0] - p1[0])
    
    def angle_between_vectors(v1, v2):
        """Angle between two vectors"""
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        return math.acos(cos_angle)
    ```

=== "🎯 Problem-Solving Patterns"

    ## **Mathematical Problem Types**
    
    | Problem Type | Approach | Common Algorithms |
    |--------------|----------|-------------------|
    | **Digit Manipulation** | Extract/process digits | Modulo, division operations |
    | **Base Conversion** | Convert between number systems | Repeated division/multiplication |
    | **Optimization** | Find minimum/maximum | Calculus, binary search |
    | **Counting** | Enumerate possibilities | Combinatorics, DP |
    | **Pattern Recognition** | Identify mathematical sequences | Sequence analysis, formula derivation |
    | **Geometric** | Spatial relationships | Coordinate geometry, trigonometry |
    
    ## **Common Mathematical Tricks**
    
    ```python
    # Check if number is power of 2
    def is_power_of_2(n):
        return n > 0 and (n & (n - 1)) == 0
    
    # Fast multiplication by powers of 2
    def multiply_by_power_of_2(n, k):
        return n << k  # n * 2^k
    
    # Check if number is palindrome
    def is_palindrome(n):
        if n < 0:
            return False
        original = n
        reversed_n = 0
        while n > 0:
            reversed_n = reversed_n * 10 + n % 10
            n //= 10
        return original == reversed_n
    
    # Sum of arithmetic sequence
    def arithmetic_sum(first, last, count):
        return count * (first + last) // 2
    
    # Sum of geometric sequence
    def geometric_sum(first, ratio, count):
        if ratio == 1:
            return first * count
        return first * (ratio**count - 1) // (ratio - 1)
    ```

=== "⚡ Optimization Techniques"

    ## **Time Complexity Optimization**
    
    | Technique | Original | Optimized | Example |
    |-----------|----------|-----------|---------|
    | **Memoization** | O(2^n) | O(n²) | Fibonacci sequence |
    | **Sieve** | O(n√n) | O(n log log n) | Prime generation |
    | **Binary Search** | O(n) | O(log n) | Search in sorted array |
    | **Matrix Exponentiation** | O(n) | O(log n) | Linear recurrence |
    | **Bit Manipulation** | O(n) | O(1) | Power of 2 check |
    
    ## **Space Optimization**
    
    ```python
    # Space-optimized Fibonacci
    def fib_optimized(n):
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    # In-place prime sieve
    def sieve_optimized(n):
        # Use array to mark composites
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        
        return [i for i in range(2, n + 1) if is_prime[i]]
    ```

=== "📊 Complexity Analysis"

    ## **Time Complexities**
    
    | Algorithm | Best Case | Average Case | Worst Case |
    |-----------|-----------|--------------|------------|
    | **Prime Check** | O(1) | O(√n) | O(√n) |
    | **GCD (Euclidean)** | O(1) | O(log n) | O(log n) |
    | **Modular Exponentiation** | O(1) | O(log n) | O(log n) |
    | **Sieve of Eratosthenes** | O(n log log n) | O(n log log n) | O(n log log n) |
    | **Matrix Multiplication** | O(n³) | O(n³) | O(n³) |
    
    ## **Space Complexities**
    
    | Algorithm | Space Complexity | Notes |
    |-----------|------------------|-------|
    | **Iterative Fibonacci** | O(1) | Constant space |
    | **Recursive Fibonacci** | O(n) | Call stack |
    | **Sieve Array** | O(n) | Boolean array |
    | **Memoization** | O(n) | Cache storage |
    | **Matrix Operations** | O(n²) | Matrix storage |

---

*Master these mathematical fundamentals to build a strong foundation for advanced algorithmic problem-solving!*
