# Mathematical Algorithms

## 📋 Overview

Mathematical algorithms form the foundation of many computational problems. These algorithms leverage mathematical concepts like number theory, combinatorics, and geometry to solve complex problems efficiently.

## 🔍 What You'll Learn

- **Number Theory**: Prime numbers, GCD/LCM, modular arithmetic
- **Combinatorics**: Permutations, combinations, counting principles
- **Geometry**: Computational geometry, distance calculations
- **Mathematical Techniques**: Fast exponentiation, sieve algorithms

## 📚 Section Contents

### 🔢 Number Theory

- **[Number Theory Fundamentals](number-theory.md)** - Primes, divisibility, modular arithmetic
- **[Prime Algorithms](primes.md)** - Sieve of Eratosthenes, primality testing
- **[GCD and LCM](gcd-lcm.md)** - Euclidean algorithm and applications
- **[Modular Arithmetic](modular.md)** - Fast exponentiation, inverse operations

### 🎲 Combinatorics

- **[Combinatorics Basics](combinatorics.md)** - Permutations, combinations, counting
- **[Probability](probability.md)** - Basic probability and expected value calculations

### 📐 Computational Geometry

- **[Geometry Fundamentals](geometry.md)** - Points, lines, distances, areas
- **[Advanced Geometry](advanced-geometry.md)** - Convex hull, line intersections

### 💪 Practice Problems

#### 🟢 Easy Problems
- **[Easy Math Problems](easy-problems.md)**
  - Basic arithmetic, simple number theory
  - GCD/LCM, prime checking

#### 🟡 Medium Problems  
- **[Medium Math Problems](medium-problems.md)**
  - Modular arithmetic, combinatorics
  - Geometric calculations

#### 🔴 Hard Problems
- **[Hard Math Problems](hard-problems.md)**
  - Advanced number theory, complex geometry
  - Mathematical optimization

## 🧮 Core Algorithms

### 1. **Euclidean Algorithm (GCD)**
```python
def gcd(a, b):
    """Greatest Common Divisor using Euclidean algorithm"""
    while b:
        a, b = b, a % b
    return a

def extended_gcd(a, b):
    """Extended Euclidean algorithm - finds x, y such that ax + by = gcd(a,b)"""
    if b == 0:
        return a, 1, 0
    
    gcd_val, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    
    return gcd_val, x, y
```

### 2. **Sieve of Eratosthenes**
```python
def sieve_of_eratosthenes(n):
    """Find all prime numbers up to n"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, n + 1) if is_prime[i]]
```

### 3. **Fast Exponentiation**
```python
def fast_power(base, exp, mod=None):
    """Compute base^exp efficiently, optionally modulo mod"""
    result = 1
    base = base % mod if mod else base
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod if mod else result * base
        exp = exp >> 1
        base = (base * base) % mod if mod else base * base
    
    return result
```

### 4. **Combinatorics Functions**
```python
def factorial_mod(n, mod):
    """Compute n! mod p efficiently"""
    result = 1
    for i in range(1, n + 1):
        result = (result * i) % mod
    return result

def nCr_mod(n, r, mod):
    """Compute C(n,r) mod p using modular arithmetic"""
    if r > n:
        return 0
    if r == 0 or r == n:
        return 1
    
    # C(n,r) = n! / (r! * (n-r)!)
    num = factorial_mod(n, mod)
    den = (factorial_mod(r, mod) * factorial_mod(n - r, mod)) % mod
    
    # Find modular inverse of denominator
    return (num * pow(den, mod - 2, mod)) % mod
```

## 📊 Algorithm Complexity

| Algorithm | Time Complexity | Space | Use Case |
|-----------|----------------|-------|----------|
| **Euclidean GCD** | O(log min(a,b)) | O(1) | GCD, LCM calculations |
| **Sieve of Eratosthenes** | O(n log log n) | O(n) | Generate all primes up to n |
| **Fast Exponentiation** | O(log n) | O(1) | Large power calculations |
| **Prime Test (Trial)** | O(√n) | O(1) | Check if number is prime |
| **Convex Hull** | O(n log n) | O(n) | Find convex boundary |

## 🎯 Common Problem Patterns

### 1. **Divisibility Problems**
```python
def count_divisors(n):
    """Count number of divisors of n"""
    count = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            count += 1 if i * i == n else 2
    return count
```

### 2. **Modular Arithmetic**
```python
def solve_linear_congruence(a, b, m):
    """Solve ax ≡ b (mod m)"""
    g, x, _ = extended_gcd(a, m)
    if b % g != 0:
        return None  # No solution
    
    x = (x * (b // g)) % m
    return x
```

### 3. **Geometric Calculations**
```python
def distance(p1, p2):
    """Euclidean distance between two points"""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def area_triangle(p1, p2, p3):
    """Area of triangle using cross product"""
    return abs((p1[0]*(p2[1] - p3[1]) + 
                p2[0]*(p3[1] - p1[1]) + 
                p3[0]*(p1[1] - p2[1])) / 2)
```

## 🔧 Problem-Solving Strategies

### For Number Theory
1. **Prime Factorization**: Break down numbers into prime factors
2. **Modular Properties**: Use (a+b) mod m = ((a mod m) + (b mod m)) mod m
3. **Euclidean Algorithm**: For GCD-related problems
4. **Sieve Techniques**: For problems involving multiple primes

### For Combinatorics
1. **Principle of Counting**: Multiplication and addition principles
2. **Inclusion-Exclusion**: Count by including and excluding cases
3. **Dynamic Programming**: For complex counting problems
4. **Modular Arithmetic**: Handle large numbers

### For Geometry
1. **Coordinate Geometry**: Use coordinates for precise calculations
2. **Vector Operations**: Cross product, dot product for angles/areas
3. **Sweep Line**: For intersection and closest pair problems
4. **Computational Complexity**: Consider precision and efficiency

## 🚀 Applications

### Cryptography
- **RSA**: Uses prime numbers and modular arithmetic
- **Hashing**: Mathematical functions for data integrity
- **Key Exchange**: Modular exponentiation for secure communication

### Computer Graphics
- **3D Transformations**: Matrix operations and geometry
- **Collision Detection**: Geometric intersection algorithms
- **Rendering**: Trigonometry and vector calculations

### Data Science
- **Probability**: Statistical calculations and modeling
- **Optimization**: Mathematical techniques for best solutions
- **Clustering**: Distance calculations and geometric algorithms

## 💡 Pro Tips

1. **Modular Arithmetic**: Always use mod to prevent overflow
2. **Edge Cases**: Handle n=0, n=1 carefully in mathematical functions
3. **Precision**: Be aware of floating-point precision in geometry
4. **Optimization**: Use mathematical properties to avoid brute force
5. **Precomputation**: Cache factorials, primes for repeated queries

---

*Mathematical algorithms are the building blocks of computational efficiency. Master these fundamentals to solve complex problems elegantly!*
