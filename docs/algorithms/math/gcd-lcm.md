# Greatest Common Divisor (GCD) and Least Common Multiple (LCM)

## 🎯 Overview

GCD and LCM are fundamental concepts in number theory that have applications across many areas of computer science, from cryptography to fraction arithmetic to optimization algorithms.

## 📋 Core Concepts

### Greatest Common Divisor (GCD)

The largest positive integer that divides each of the given integers without a remainder.

Example: GCD(12, 18) = 6 because 6 is the largest integer that divides both 12 and 18.

### Least Common Multiple (LCM)

The smallest positive integer that is divisible by each of the given integers.

Example: LCM(12, 18) = 36 because 36 is the smallest integer that is divisible by both 12 and 18.

### Key Relationship

The GCD and LCM of two numbers are related by this formula:

$$LCM(a, b) = \frac{a \times b}{GCD(a, b)}$$

## ⚙️ Algorithms

### Computing GCD

#### Euclidean Algorithm

The most efficient method to calculate GCD:

```python
def gcd(a, b):
    """Calculate GCD using the Euclidean algorithm"""
    while b:
        a, b = b, a % b
    return a
```

#### Binary GCD Algorithm (Stein's Algorithm)

This algorithm is more efficient when dealing with large numbers:

```python
def binary_gcd(a, b):
    """Calculate GCD using Stein's algorithm (Binary GCD)"""
    if a == 0:
        return b
    if b == 0:
        return a
    
    # Find common factor of 2
    common_factor_of_2 = 0
    while (a | b) & 1 == 0:  # Both a and b are even
        a >>= 1
        b >>= 1
        common_factor_of_2 += 1
    
    # Remove all factors of 2 from a
    while a & 1 == 0:
        a >>= 1
    
    # From here on, a is always odd
    while b != 0:
        # Remove all factors of 2 from b (as they don't contribute to GCD)
        while b & 1 == 0:
            b >>= 1
        
        # Swap if needed so that a ≥ b
        if a < b:
            a, b = b, a
        
        # Subtract (a-b will be even)
        a -= b
    
    # Restore the common factors of 2
    return a << common_factor_of_2
```

### Computing LCM

Using the relation between GCD and LCM:

```python
def lcm(a, b):
    """Calculate LCM using the GCD"""
    return a * b // gcd(a, b)
```

### GCD of Multiple Numbers

```python
def gcd_multiple(numbers):
    """Calculate GCD of multiple numbers"""
    result = numbers[0]
    for num in numbers[1:]:
        result = gcd(result, num)
    return result
```

### LCM of Multiple Numbers

```python
def lcm_multiple(numbers):
    """Calculate LCM of multiple numbers"""
    result = numbers[0]
    for num in numbers[1:]:
        result = lcm(result, num)
    return result
```

## 🔍 Complexity Analysis

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Euclidean GCD | O(log(min(a, b))) | O(1) |
| Binary GCD | O(log(min(a, b))) | O(1) |
| LCM | O(log(min(a, b))) | O(1) |
| GCD of n numbers | O(n × log(max(numbers))) | O(1) |
| LCM of n numbers | O(n × log(max(numbers))) | O(1) |

## 🧩 Applications

1. **Fraction Simplification**: Reducing fractions to lowest terms
2. **Cryptography**: Essential in algorithms like RSA
3. **Computer Graphics**: Finding proper divisors for screen resolutions
4. **Number Theory Problems**: Solving Diophantine equations
5. **Combinatorial Calculations**: Computing combinations and arrangements efficiently

## 📝 Practice Problems

1. **Fraction Addition**: Add/subtract fractions using LCM for denominators
2. **Coprime Check**: Determine if two numbers are coprime (GCD is 1)
3. **GCD Sum**: Find the sum of GCD(i, n) for all 1 ≤ i ≤ n
4. **Array GCD**: Find the GCD of all elements in an array
5. **LCM Challenge**: Find the LCM of first n natural numbers

## 🌟 Pro Tips

- Always remember that GCD(0, n) = n for any n > 0
- If GCD(a, b) = 1, we say a and b are coprime or relatively prime
- For negative numbers, GCD(a, b) = GCD(|a|, |b|)
- When calculating LCM of multiple numbers, compute pairwise to avoid overflow
- Use the formula LCM(a,b) = a*b / GCD(a,b) but implement it as a=(a/GCD(a,b))*b to avoid overflow

## 🔗 Related Algorithms

- [Euclidean Algorithm](euclidean-algorithm.md)
- [Extended Euclidean Algorithm](extended-euclidean.md)
- [Modular Arithmetic](modular-arithmetic.md)
- [Chinese Remainder Theorem](chinese-remainder.md)
