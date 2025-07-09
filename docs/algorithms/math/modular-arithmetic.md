# Modular Arithmetic

## 🎯 Overview

Modular arithmetic is a system of arithmetic for integers where numbers "wrap around" upon reaching a certain value, called the modulus. It's often described as "clock arithmetic" because it resembles the way hours on a clock wrap around after 12.

Modular arithmetic is crucial in computer science for applications ranging from cryptography to hash functions to handling overflow in integer operations.

## 📋 Core Concepts

### Basic Operations

For integers a, b, and modulus m:

- **Addition**: (a + b) mod m = ((a mod m) + (b mod m)) mod m
- **Subtraction**: (a - b) mod m = ((a mod m) - (b mod m) + m) mod m
- **Multiplication**: (a × b) mod m = ((a mod m) × (b mod m)) mod m
- **Exponentiation**: (a^b) mod m = ((a mod m)^b) mod m

### Modular Congruence

Two integers a and b are congruent modulo m if they have the same remainder when divided by m:

a ≡ b (mod m) if m divides (a - b)

Example: 17 ≡ 2 (mod 5) because 17 mod 5 = 2 and 2 mod 5 = 2

### Modular Inverse

The modular multiplicative inverse of an integer a with respect to modulus m is an integer a⁻¹ such that:

a × a⁻¹ ≡ 1 (mod m)

The inverse exists if and only if a and m are coprime (gcd(a, m) = 1).

## ⚙️ Algorithm Implementations

### Basic Modular Operations

```python
def mod_add(a, b, m):
    """Addition under modulo m"""
    return (a % m + b % m) % m

def mod_subtract(a, b, m):
    """Subtraction under modulo m"""
    return (a % m - b % m + m) % m

def mod_multiply(a, b, m):
    """Multiplication under modulo m"""
    return (a % m * b % m) % m
```

### Fast Modular Exponentiation (Binary Exponentiation)

```python
def mod_pow(base, exponent, modulus):
    """
    Compute (base^exponent) % modulus efficiently
    Time complexity: O(log(exponent))
    """
    if modulus == 1:
        return 0
    
    result = 1
    base = base % modulus
    
    while exponent > 0:
        # If exponent is odd, multiply result with base
        if exponent % 2 == 1:
            result = (result * base) % modulus
            
        # Exponent is even now
        exponent = exponent >> 1  # exponent = exponent // 2
        base = (base * base) % modulus
    
    return result
```

### Finding Modular Inverse

Using the Extended Euclidean Algorithm:

```python
def extended_gcd(a, b):
    """
    Returns (gcd, x, y) such that a*x + b*y = gcd
    """
    if a == 0:
        return (b, 0, 1)
    
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return (gcd, x, y)

def mod_inverse(a, m):
    """
    Find the modular multiplicative inverse of a under modulo m
    Returns an integer x such that a*x ≡ 1 (mod m)
    """
    gcd, x, y = extended_gcd(a, m)
    
    if gcd != 1:
        raise Exception("Modular inverse doesn't exist")
    else:
        return (x % m + m) % m  # Ensure the result is positive
```

For special case when m is prime, using Fermat's Little Theorem:

```python
def mod_inverse_prime(a, m):
    """
    Find modular inverse using Fermat's Little Theorem.
    Works only when m is prime.
    a^(m-1) ≡ 1 (mod m), therefore a^(m-2) ≡ a^(-1) (mod m)
    """
    return mod_pow(a, m - 2, m)
```

## 🔍 Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Basic Operations (±, ×) | O(1) | O(1) |
| Fast Modular Exponentiation | O(log n) | O(1) |
| Modular Inverse (Extended GCD) | O(log m) | O(1) |
| Modular Inverse (Fermat) | O(log m) | O(1) |

## 🧩 Applications

1. **Cryptography**: RSA, Diffie-Hellman, and many other cryptographic algorithms
2. **Hash Functions**: Computing hash codes for hash tables
3. **Random Number Generation**: Linear congruential generators
4. **Computer Graphics**: Wrapping textures and calculating cyclic patterns
5. **Error Detection**: Checksum algorithms
6. **Competitive Programming**: Solving number theory problems efficiently

## 📝 Practice Problems

1. **Modular Exponentiation**: Calculate a^b mod m for large values efficiently
2. **Modular Inverse**: Find the modular inverse of a number
3. **Linear Congruence**: Solve linear congruences of the form ax ≡ b (mod m)
4. **Chinese Remainder Theorem**: Solve systems of linear congruences
5. **Fibonacci Modular**: Find the n-th Fibonacci number modulo m efficiently

## 🌟 Pro Tips

- Always apply modulo operation at each step to avoid integer overflow
- For large exponentiation, use binary exponentiation method (mod_pow)
- Remember that (a/b) mod m is NOT equal to ((a mod m) / (b mod m)) mod m
- Instead, use modular inverse: (a/b) mod m = (a * b^(-1)) mod m
- When working with negative numbers, add the modulus to ensure positive result: ((a % m) + m) % m
- Avoid division in modular arithmetic; use multiplication by modular inverse instead

## 🔗 Related Algorithms

- [Euclidean Algorithm](euclidean-algorithm.md)
- [Extended Euclidean Algorithm](extended-euclidean.md)
- [Chinese Remainder Theorem](chinese-remainder.md)
- [Binary Exponentiation](binary-exponentiation.md)
