# Extended Euclidean Algorithm

## 🎯 Overview

The Extended Euclidean Algorithm is an extension of the Euclidean Algorithm that, in addition to finding the Greatest Common Divisor (GCD) of two integers, also finds the coefficients of Bézout's identity. That is, it finds integers x and y such that:

$$ax + by = gcd(a, b)$$

This algorithm is particularly important in number theory and has applications in cryptography, modular arithmetic, and solving Diophantine equations.

## 📋 Core Concepts

### Bézout's Identity

For any two integers a and b, there exist integers x and y such that:

$$ax + by = gcd(a, b)$$

The Extended Euclidean Algorithm efficiently computes these coefficients x and y.

## ⚙️ Algorithm Implementation

### Iterative Implementation

```python
def extended_gcd(a, b):
    """
    Returns the GCD of a and b, as well as coefficients x and y such that
    ax + by = gcd(a, b)
    """
    if a == 0:
        return (b, 0, 1)
    
    # Initialize values
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    
    # At this point, old_r is the gcd, and old_s and old_t are the coefficients
    return (old_r, old_s, old_t)
```

### Recursive Implementation

```python
def extended_gcd_recursive(a, b):
    """
    Recursive implementation of the Extended Euclidean Algorithm
    Returns (gcd, x, y) such that ax + by = gcd
    """
    if a == 0:
        return (b, 0, 1)
    
    gcd, x1, y1 = extended_gcd_recursive(b % a, a)
    
    x = y1 - (b // a) * x1
    y = x1
    
    return (gcd, x, y)
```

## 🔍 How It Works

Let's trace through the algorithm with an example: Finding the GCD of 42 and 30, and the coefficients x and y.

| Step | a  | b  | quotient | r  | x  | y  |
|------|----|----|----------|----|----|-----|
| 0    | 42 | 30 | -        | -  | 1  | 0   |
| 1    | 30 | 12 | 1        | 12 | 0  | 1   |
| 2    | 12 | 6  | 2        | 6  | 1  | -2  |
| 3    | 6  | 0  | 2        | 0  | -1 | 3   |

So, GCD(42, 30) = 6, and we have coefficients x = -1, y = 3 such that:
42 × (-1) + 30 × 3 = 6

## 🧩 Applications

1. **Modular Multiplicative Inverse**: Find the modular inverse of a number (a⁻¹ mod m)
2. **Solving Linear Diophantine Equations**: Find integer solutions to ax + by = c
3. **Chinese Remainder Theorem**: Solving systems of modular congruences
4. **Public Key Cryptography**: Used in RSA and other cryptographic algorithms
5. **Number Theory Problems**: Solving various number-theoretic problems

### Finding Modular Multiplicative Inverse

When gcd(a, m) = 1, the modular multiplicative inverse of a modulo m exists and can be found using the Extended Euclidean Algorithm:

```python
def mod_inverse(a, m):
    """
    Compute the modular multiplicative inverse of a modulo m.
    Returns x such that (a * x) % m == 1 if such x exists.
    """
    gcd, x, y = extended_gcd(a, m)
    
    if gcd != 1:
        raise Exception('Modular inverse does not exist')
    else:
        return x % m
```

## ⚙️ Complexity Analysis

- **Time Complexity**: O(log(min(a, b)))
  - Similar to the regular Euclidean algorithm
- **Space Complexity**: O(1) for iterative, O(log(min(a, b))) for recursive

## 📝 Practice Problems

1. **Modular Inverse**: Find the modular inverse of a number modulo m
2. **Linear Diophantine Equations**: Find integer solutions to ax + by = c
3. **GCD Property**: Prove that if gcd(a, b) = d, then gcd(a/d, b/d) = 1
4. **System of Equations**: Solve systems of linear congruences using the Chinese Remainder Theorem

## 🌟 Pro Tips

- Use the iterative version for better space complexity and to avoid stack overflow for large inputs
- The algorithm also proves that the GCD of two numbers is always expressible as a linear combination of those numbers
- If gcd(a, m) ≠ 1, the modular inverse of a modulo m doesn't exist
- Carefully handle the sign of coefficients when implementing the algorithm

## 🔗 Related Algorithms

- [Euclidean Algorithm](euclidean-algorithm.md)
- [Modular Arithmetic](modular-arithmetic.md)
- [Chinese Remainder Theorem](chinese-remainder.md)
- [Diophantine Equations](diophantine-equations.md)
