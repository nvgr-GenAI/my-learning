# Euclidean Algorithm

## 🎯 Overview

The Euclidean Algorithm is a highly efficient method for finding the Greatest Common Divisor (GCD) of two numbers. It is one of the oldest algorithms in common use, dating back to Euclid's Elements (circa 300 BCE).

## 📋 Core Concepts

The algorithm works on a simple principle:
- If we have two positive integers a and b where a > b
- Then gcd(a, b) = gcd(b, a mod b)
- Continue this process until the remainder is 0

### Basic Euclidean Algorithm

```python
def gcd_euclidean(a, b):
    """
    Calculate the Greatest Common Divisor of a and b using the Euclidean algorithm.
    """
    while b:
        a, b = b, a % b
    return a
```

### Recursive Implementation

```python
def gcd_recursive(a, b):
    """
    Recursive implementation of the Euclidean algorithm
    """
    if b == 0:
        return a
    return gcd_recursive(b, a % b)
```

## 🔍 How It Works

The algorithm works by repeatedly applying the property that `gcd(a, b) = gcd(b, a mod b)`.

Let's trace through an example: Finding gcd(48, 18)

| Step | a  | b  | a % b |
|------|----|----|-------|
| 1    | 48 | 18 | 12    |
| 2    | 18 | 12 | 6     |
| 3    | 12 | 6  | 0     |

Since b = 0 at the end, the gcd is 6.

## ⚙️ Complexity Analysis

- **Time Complexity**: O(log(min(a, b)))
  - The number of steps is proportional to the logarithm of the smaller number
- **Space Complexity**: O(1) for iterative, O(log(min(a, b))) for recursive due to call stack

## 🧩 Applications

1. **Finding GCD**: Directly calculates the GCD of two integers
2. **Reducing Fractions**: Used to simplify fractions to lowest terms
3. **Modular Arithmetic**: Foundation for solving modular equations
4. **Cryptography**: Used in RSA algorithm and other encryption methods
5. **Extended Euclidean Algorithm**: Basis for finding modular multiplicative inverses

## 📝 Practice Problems

1. **Find the GCD of multiple numbers**: Extend the algorithm to find the GCD of three or more integers.
2. **Least Common Multiple (LCM)**: Use the relationship `lcm(a, b) = (a * b) / gcd(a, b)` to find the LCM.
3. **Coprime Check**: Determine if two numbers are coprime (their GCD is 1).

## 🌟 Pro Tips

- The Euclidean algorithm is much faster than checking all possible factors, especially for large numbers.
- The GCD of two numbers is always positive, regardless of the input signs.
- The GCD of 0 and any number n is n itself.
- If both numbers are 0, their GCD is undefined (sometimes defined as 0).

## 🔗 Related Algorithms

- [Extended Euclidean Algorithm](extended-euclidean.md)
- [GCD and LCM](gcd-lcm.md)
- [Modular Arithmetic](modular-arithmetic.md)
- [Diophantine Equations](diophantine-equations.md)
