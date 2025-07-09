# Chinese Remainder Theorem (CRT)

## 🎯 Overview

The Chinese Remainder Theorem (CRT) is a fundamental result in number theory that provides a way to solve systems of linear congruences with coprime moduli. It states that if one knows the remainders of the Euclidean division of an integer n by several integers, then one can determine uniquely the remainder of n modulo the product of these integers, under certain conditions.

The theorem is named "Chinese" because it was discovered by the Chinese mathematician Sun Tzu in the 3rd century CE.

## 📋 Core Concepts

### The Problem

Given a system of congruences:
- x ≡ a₁ (mod m₁)
- x ≡ a₂ (mod m₂)
- ...
- x ≡ aₙ (mod mₙ)

Where m₁, m₂, ..., mₙ are pairwise coprime (gcd of any pair is 1), find x modulo M, where M = m₁ × m₂ × ... × mₙ.

### The Solution

1. Compute M = m₁ × m₂ × ... × mₙ
2. For each i, compute Mᵢ = M / mᵢ
3. For each i, compute the modular multiplicative inverse of Mᵢ modulo mᵢ, call it M⁻¹ᵢ
4. The solution is: x ≡ (∑ aᵢ × Mᵢ × M⁻¹ᵢ) (mod M)

## ⚙️ Algorithm Implementation

```python
def extended_gcd(a, b):
    """Returns (gcd, x, y) such that a*x + b*y = gcd"""
    if a == 0:
        return (b, 0, 1)
    else:
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return (gcd, x, y)

def mod_inverse(a, m):
    """Compute the modular multiplicative inverse of a modulo m"""
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        raise Exception('Modular inverse does not exist')
    else:
        return x % m

def chinese_remainder_theorem(remainders, moduli):
    """
    Solve the system of congruences using Chinese Remainder Theorem
    remainders[i] = x mod moduli[i]
    Returns x modulo product of all moduli
    """
    # Check if moduli are pairwise coprime
    for i in range(len(moduli)):
        for j in range(i+1, len(moduli)):
            if extended_gcd(moduli[i], moduli[j])[0] != 1:
                raise Exception('Moduli are not pairwise coprime')
    
    # Calculate product of all moduli
    M = 1
    for m in moduli:
        M *= m
    
    # Calculate partial products and their inverses
    result = 0
    for i in range(len(remainders)):
        a_i = remainders[i]
        m_i = moduli[i]
        M_i = M // m_i  # M / m_i
        M_i_inv = mod_inverse(M_i, m_i)  # Inverse of M_i modulo m_i
        
        result += a_i * M_i * M_i_inv
    
    return result % M
```

### Example Usage

```python
# Solve the system:
# x ≡ 2 (mod 3)
# x ≡ 3 (mod 5)
# x ≡ 2 (mod 7)
remainders = [2, 3, 2]
moduli = [3, 5, 7]
solution = chinese_remainder_theorem(remainders, moduli)
print(f"Solution: x ≡ {solution} (mod {3*5*7})")
```

## 🔍 How It Works

Let's work through a simple example:
- x ≡ 2 (mod 3)
- x ≡ 3 (mod 5)
- x ≡ 2 (mod 7)

1. Calculate M = 3 × 5 × 7 = 105
2. Calculate M₁ = 105/3 = 35, M₂ = 105/5 = 21, M₃ = 105/7 = 15
3. Calculate inverses:
   - 35⁻¹ ≡ 2 (mod 3) since 35 × 2 ≡ 1 (mod 3)
   - 21⁻¹ ≡ 1 (mod 5) since 21 × 1 ≡ 1 (mod 5)
   - 15⁻¹ ≡ 1 (mod 7) since 15 × 1 ≡ 1 (mod 7)
4. Compute solution:
   - x ≡ (2 × 35 × 2 + 3 × 21 × 1 + 2 × 15 × 1) (mod 105)
   - x ≡ (140 + 63 + 30) (mod 105)
   - x ≡ 233 (mod 105)
   - x ≡ 23 (mod 105)

So the unique solution modulo 105 is x ≡ 23.

## ⚙️ Complexity Analysis

- **Time Complexity**: O(n + log(M)), where n is the number of congruences and M is the product of all moduli
  - Computing modular inverses takes O(log(m)) for each modulus m
  - Overall computation of the solution is O(n)
- **Space Complexity**: O(n)

## 🧩 Applications

1. **Cryptography**: Used in the RSA algorithm and other cryptographic systems
2. **Calendar Calculations**: Computing dates in various calendar systems
3. **Error Correction Codes**: Used in coding theory
4. **Number Systems**: Converting between different number representations
5. **Optimization Problems**: Solving certain types of constraint satisfaction problems

## 📝 Practice Problems

1. **Basic CRT**: Solve simple systems of congruences
2. **Calendar Problems**: Determine the day of the week for a given date using CRT
3. **Large Numbers**: Find the smallest positive integer that leaves specific remainders when divided by different numbers
4. **CRT with Non-Coprime Moduli**: Extend the algorithm to handle cases where moduli aren't coprime

## 🌟 Pro Tips

- Ensure that all moduli are pairwise coprime before applying CRT
- For non-coprime moduli, you need to use a generalized version of CRT
- When implementing, be careful about integer overflow for large moduli
- Use precomputed values of modular inverses if the same system needs to be solved multiple times
- Remember that the solution is unique modulo the product of all moduli
- Use fast modular exponentiation when computing modular inverses using Fermat's Little Theorem

## 🔗 Related Algorithms

- [Extended Euclidean Algorithm](extended-euclidean.md)
- [Modular Arithmetic](modular-arithmetic.md)
- [Modular Multiplicative Inverse](modular-arithmetic.md#finding-modular-inverse)
- [Diophantine Equations](diophantine-equations.md)
