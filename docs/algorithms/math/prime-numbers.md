# Prime Numbers

## 🎯 Overview

Prime numbers are fundamental in number theory and have wide applications in cryptography, hashing, and other areas of computer science. A prime number is a natural number greater than 1 that is not divisible without remainder by any natural number other than 1 and itself.

## 📋 Core Concepts

### Prime Number Definition
A natural number greater than 1 that has no positive divisors other than 1 and itself.

### Properties of Prime Numbers
- There are infinitely many prime numbers
- 2 is the only even prime number
- Any natural number can be uniquely factored as a product of primes (Fundamental Theorem of Arithmetic)

## ⚙️ Prime Number Algorithms

### Prime Number Check

```python
def is_prime(n):
    """
    Check if a number is prime.
    Time complexity: O(sqrt(n))
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Check divisibility by numbers of the form 6k ± 1
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True
```

### Sieve of Eratosthenes

This algorithm efficiently finds all prime numbers up to any given limit.

```python
def sieve_of_eratosthenes(n):
    """
    Find all prime numbers up to n using the Sieve of Eratosthenes.
    Time complexity: O(n * log(log(n)))
    Space complexity: O(n)
    """
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark multiples of i as non-prime
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, n + 1) if is_prime[i]]
```

### Segmented Sieve

For finding primes in a range, especially when the range is very large:

```python
def segmented_sieve(low, high):
    """
    Find all primes in the range [low, high] using segmented sieve.
    Useful when high is very large but high-low is manageable.
    """
    # Find all primes up to sqrt(high) using regular sieve
    limit = int(high**0.5) + 1
    base_primes = sieve_of_eratosthenes(limit)
    
    # Initialize segment array
    segment_size = high - low + 1
    is_prime = [True] * segment_size
    
    # Mark multiples of base_primes in segment
    for p in base_primes:
        # Find the first multiple of p that is >= low
        start = max(p * p, ((low + p - 1) // p) * p)
        for i in range(start, high + 1, p):
            is_prime[i - low] = False
    
    # Collect primes from segment
    result = []
    for i in range(segment_size):
        if is_prime[i] and i + low >= 2:
            result.append(i + low)
            
    return result
```

### Prime Factorization

```python
def prime_factorization(n):
    """
    Decompose n into its prime factors.
    Returns a dictionary where keys are prime factors and values are their exponents.
    """
    factors = {}
    # Handle divisibility by 2 separately
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2
    
    # Check odd factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 2
    
    # If n is a prime number greater than 2
    if n > 2:
        factors[n] = factors.get(n, 0) + 1
    
    return factors
```

## 🔍 Complexity Analysis

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Prime Check (naive) | O(n) | O(1) |
| Prime Check (optimized) | O(√n) | O(1) |
| Sieve of Eratosthenes | O(n log log n) | O(n) |
| Segmented Sieve | O((high-low) log log high + √high) | O(√high + (high-low)) |
| Prime Factorization | O(√n) | O(log n) |

## 🧩 Applications

1. **Cryptography**: RSA encryption relies on the difficulty of factoring large prime numbers
2. **Hashing**: Prime numbers are used in hash function designs
3. **Random Number Generation**: Used in some random number generation algorithms
4. **Number Theory**: Fundamental to many number theoretical problems
5. **Primality Testing**: Used in blockchain and other security applications

## 📝 Practice Problems

1. **Count Primes**: Count the number of prime numbers less than a non-negative number n
2. **Prime Factorization**: Express a number as a product of its prime factors
3. **Goldbach's Conjecture**: Express an even integer as a sum of two primes
4. **Prime Gaps**: Find the largest gap between consecutive primes less than n

## 🌟 Pro Tips

- For primality testing of large numbers, consider probabilistic algorithms like Miller-Rabin
- Pre-compute primes using sieve when multiple primality tests are needed
- For very large ranges, use segmented sieve to reduce memory usage
- 2 and 3 are often special cases that can be handled separately for optimization

## 🔗 Related Algorithms

- [GCD and LCM](gcd-lcm.md)
- [Euclidean Algorithm](euclidean-algorithm.md)
- [Modular Arithmetic](modular-arithmetic.md)
- [Extended Euclidean Algorithm](extended-euclidean.md)
