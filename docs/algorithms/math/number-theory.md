# Number Theory

## Prime Numbers

### Sieve of Eratosthenes

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

### Prime Factorization

```python
def prime_factors(n):
    """Find all prime factors of n"""
    factors = []
    d = 2
    
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    
    if n > 1:
        factors.append(n)
    
    return factors

def prime_factors_count(n):
    """Count of each prime factor"""
    factors = {}
    d = 2
    
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    return factors
```

### Check if Prime

```python
def is_prime(n):
    """Check if n is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    
    return True
```

## GCD and LCM

### Greatest Common Divisor

```python
def gcd(a, b):
    """Euclidean algorithm for GCD"""
    while b:
        a, b = b, a % b
    return a

def gcd_recursive(a, b):
    """Recursive GCD"""
    if b == 0:
        return a
    return gcd_recursive(b, a % b)

def gcd_multiple(numbers):
    """GCD of multiple numbers"""
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = gcd(result, numbers[i])
    return result
```

### Least Common Multiple

```python
def lcm(a, b):
    """LCM using GCD"""
    return abs(a * b) // gcd(a, b)

def lcm_multiple(numbers):
    """LCM of multiple numbers"""
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = lcm(result, numbers[i])
    return result
```

## Modular Arithmetic

### Modular Exponentiation

```python
def power_mod(base, exp, mod):
    """Calculate (base^exp) % mod efficiently"""
    result = 1
    base = base % mod
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    
    return result
```

### Modular Inverse

```python
def mod_inverse(a, m):
    """Find modular inverse of a under modulo m"""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        return None  # Modular inverse doesn't exist
    return (x % m + m) % m
```

## Combinatorics

### Factorial

```python
def factorial(n):
    """Calculate n!"""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def factorial_mod(n, mod):
    """Calculate n! % mod"""
    result = 1
    for i in range(1, n + 1):
        result = (result * i) % mod
    return result
```

### Combinations and Permutations

```python
def combination(n, r):
    """Calculate C(n, r) = n! / (r! * (n-r)!)"""
    if r > n or r < 0:
        return 0
    if r == 0 or r == n:
        return 1
    
    # Use the property C(n,r) = C(n,n-r)
    r = min(r, n - r)
    
    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)
    
    return result

def permutation(n, r):
    """Calculate P(n, r) = n! / (n-r)!"""
    if r > n or r < 0:
        return 0
    
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    
    return result
```

### Pascal's Triangle

```python
def generate_pascals_triangle(n):
    """Generate first n rows of Pascal's triangle"""
    triangle = []
    
    for i in range(n):
        row = [1] * (i + 1)
        for j in range(1, i):
            row[j] = triangle[i-1][j-1] + triangle[i-1][j]
        triangle.append(row)
    
    return triangle
```

## Number Systems

### Base Conversion

```python
def decimal_to_base(num, base):
    """Convert decimal to any base"""
    if num == 0:
        return "0"
    
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    
    while num > 0:
        result = digits[num % base] + result
        num //= base
    
    return result

def base_to_decimal(num_str, base):
    """Convert from any base to decimal"""
    result = 0
    power = 0
    
    for digit in reversed(num_str):
        if digit.isdigit():
            digit_val = int(digit)
        else:
            digit_val = ord(digit.upper()) - ord('A') + 10
        
        result += digit_val * (base ** power)
        power += 1
    
    return result
```

## Applications

### Finding Divisors

```python
def find_divisors(n):
    """Find all divisors of n"""
    divisors = []
    
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    return sorted(divisors)

def count_divisors(n):
    """Count number of divisors"""
    count = 0
    
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    
    return count
```

## Practice Problems

- [ ] Count Primes
- [ ] Ugly Number II
- [ ] Perfect Squares
- [ ] Happy Number
- [ ] Factorial Trailing Zeroes
- [ ] Power of Two
- [ ] Power of Three
- [ ] Excel Sheet Column Number
