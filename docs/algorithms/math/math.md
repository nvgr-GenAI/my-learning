# Mathematical Algorithms

Mathematical algorithms form the foundation of many computer science applications, from cryptography to computer graphics. This section covers essential mathematical concepts and their algorithmic implementations.

## Number Theory

### Greatest Common Divisor (GCD)

=== "Euclidean Algorithm"
    ```python
    def gcd(a, b):
        """
        Euclidean algorithm for GCD
        Time: O(log(min(a,b))), Space: O(1)
        """
        while b:
            a, b = b, a % b
        return a
    ```

=== "Extended Euclidean Algorithm"
    ```python
    def extended_gcd(a, b):
        """
        Returns gcd(a,b) and coefficients x,y such that ax + by = gcd(a,b)
        """
        if b == 0:
            return a, 1, 0
        
        gcd, x1, y1 = extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        
        return gcd, x, y
    ```

### Least Common Multiple (LCM)

```python
def lcm(a, b):
    """LCM using GCD relationship"""
    return abs(a * b) // gcd(a, b)

def lcm_multiple(numbers):
    """LCM of multiple numbers"""
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = lcm(result, numbers[i])
    return result
```

### Prime Numbers

=== "Sieve of Eratosthenes"
    ```python
    def sieve_of_eratosthenes(n):
        """
        Find all primes up to n
        Time: O(n log log n), Space: O(n)
        """
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        
        return [i for i in range(2, n + 1) if is_prime[i]]
    ```

=== "Primality Test"
    ```python
    def is_prime(n):
        """
        Check if number is prime
        Time: O(√n)
        """
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

=== "Miller-Rabin Test"
    ```python
    import random
    
    def miller_rabin(n, k=5):
        """
        Probabilistic primality test
        Time: O(k log³ n)
        """
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        
        # Write n-1 as d * 2^r
        r = 0
        d = n - 1
        while d % 2 == 0:
            d //= 2
            r += 1
        
        # Witness loop
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    ```

### Modular Arithmetic

```python
def mod_inverse(a, m):
    """
    Find modular multiplicative inverse of a modulo m
    Using extended Euclidean algorithm
    """
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError("Modular inverse doesn't exist")
    return (x % m + m) % m

def mod_power(base, exp, mod):
    """
    Fast modular exponentiation
    Time: O(log exp)
    """
    result = 1
    base = base % mod
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    
    return result

def chinese_remainder_theorem(remainders, moduli):
    """
    Solve system of congruences using CRT
    """
    if len(remainders) != len(moduli):
        raise ValueError("Lists must have same length")
    
    n = len(remainders)
    M = 1
    for m in moduli:
        M *= m
    
    result = 0
    for i in range(n):
        Mi = M // moduli[i]
        yi = mod_inverse(Mi, moduli[i])
        result += remainders[i] * Mi * yi
    
    return result % M
```

## Combinatorics

### Factorial and Permutations

```python
def factorial(n):
    """Calculate factorial"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def factorial_iterative(n):
    """Iterative factorial to avoid recursion depth"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def permutations(n, r):
    """Number of permutations P(n,r) = n!/(n-r)!"""
    if r > n:
        return 0
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    return result

def combinations(n, r):
    """Number of combinations C(n,r) = n!/(r!(n-r)!)"""
    if r > n or r < 0:
        return 0
    if r > n - r:  # Take advantage of symmetry
        r = n - r
    
    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)
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

def binomial_coefficient(n, k):
    """Calculate C(n,k) using Pascal's triangle property"""
    if k > n - k:
        k = n - k
    
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
```

## Geometry Algorithms

### Basic Geometric Primitives

```python
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance_to(self, other):
        """Euclidean distance between two points"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

def cross_product(o, a, b):
    """Cross product of vectors OA and OB"""
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

def orientation(p, q, r):
    """
    Find orientation of ordered triplet (p, q, r)
    Returns:
    0 -> p, q and r are colinear
    1 -> Clockwise
    2 -> Counterclockwise
    """
    val = cross_product(p, q, r)
    if val == 0:
        return 0
    return 1 if val > 0 else 2
```

### Convex Hull

```python
def convex_hull_graham(points):
    """
    Graham scan algorithm for convex hull
    Time: O(n log n), Space: O(n)
    """
    def polar_angle(p0, p1):
        return math.atan2(p1.y - p0.y, p1.x - p0.x)
    
    n = len(points)
    if n < 3:
        return points
    
    # Find bottom-most point (and leftmost in case of tie)
    start = min(points, key=lambda p: (p.y, p.x))
    
    # Sort points by polar angle with respect to start point
    sorted_points = sorted([p for p in points if p != start],
                          key=lambda p: (polar_angle(start, p), start.distance_to(p)))
    
    # Create hull
    hull = [start, sorted_points[0]]
    
    for i in range(1, len(sorted_points)):
        # Remove points that make clockwise turn
        while (len(hull) > 1 and 
               cross_product(hull[-2], hull[-1], sorted_points[i]) <= 0):
            hull.pop()
        hull.append(sorted_points[i])
    
    return hull

def convex_hull_jarvis(points):
    """
    Jarvis march (Gift wrapping) algorithm
    Time: O(nh) where h is number of hull points
    """
    n = len(points)
    if n < 3:
        return points
    
    # Find leftmost point
    leftmost = min(points, key=lambda p: p.x)
    
    hull = []
    current = leftmost
    
    while True:
        hull.append(current)
        
        # Find the most counterclockwise point
        next_point = points[0]
        for point in points[1:]:
            if (point == current or 
                cross_product(current, next_point, point) > 0 or
                (cross_product(current, next_point, point) == 0 and
                 current.distance_to(point) > current.distance_to(next_point))):
                next_point = point
        
        current = next_point
        if current == leftmost:  # Back to start
            break
    
    return hull
```

### Line Intersection

```python
def line_intersection(p1, q1, p2, q2):
    """
    Find intersection point of two line segments
    Returns None if lines don't intersect
    """
    def on_segment(p, q, r):
        """Check if point q lies on segment pr"""
        return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))
    
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    # General case
    if o1 != o2 and o3 != o4:
        # Calculate intersection point
        denom = (p1.x - q1.x) * (p2.y - q2.y) - (p1.y - q1.y) * (p2.x - q2.x)
        if denom == 0:
            return None
        
        t = ((p1.x - p2.x) * (p2.y - q2.y) - (p1.y - p2.y) * (p2.x - q2.x)) / denom
        x = p1.x + t * (q1.x - p1.x)
        y = p1.y + t * (q1.y - p1.y)
        return Point(x, y)
    
    # Special cases (collinear points)
    if (o1 == 0 and on_segment(p1, p2, q1)) or \
       (o2 == 0 and on_segment(p1, q2, q1)) or \
       (o3 == 0 and on_segment(p2, p1, q2)) or \
       (o4 == 0 and on_segment(p2, q1, q2)):
        return "Overlap"  # Segments overlap
    
    return None
```

## Matrix Operations

### Matrix Multiplication

```python
def matrix_multiply(A, B):
    """
    Standard matrix multiplication O(n³)
    """
    if len(A[0]) != len(B):
        raise ValueError("Incompatible matrix dimensions")
    
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    result = [[0] * cols_B for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

def matrix_power(matrix, n):
    """
    Fast matrix exponentiation O(log n)
    Useful for solving recurrence relations
    """
    size = len(matrix)
    
    # Identity matrix
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    base = [row[:] for row in matrix]  # Copy matrix
    
    while n > 0:
        if n % 2 == 1:
            result = matrix_multiply(result, base)
        base = matrix_multiply(base, base)
        n //= 2
    
    return result
```

### Gaussian Elimination

```python
def gaussian_elimination(A, b):
    """
    Solve system Ax = b using Gaussian elimination
    Returns solution vector x
    """
    n = len(A)
    
    # Augment matrix [A|b]
    augmented = [A[i][:] + [b[i]] for i in range(n)]
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        
        # Swap rows
        augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            if augmented[i][i] == 0:
                continue
            factor = augmented[k][i] / augmented[i][i]
            for j in range(i, n + 1):
                augmented[k][j] -= factor * augmented[i][j]
    
    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i][n]
        for j in range(i + 1, n):
            x[i] -= augmented[i][j] * x[j]
        x[i] /= augmented[i][i]
    
    return x
```

## Fast Fourier Transform (FFT)

```python
import cmath

def fft(x):
    """
    Fast Fourier Transform
    Time: O(n log n)
    """
    n = len(x)
    if n <= 1:
        return x
    
    # Divide
    even = fft([x[i] for i in range(0, n, 2)])
    odd = fft([x[i] for i in range(1, n, 2)])
    
    # Conquer
    T = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
    
    return [even[k] + T[k] for k in range(n // 2)] + \
           [even[k] - T[k] for k in range(n // 2)]

def ifft(x):
    """Inverse FFT"""
    n = len(x)
    # Conjugate the complex numbers
    x_conj = [complex(x[i].real, -x[i].imag) for i in range(n)]
    
    # Apply FFT
    result = fft(x_conj)
    
    # Conjugate again and scale
    return [complex(result[i].real / n, -result[i].imag / n) for i in range(n)]
```

## Applications

### Fast Fibonacci using Matrix Exponentiation

```python
def fibonacci_matrix(n):
    """
    Calculate Fibonacci number using matrix exponentiation
    Time: O(log n)
    """
    if n <= 1:
        return n
    
    # Fibonacci matrix [[1,1], [1,0]]
    fib_matrix = [[1, 1], [1, 0]]
    result_matrix = matrix_power(fib_matrix, n - 1)
    
    return result_matrix[0][0]
```

### Linear Recurrence Relations

```python
def solve_linear_recurrence(coeffs, initial, n):
    """
    Solve linear recurrence relation a_n = c1*a_{n-1} + c2*a_{n-2} + ... + ck*a_{n-k}
    Using matrix exponentiation
    """
    k = len(coeffs)
    if n < k:
        return initial[n]
    
    # Companion matrix
    companion = [[0] * k for _ in range(k)]
    
    # Last row contains coefficients
    companion[k-1] = coeffs[:]
    
    # Other rows form identity-like pattern
    for i in range(k-1):
        companion[i][i+1] = 1
    
    # Initial state vector [a_{k-1}, a_{k-2}, ..., a_0]
    state = initial[::-1]
    
    if n == k:
        return sum(coeffs[i] * initial[k-1-i] for i in range(k))
    
    # Calculate result using matrix power
    result_matrix = matrix_power(companion, n - k + 1)
    
    result = 0
    for i in range(k):
        result += result_matrix[k-1][i] * state[i]
    
    return result
```

## Resources

- [Number Theory Algorithms](https://cp-algorithms.com/algebra/)
- [Computational Geometry](https://www.geeksforgeeks.org/computational-geometry-set-1-introduction/)
- [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform)

---

*Mathematical algorithms are essential for solving complex computational problems efficiently. Master these concepts to tackle advanced algorithmic challenges!*
