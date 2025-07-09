# Diophantine Equations

## 🎯 Overview

Diophantine equations are polynomial equations with integer coefficients where integer solutions are sought. Named after the ancient Greek mathematician Diophantus, these equations are fundamental in number theory and have applications in cryptography, computer science, and mathematical modeling.

The most common type is the linear Diophantine equation in the form:

$$ax + by = c$$

Where we seek integer solutions for x and y, given integers a, b, and c.

## 📋 Core Concepts

### Existence of Solutions

A linear Diophantine equation ax + by = c has integer solutions if and only if c is divisible by the greatest common divisor (GCD) of a and b.

Specifically:
- If gcd(a, b) divides c, then there are infinitely many solutions
- If gcd(a, b) does not divide c, then there are no solutions

### Structure of Solutions

If (x₀, y₀) is a particular solution to ax + by = c, then all solutions are given by:

$$x = x₀ + k \cdot \frac{b}{d}$$
$$y = y₀ - k \cdot \frac{a}{d}$$

Where d = gcd(a, b) and k is any integer.

## ⚙️ Algorithm Implementation

### Solving Linear Diophantine Equations

```python
def extended_gcd(a, b):
    """
    Extended Euclidean Algorithm
    Returns (gcd, x, y) such that a*x + b*y = gcd
    """
    if a == 0:
        return (b, 0, 1)
    else:
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return (gcd, x, y)

def diophantine(a, b, c):
    """
    Solves the Diophantine equation ax + by = c
    Returns a particular solution (x0, y0) if exists, otherwise None
    """
    # Find gcd and coefficients using Extended Euclidean Algorithm
    gcd, x0, y0 = extended_gcd(abs(a), abs(b))
    
    # Check if solution exists
    if c % gcd != 0:
        return None  # No solution
    
    # Adjust for sign of a and b
    x0 *= 1 if a > 0 else -1
    y0 *= 1 if b > 0 else -1
    
    # Scale to get the solution for the original equation
    x0 = x0 * (c // gcd)
    y0 = y0 * (c // gcd)
    
    return (x0, y0)

def all_solutions(a, b, c, num_solutions=5):
    """
    Generate several solutions to ax + by = c
    """
    # Find a particular solution
    particular = diophantine(a, b, c)
    if particular is None:
        return []
    
    x0, y0 = particular
    gcd = extended_gcd(abs(a), abs(b))[0]
    
    solutions = []
    for k in range(-num_solutions // 2, num_solutions // 2 + 1):
        x = x0 + k * (b // gcd)
        y = y0 - k * (a // gcd)
        solutions.append((x, y))
    
    return solutions
```

### Handling Non-Linear Diophantine Equations

While the general case is much more complex, here's how to solve a specific case - Pell's equation:

```python
def pells_equation_solver(d):
    """
    Find the smallest positive solution to x² - d·y² = 1
    Uses the continued fraction method
    """
    if int(d**0.5)**2 == d:  # d is a perfect square
        return None  # No non-trivial solutions
    
    # Initialize values for continued fraction algorithm
    a0 = int(d**0.5)
    m, d_new = 0, 1
    a = a0
    
    # Convergents
    p_prev, p = 1, a0
    q_prev, q = 0, 1
    
    while p*p - d*q*q != 1:
        m = a * d_new - m
        d_new = (d - m*m) // d_new
        a = (a0 + m) // d_new
        
        p_new = a * p + p_prev
        q_new = a * q + q_prev
        
        p_prev, p = p, p_new
        q_prev, q = q, q_new
    
    return (p, q)
```

## 🔍 How It Works

Let's solve the equation 3x + 6y = 12:

1. Calculate gcd(3, 6) = 3
2. Check if gcd(3, 6) = 3 divides 12? Yes, 12 ÷ 3 = 4
3. Use the extended GCD to find x₀, y₀ such that 3x₀ + 6y₀ = 3
   - This gives us x₀ = 1, y₀ = 0
4. Scale to get a solution for the original equation:
   - x = x₀ * (c/gcd) = 1 * 4 = 4
   - y = y₀ * (c/gcd) = 0 * 4 = 0
5. A particular solution is (4, 0)
6. All solutions are given by:
   - x = 4 + k * (6/3) = 4 + 2k
   - y = 0 - k * (3/3) = 0 - k
   - For integer k, we get solutions like (4, 0), (6, -1), (2, 1), (8, -2), etc.

## ⚙️ Complexity Analysis

- **Time Complexity for Linear Diophantine**: O(log(min(|a|, |b|))), dominated by the extended GCD computation
- **Space Complexity**: O(1)
- **Time Complexity for Pell's Equation**: O(log(d) * sqrt(d)) in the worst case

## 🧩 Applications

1. **Cryptography**: Solving equations in RSA and other cryptographic systems
2. **Integer Programming**: Many problems can be reduced to finding integer solutions to equations
3. **Computer Graphics**: Calculating Bezier curves and other geometric functions with integer coordinates
4. **Combinatorial Problems**: Solving counting problems with multiple constraints
5. **Financial Mathematics**: Problems involving indivisible assets or when fractional solutions are not allowed

## 📝 Practice Problems

1. **Basic Linear Diophantine**: Find all integer solutions to 5x + 7y = 23
2. **Constrained Solutions**: Find positive integer solutions to ax + by = c
3. **System of Diophantine Equations**: Solve multiple equations simultaneously
4. **Pell's Equation**: Find the smallest solution to x² - dy² = 1 for different values of d
5. **Chicken and Rabbit Problem**: Classical word problem that translates to a Diophantine equation

## 🌟 Pro Tips

- Always check if gcd(a, b) divides c before attempting to find solutions
- To find positive solutions, use the property that if (x, y) is a solution, so is (x + kb/gcd, y - ka/gcd)
- For equations with more variables, reduce them to a sequence of equations with fewer variables
- For non-linear Diophantine equations, there are no general algorithms, but specific equations have specialized methods
- Continued fractions are particularly useful for solving Pell's equation and similar quadratic Diophantine equations

## 🔗 Related Algorithms

- [Euclidean Algorithm](euclidean-algorithm.md)
- [Extended Euclidean Algorithm](extended-euclidean.md)
- [Modular Arithmetic](modular-arithmetic.md)
- [Chinese Remainder Theorem](chinese-remainder.md)
- [Continued Fractions](continued-fractions.md)
