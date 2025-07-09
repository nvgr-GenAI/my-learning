# Newton's Method

Newton's method (also known as the Newton-Raphson method) is a powerful numerical technique used to find successively better approximations to the roots (or zeroes) of a real-valued function.

## Concept and Theory

Newton's method is based on the simple idea of linear approximation. For a function f(x), we can approximate it at a point x₀ using its tangent line:

$$f(x) \approx f(x_0) + f'(x_0)(x - x_0)$$

To find the root of f(x), we set f(x) = 0 and solve for x:

$$0 = f(x_0) + f'(x_0)(x - x_0)$$
$$x = x_0 - \frac{f(x_0)}{f'(x_0)}$$

This gives us the iterative formula:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

## Algorithm Implementation

```python
def newton_method(f, f_prime, x0, tolerance=1e-6, max_iterations=100):
    """
    Find the root of function f using Newton's method.
    
    Parameters:
    f: Function for which we want to find the root
    f_prime: Derivative of f
    x0: Initial guess
    tolerance: Error tolerance
    max_iterations: Maximum number of iterations
    
    Returns:
    x: Approximated root
    iterations: Number of iterations performed
    """
    x = x0
    iterations = 0
    
    while iterations < max_iterations:
        # Compute the function value and its derivative
        f_value = f(x)
        f_prime_value = f_prime(x)
        
        # Check if the derivative is close to zero to avoid division by zero
        if abs(f_prime_value) < 1e-10:
            print("Derivative too small, Newton's method failed")
            return None, iterations
        
        # Compute the next approximation
        x_next = x - f_value / f_prime_value
        
        # Check for convergence
        if abs(x_next - x) < tolerance:
            return x_next, iterations
        
        x = x_next
        iterations += 1
    
    print("Maximum iterations reached, Newton's method may not have converged")
    return x, iterations
```

### Example Usage:

```python
# Define a function and its derivative
def f(x):
    return x**3 - 2*x - 5

def f_prime(x):
    return 3*x**2 - 2

# Find a root starting with initial guess x0 = 2
root, iterations = newton_method(f, f_prime, 2.0)
print(f"Root found: {root} after {iterations} iterations")
print(f"f({root}) = {f(root)}")  # Should be close to 0
```

## Convergence and Analysis

### Time Complexity
- Each iteration involves evaluating f and f' at a point, which is typically O(1)
- The number of iterations depends on:
  - The initial guess
  - The tolerance
  - The nature of the function

### Convergence Properties
- When it converges, Newton's method exhibits quadratic convergence, meaning the number of correct digits roughly doubles with each iteration
- For a root r, if x₀ is sufficiently close to r, then:
  $$|x_{n+1} - r| \approx C|x_n - r|^2$$
  where C is some constant

### Conditions for Convergence
- The function must be differentiable
- The derivative must not be zero at the root
- The initial guess must be sufficiently close to a root

## Handling Multiple Roots and Convergence Issues

### Multiple Roots
When a function has multiple roots, Newton's method will converge to one of them depending on the initial guess.

### Convergence Issues
1. **Division by zero**: Occurs when f'(xₙ) = 0. This happens at critical points of f.
2. **Cycling**: The method might cycle between points without converging.
3. **Divergence**: For certain functions and initial points, the sequence might diverge.

### Modified Newton's Method
For multiple roots, we can use a modified Newton's method:

$$x_{n+1} = x_n - m \cdot \frac{f(x_n)}{f'(x_n)}$$

where m is the multiplicity of the root.

## Applications

1. **Solving Nonlinear Equations**: Finding roots of complex equations
2. **Optimization**: Finding extrema of functions (by finding roots of the derivative)
3. **Numerical Methods**: Used in numerical analysis for:
   - Computing square roots and other nth roots
   - Computing reciprocals of numbers
   - Finding intersections of curves
4. **Machine Learning**: Used in optimization algorithms like Newton-Raphson for gradient descent

## Extensions and Variations

### Newton-Raphson for Systems of Equations
Newton's method can be extended to systems of nonlinear equations using the Jacobian matrix:

$$\mathbf{x}_{n+1} = \mathbf{x}_n - [J_f(\mathbf{x}_n)]^{-1}f(\mathbf{x}_n)$$

### Quasi-Newton Methods
These methods approximate the Jacobian or Hessian matrix to reduce computational cost:
- Broyden's method
- BFGS method
- DFP method

## Practice Problems

1. Use Newton's method to find the square root of a number
2. Solve the equation x³ - 3x² + 2 = 0 using Newton's method
3. Find the intersection of the curves y = sin(x) and y = x²/4
4. Implement Newton's method for finding cube roots
5. Use Newton's method to find the minimum of a function

## Pro Tips

1. **Good Initial Guess**: The closer your initial guess is to the actual root, the faster the convergence.
2. **Check Derivative**: Ensure the derivative isn't zero or very small at your initial guess.
3. **Multiple Starting Points**: Try multiple initial guesses to find different roots.
4. **Graphical Analysis**: Visualize the function to understand its behavior and choose better initial points.
5. **Damping**: For some functions, using a damped version of Newton's method helps with convergence:
   $$x_{n+1} = x_n - \alpha \frac{f(x_n)}{f'(x_n)}$$
   where 0 < α ≤ 1 is the damping factor.

## Related Topics

- [Binary Exponentiation](binary-exponentiation.md)
- [Fast Fourier Transform](fft.md)
- [Linear Algebra Basics](linear-algebra.md)
- [Numerical Methods](../numerical-methods.md)
