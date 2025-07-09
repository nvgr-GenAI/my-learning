# Linear Algebra Basics

Linear algebra is a branch of mathematics that deals with vector spaces, linear transformations, systems of linear equations, and their representations using matrices and vectors. It forms the foundation of many algorithms in computer science, machine learning, computer graphics, and optimization.

## Vectors

A vector is an ordered collection of numbers. In an n-dimensional space, a vector has n components.

### Vector Operations

#### Vector Addition

```python
def vector_add(v, w):
    """Add corresponding elements"""
    return [v_i + w_i for v_i, w_i in zip(v, w)]
```

#### Scalar Multiplication

```python
def scalar_multiply(c, v):
    """Multiply each element by a scalar c"""
    return [c * v_i for v_i in v]
```

#### Dot Product

```python
def dot_product(v, w):
    """Sum of component-wise products"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
```

#### Vector Magnitude (Length)

```python
import math

def magnitude(v):
    """Length of the vector"""
    return math.sqrt(sum(v_i ** 2 for v_i in v))
```

#### Vector Distance

```python
def distance(v, w):
    """Euclidean distance between vectors"""
    return magnitude([v_i - w_i for v_i, w_i in zip(v, w)])
```

## Matrices

A matrix is a rectangular array of numbers arranged in rows and columns.

### Matrix Operations

#### Matrix Addition

```python
def matrix_add(A, B):
    """Add corresponding elements"""
    n, m = len(A), len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]
```

#### Scalar Multiplication

```python
def matrix_scalar_multiply(c, A):
    """Multiply each element by a scalar c"""
    n, m = len(A), len(A[0])
    return [[c * A[i][j] for j in range(m)] for i in range(n)]
```

#### Matrix Multiplication

```python
def matrix_multiply(A, B):
    """Multiply matrices A and B"""
    n, m, p = len(A), len(A[0]), len(B[0])
    result = [[0 for _ in range(p)] for _ in range(n)]
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                result[i][j] += A[i][k] * B[k][j]
    
    return result
```

#### Matrix Transpose

```python
def matrix_transpose(A):
    """Swap rows with columns"""
    n, m = len(A), len(A[0])
    return [[A[j][i] for j in range(n)] for i in range(m)]
```

## Systems of Linear Equations

A system of linear equations can be represented as a matrix equation Ax = b.

### Gaussian Elimination

Gaussian elimination is used to solve systems of linear equations by transforming the augmented matrix [A|b] into row echelon form.

```python
def gaussian_elimination(A, b):
    """Solve Ax = b using Gaussian elimination"""
    n = len(A)
    # Create the augmented matrix
    augmented = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination
    for i in range(n):
        # Find the maximum element in the current column
        max_element = abs(augmented[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > max_element:
                max_element = abs(augmented[k][i])
                max_row = k
        
        # Swap rows
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Eliminate current column below diagonal
        for k in range(i + 1, n):
            c = -augmented[k][i] / augmented[i][i]
            for j in range(i, n + 1):
                if i == j:
                    augmented[k][j] = 0
                else:
                    augmented[k][j] += c * augmented[i][j]
    
    # Back substitution
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i][n] / augmented[i][i]
        for k in range(i - 1, -1, -1):
            augmented[k][n] -= augmented[k][i] * x[i]
    
    return x
```

## Matrix Properties

### Determinant

The determinant of a matrix is a scalar value that can be calculated from its elements. For a 2×2 matrix, the determinant is:

$$\det\left(\begin{bmatrix} a & b \\ c & d \end{bmatrix}\right) = ad - bc$$

For larger matrices, the determinant can be calculated recursively using cofactor expansion.

```python
def determinant(A):
    """Calculate the determinant of a square matrix"""
    n = len(A)
    
    # Base case for 1x1 matrix
    if n == 1:
        return A[0][0]
    
    # Base case for 2x2 matrix
    if n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    
    # Recursive case for larger matrices
    result = 0
    for j in range(n):
        # Create the submatrix by excluding row 0 and column j
        submatrix = [[A[i][k] for k in range(n) if k != j] for i in range(1, n)]
        # Add the term to the result (with the appropriate sign)
        sign = (-1) ** j
        result += sign * A[0][j] * determinant(submatrix)
    
    return result
```

### Inverse Matrix

The inverse of a matrix A is another matrix A⁻¹ such that A × A⁻¹ = A⁻¹ × A = I (identity matrix). A matrix has an inverse if and only if its determinant is non-zero.

```python
def matrix_inverse(A):
    """Calculate the inverse of a matrix using Gaussian elimination"""
    n = len(A)
    # Create the augmented matrix [A|I]
    augmented = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(A)]
    
    # Forward elimination
    for i in range(n):
        # Normalize the pivot row
        pivot = augmented[i][i]
        for j in range(i, 2 * n):
            augmented[i][j] /= pivot
        
        # Eliminate other rows
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(i, 2 * n):
                    augmented[k][j] -= factor * augmented[i][j]
    
    # Extract the inverse matrix
    inverse = [[augmented[i][j + n] for j in range(n)] for i in range(n)]
    return inverse
```

## Eigenvalues and Eigenvectors

An eigenvector of a square matrix A is a non-zero vector v such that Av = λv, where λ is a scalar known as the eigenvalue.

Computing eigenvalues and eigenvectors is typically done using iterative methods like the Power Method or QR Algorithm. Here's a simplified implementation of the Power Method:

```python
def power_method(A, iterations=100, epsilon=1e-10):
    """Find the dominant eigenvalue and eigenvector using the Power Method"""
    n = len(A)
    # Start with a random vector
    x = [1.0 for _ in range(n)]
    
    for _ in range(iterations):
        # Normalize the vector
        norm = math.sqrt(sum(x_i ** 2 for x_i in x))
        x = [x_i / norm for x_i in x]
        
        # Multiply by A
        x_new = [0.0 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                x_new[i] += A[i][j] * x[j]
        
        # Check for convergence
        if all(abs(x_new[i] - x[i] * (dot_product(x_new, x) / dot_product(x, x))) < epsilon for i in range(n)):
            break
        
        x = x_new
    
    # Calculate the eigenvalue
    eigenvalue = dot_product(matrix_multiply(A, [x]), x) / dot_product(x, x)
    
    return eigenvalue, x
```

## Applications in Computer Science

1. **Computer Graphics**: Transformations like translation, rotation, and scaling use matrices.
2. **Machine Learning**: Linear algebra is fundamental for algorithms like linear regression, PCA, SVD, and neural networks.
3. **Graph Algorithms**: The adjacency matrix representation of graphs enables various algorithms.
4. **Cryptography**: Matrix operations are used in some encryption methods.
5. **Optimization**: Many optimization algorithms rely on linear algebra concepts.

## Practice Problems

1. Implement matrix multiplication and verify its correctness
2. Solve a system of linear equations using Gaussian elimination
3. Compute the determinant of a matrix
4. Find the inverse of a matrix
5. Implement the power method to find eigenvalues

## Pro Tips

1. **Use Libraries**: In practice, use optimized libraries like NumPy, BLAS, LAPACK, or Eigen rather than implementing these operations from scratch.
2. **Sparse Matrices**: For large, sparse matrices (with many zeros), use specialized data structures and algorithms.
3. **Numerical Stability**: Be aware of numerical stability issues in operations like Gaussian elimination.
4. **Efficiency**: Matrix multiplication has more efficient algorithms than the naive O(n³) approach (e.g., Strassen's algorithm).
5. **Parallelization**: Many linear algebra operations can be parallelized for better performance.

## Related Topics

- [Vectors and Matrices](vectors-matrices.md)
- [Binary Exponentiation](binary-exponentiation.md)
- [Fast Fourier Transform](fft.md)
- [Systems of Linear Equations](systems-of-equations.md)
