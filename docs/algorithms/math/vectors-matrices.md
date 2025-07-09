# Vectors and Matrices

## 🎯 Overview

Vectors and matrices are fundamental mathematical structures used extensively in computer graphics, physics simulations, machine learning, and many other computational domains. This document covers the essential operations, algorithms, and implementations for working with vectors and matrices in computational contexts.

## 📋 Core Concepts

### Vectors

A vector represents a quantity with both magnitude and direction. In computational contexts, vectors are typically represented as arrays of numbers.

#### Vector Properties

- **Dimension**: The number of components in the vector
- **Magnitude (Norm)**: The length or size of the vector
- **Direction**: The orientation of the vector in space

#### Basic Vector Operations

- **Addition**: Component-wise addition of vectors
- **Scalar Multiplication**: Multiplying all components by a scalar
- **Dot Product**: Sum of component-wise products, resulting in a scalar
- **Cross Product**: A vector perpendicular to two input vectors (3D only)
- **Normalization**: Converting a vector to a unit vector (magnitude = 1)

### Matrices

A matrix is a rectangular array of numbers arranged in rows and columns. Matrices are used to represent linear transformations, systems of linear equations, and data sets.

#### Matrix Properties

- **Dimensions**: The number of rows and columns (m × n)
- **Square Matrix**: A matrix with the same number of rows and columns
- **Identity Matrix**: A square matrix with 1s on the diagonal and 0s elsewhere
- **Transpose**: A matrix obtained by swapping rows and columns
- **Determinant**: A scalar value computed from a square matrix
- **Inverse**: A matrix that, when multiplied with the original, yields the identity matrix

#### Basic Matrix Operations

- **Addition**: Component-wise addition
- **Scalar Multiplication**: Multiply all elements by a scalar
- **Matrix Multiplication**: Multiply rows of first matrix with columns of second
- **Transpose**: Swap rows and columns
- **Determinant Calculation**: Compute the determinant of a square matrix
- **Matrix Inversion**: Find the inverse of a matrix (if it exists)

## ⚙️ Algorithm Implementations

### Vector Operations

```python
import math
import numpy as np  # For more efficient implementations

class Vector:
    def __init__(self, components):
        """Initialize a vector with given components"""
        self.components = list(components)
        self.dimension = len(components)
    
    def __str__(self):
        """String representation of the vector"""
        return str(self.components)
    
    def __add__(self, other):
        """Vector addition"""
        if self.dimension != other.dimension:
            raise ValueError("Vector dimensions must match")
        return Vector([a + b for a, b in zip(self.components, other.components)])
    
    def __sub__(self, other):
        """Vector subtraction"""
        if self.dimension != other.dimension:
            raise ValueError("Vector dimensions must match")
        return Vector([a - b for a, b in zip(self.components, other.components)])
    
    def __mul__(self, scalar):
        """Scalar multiplication"""
        return Vector([scalar * a for a in self.components])
    
    def dot(self, other):
        """Dot product"""
        if self.dimension != other.dimension:
            raise ValueError("Vector dimensions must match")
        return sum(a * b for a, b in zip(self.components, other.components))
    
    def cross(self, other):
        """Cross product (for 3D vectors only)"""
        if self.dimension != 3 or other.dimension != 3:
            raise ValueError("Cross product is defined only for 3D vectors")
        
        a = self.components
        b = other.components
        
        return Vector([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ])
    
    def magnitude(self):
        """Calculate vector magnitude (norm)"""
        return math.sqrt(sum(a * a for a in self.components))
    
    def normalize(self):
        """Return a normalized unit vector"""
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return Vector([a / mag for a in self.components])
    
    def angle_with(self, other):
        """Calculate the angle between two vectors in radians"""
        dot_product = self.dot(other)
        mag_product = self.magnitude() * other.magnitude()
        
        # Handle floating point precision issues
        cosine = min(1.0, max(-1.0, dot_product / mag_product))
        return math.acos(cosine)
```

### Matrix Operations

```python
class Matrix:
    def __init__(self, rows):
        """Initialize a matrix with given rows"""
        self.rows = [list(row) for row in rows]
        self.m = len(rows)  # Number of rows
        self.n = len(rows[0]) if self.m > 0 else 0  # Number of columns
        
        # Ensure all rows have the same length
        if any(len(row) != self.n for row in self.rows):
            raise ValueError("All rows must have the same length")
    
    def __str__(self):
        """String representation of the matrix"""
        return '\n'.join(' '.join(str(x) for x in row) for row in self.rows)
    
    def __add__(self, other):
        """Matrix addition"""
        if self.m != other.m or self.n != other.n:
            raise ValueError("Matrix dimensions must match for addition")
        
        result = []
        for i in range(self.m):
            result.append([a + b for a, b in zip(self.rows[i], other.rows[i])])
        
        return Matrix(result)
    
    def __mul__(self, scalar):
        """Scalar multiplication"""
        result = []
        for i in range(self.m):
            result.append([scalar * a for a in self.rows[i]])
        
        return Matrix(result)
    
    def multiply(self, other):
        """Matrix multiplication"""
        if self.n != other.m:
            raise ValueError("Number of columns in first matrix must equal number of rows in second matrix")
        
        result = []
        for i in range(self.m):
            row = []
            for j in range(other.n):
                element = sum(self.rows[i][k] * other.rows[k][j] for k in range(self.n))
                row.append(element)
            result.append(row)
        
        return Matrix(result)
    
    def transpose(self):
        """Compute the transpose of the matrix"""
        if self.m == 0 or self.n == 0:
            return Matrix([])
        
        result = []
        for j in range(self.n):
            result.append([self.rows[i][j] for i in range(self.m)])
        
        return Matrix(result)
    
    def determinant(self):
        """Calculate the determinant of a square matrix"""
        if self.m != self.n:
            raise ValueError("Determinant is defined only for square matrices")
        
        if self.m == 1:
            return self.rows[0][0]
        
        if self.m == 2:
            return self.rows[0][0] * self.rows[1][1] - self.rows[0][1] * self.rows[1][0]
        
        # For larger matrices, use cofactor expansion
        det = 0
        for j in range(self.n):
            det += ((-1) ** j) * self.rows[0][j] * self.minor(0, j).determinant()
        
        return det
    
    def minor(self, row, col):
        """Return the minor matrix by removing specified row and column"""
        result = []
        for i in range(self.m):
            if i != row:
                result.append([self.rows[i][j] for j in range(self.n) if j != col])
        
        return Matrix(result)
    
    def inverse(self):
        """Compute the inverse of a square matrix"""
        if self.m != self.n:
            raise ValueError("Inverse is defined only for square matrices")
        
        det = self.determinant()
        if abs(det) < 1e-10:  # Check if determinant is too close to zero
            raise ValueError("Matrix is singular, inverse does not exist")
        
        # For 2x2 matrix, we can use a simple formula
        if self.m == 2:
            a, b = self.rows[0][0], self.rows[0][1]
            c, d = self.rows[1][0], self.rows[1][1]
            return Matrix([[d/det, -b/det], [-c/det, a/det]])
        
        # For larger matrices, use adjugate matrix
        adjugate = []
        for i in range(self.m):
            row = []
            for j in range(self.n):
                # Cofactor is (-1)^(i+j) * determinant of minor
                cofactor = ((-1) ** (i + j)) * self.minor(i, j).determinant()
                row.append(cofactor)
            adjugate.append(row)
        
        # Transpose the cofactor matrix to get the adjugate
        adjugate = Matrix(adjugate).transpose()
        
        # Multiply adjugate by 1/det
        return adjugate * (1.0 / det)
```

### Efficient Implementations using NumPy

```python
import numpy as np

def efficient_matrix_operations():
    # Create matrices
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    
    # Basic operations
    addition = A + B
    subtraction = A - B
    scalar_mul = 2 * A
    
    # Matrix multiplication
    matrix_product = np.matmul(A, B)  # or A @ B in Python 3.5+
    
    # Transpose
    A_transpose = A.T
    
    # Inverse (if exists)
    try:
        A_inverse = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("Matrix is not invertible")
    
    # Determinant
    det_A = np.linalg.det(A)
    
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Singular Value Decomposition
    U, S, V = np.linalg.svd(A)
    
    return {
        "addition": addition,
        "subtraction": subtraction,
        "scalar_multiplication": scalar_mul,
        "matrix_product": matrix_product,
        "transpose": A_transpose,
        "determinant": det_A,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "svd": (U, S, V)
    }
```

## 🔍 Advanced Matrix Algorithms

### Gaussian Elimination

```python
def gaussian_elimination(A, b):
    """
    Solve a system of linear equations Ax = b using Gaussian elimination
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        
    Returns:
        Solution vector x
    """
    n = len(A)
    # Augment A with b
    augmented = [A[i][:] + [b[i]] for i in range(n)]
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_element = abs(augmented[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > max_element:
                max_element = abs(augmented[k][i])
                max_row = k
        
        # Swap rows
        augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
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

### LU Decomposition

```python
def lu_decomposition(A):
    """
    Decompose matrix A into lower and upper triangular matrices
    
    Args:
        A: Input matrix (must be square)
        
    Returns:
        L: Lower triangular matrix
        U: Upper triangular matrix
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        # Upper triangular matrix
        for k in range(i, n):
            sum_val = 0
            for j in range(i):
                sum_val += L[i][j] * U[j][k]
            U[i][k] = A[i][k] - sum_val
        
        # Lower triangular matrix
        L[i][i] = 1  # Diagonal elements of L are 1
        for k in range(i + 1, n):
            sum_val = 0
            for j in range(i):
                sum_val += L[k][j] * U[j][i]
            L[k][i] = (A[k][i] - sum_val) / U[i][i]
    
    return L, U
```

## ⚙️ Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Vector Addition/Subtraction | O(n) | O(n) |
| Vector Dot Product | O(n) | O(1) |
| Vector Cross Product (3D) | O(1) | O(1) |
| Matrix Addition | O(m*n) | O(m*n) |
| Matrix Multiplication | O(n³) | O(m*p) |
| Matrix Determinant | O(n³) | O(n²) |
| Matrix Inversion | O(n³) | O(n²) |
| Gaussian Elimination | O(n³) | O(n²) |
| LU Decomposition | O(n³) | O(n²) |

Where n is the dimension of vectors or the size of square matrices, and m×n and n×p are the dimensions of matrices being multiplied.

## 🧩 Applications

1. **Computer Graphics**: Transformations, projections, camera positioning
2. **Physics Simulations**: Forces, velocities, accelerations
3. **Machine Learning**: Linear regression, neural networks
4. **Robotics**: Kinematics, motion planning
5. **Signal Processing**: Filters, transformations
6. **Optimization Problems**: Systems of linear equations
7. **Data Analysis**: Principal Component Analysis, dimensionality reduction

## 📝 Practice Problems

1. **Matrix Transformations**: Implement rotations, scaling, and translations
2. **Linear System Solving**: Solve systems of linear equations
3. **Eigen Problems**: Find eigenvalues and eigenvectors
4. **Orthogonalization**: Implement Gram-Schmidt orthogonalization
5. **Matrix Factorization**: Implement QR, SVD, or Cholesky decomposition

## 🌟 Pro Tips

- Use specialized libraries like NumPy or BLAS for performance-critical applications
- Be aware of numerical stability issues in matrix operations
- For large matrices, consider sparse representations when appropriate
- Avoid explicit matrix inversion when solving linear systems
- Cache results of expensive operations when possible
- For 3D graphics, consider using quaternions for rotations to avoid gimbal lock
- Use vectorized operations when available for better performance

## 🔗 Related Algorithms

- [Linear Algebra Basics](linear-algebra.md)
- [Convex Hull](convex-hull.md)
- [Line Intersection](line-intersection.md)
- [Polygon Area](polygon-area.md)
- [Numerical Methods](../numerical-methods/index.md)
