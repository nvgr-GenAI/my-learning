# Multidimensional Arrays

## 🔍 Overview

Multidimensional arrays extend the concept of arrays to multiple dimensions, commonly used for matrices, images, scientific data, and gaming applications. They provide efficient storage and access patterns for structured data.

---

## 📊 Characteristics

### Key Properties

- **Multiple Dimensions**: 2D (matrices), 3D (cubes), or higher dimensions
- **Structured Access**: Elements accessed via multiple indices
- **Memory Layout**: Row-major or column-major storage
- **Efficient Iteration**: Optimized for specific access patterns
- **Mathematical Operations**: Support for linear algebra operations

### Visual Representation

```text
2D Array (Matrix):
[1, 2, 3]
[4, 5, 6]
[7, 8, 9]
Access: arr[row][col] or arr[i][j]

3D Array (Cube):
Layer 0: [1, 2]    Layer 1: [5, 6]
         [3, 4]             [7, 8]
Access: arr[layer][row][col] or arr[i][j][k]
```

---

## ⏱️ Time Complexities

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| **Access** | O(1) | Direct calculation of memory address |
| **Search** | O(n×m×...) | Linear search through all dimensions |
| **Insert Row/Col** | O(n×m×...) | May require shifting large amounts of data |
| **Delete Row/Col** | O(n×m×...) | Requires restructuring |
| **Transpose** | O(n×m) | For 2D arrays |
| **Matrix Multiply** | O(n³) | Standard algorithm for n×n matrices |

---

## 💻 Implementations

### 2D Arrays (Matrices)

```python
def matrix_operations():
    """Demonstrate 2D array operations."""
    
    # Create matrix using nested lists
    def create_matrix(rows, cols, initial_value=0):
        """Create matrix with given dimensions."""
        return [[initial_value for _ in range(cols)] for _ in range(rows)]
    
    # Initialize with specific values
    def create_identity_matrix(n):
        """Create n×n identity matrix."""
        matrix = create_matrix(n, n)
        for i in range(n):
            matrix[i][i] = 1
        return matrix
    
    def create_random_matrix(rows, cols):
        """Create matrix with random values."""
        import random
        return [[random.randint(1, 10) for _ in range(cols)] 
                for _ in range(rows)]
    
    # Matrix operations
    def print_matrix(matrix):
        """Pretty print matrix."""
        for row in matrix:
            print(' '.join(f'{x:3}' for x in row))
        print()
    
    def matrix_add(A, B):
        """Add two matrices."""
        rows, cols = len(A), len(A[0])
        if len(B) != rows or len(B[0]) != cols:
            raise ValueError("Matrices must have same dimensions")
        
        result = create_matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] + B[i][j]
        return result
    
    def matrix_multiply(A, B):
        """Multiply two matrices."""
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        
        if cols_A != rows_B:
            raise ValueError("Invalid dimensions for matrix multiplication")
        
        result = create_matrix(rows_A, cols_B)
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    
    def transpose(matrix):
        """Transpose a matrix."""
        rows, cols = len(matrix), len(matrix[0])
        result = create_matrix(cols, rows)
        for i in range(rows):
            for j in range(cols):
                result[j][i] = matrix[i][j]
        return result
    
    # Example usage
    print("Matrix Operations Demo:")
    
    # Create matrices
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    
    print("Matrix A:")
    print_matrix(A)
    
    print("Matrix B:")
    print_matrix(B)
    
    # Matrix multiplication
    C = matrix_multiply(A, B)
    print("A × B =")
    print_matrix(C)
    
    # Transpose
    A_T = transpose(A)
    print("A^T =")
    print_matrix(A_T)
    
    return create_matrix, matrix_add, matrix_multiply, transpose

def advanced_matrix_operations():
    """Demonstrate advanced matrix operations."""
    
    def determinant_2x2(matrix):
        """Calculate determinant of 2×2 matrix."""
        if len(matrix) != 2 or len(matrix[0]) != 2:
            raise ValueError("Matrix must be 2×2")
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    def determinant_recursive(matrix):
        """Calculate determinant using recursive expansion."""
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        if n == 2:
            return determinant_2x2(matrix)
        
        det = 0
        for j in range(n):
            # Create minor matrix
            minor = []
            for i in range(1, n):
                row = []
                for k in range(n):
                    if k != j:
                        row.append(matrix[i][k])
                minor.append(row)
            
            # Calculate cofactor
            cofactor = ((-1) ** j) * matrix[0][j] * determinant_recursive(minor)
            det += cofactor
        
        return det
    
    def matrix_power(matrix, n):
        """Calculate matrix raised to power n."""
        size = len(matrix)
        if n == 0:
            # Return identity matrix
            result = [[0] * size for _ in range(size)]
            for i in range(size):
                result[i][i] = 1
            return result
        
        if n == 1:
            return [row[:] for row in matrix]  # Deep copy
        
        if n % 2 == 0:
            half_power = matrix_power(matrix, n // 2)
            return matrix_multiply(half_power, half_power)
        else:
            return matrix_multiply(matrix, matrix_power(matrix, n - 1))
    
    def rotate_matrix_90(matrix):
        """Rotate matrix 90 degrees clockwise."""
        n = len(matrix)
        # Transpose then reverse each row
        transposed = transpose(matrix)
        for i in range(n):
            transposed[i].reverse()
        return transposed
    
    def spiral_traversal(matrix):
        """Traverse matrix in spiral order."""
        if not matrix:
            return []
        
        result = []
        rows, cols = len(matrix), len(matrix[0])
        top, bottom, left, right = 0, rows - 1, 0, cols - 1
        
        while top <= bottom and left <= right:
            # Traverse right
            for j in range(left, right + 1):
                result.append(matrix[top][j])
            top += 1
            
            # Traverse down
            for i in range(top, bottom + 1):
                result.append(matrix[i][right])
            right -= 1
            
            # Traverse left (if we have rows left)
            if top <= bottom:
                for j in range(right, left - 1, -1):
                    result.append(matrix[bottom][j])
                bottom -= 1
            
            # Traverse up (if we have columns left)
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    result.append(matrix[i][left])
                left += 1
        
        return result
    
    # Example usage
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print("Original matrix:")
    for row in matrix:
        print(row)
    
    print(f"Spiral traversal: {spiral_traversal(matrix)}")
    
    rotated = rotate_matrix_90(matrix)
    print("Rotated 90° clockwise:")
    for row in rotated:
        print(row)
    
    return determinant_recursive, matrix_power, rotate_matrix_90, spiral_traversal
```

### NumPy Multidimensional Arrays

```python
import numpy as np

def numpy_multidimensional():
    """Demonstrate NumPy multidimensional arrays."""
    
    # Create arrays of different dimensions
    def create_arrays():
        """Create various multidimensional arrays."""
        
        # 1D array
        arr_1d = np.array([1, 2, 3, 4, 5])
        print(f"1D array: {arr_1d}, shape: {arr_1d.shape}")
        
        # 2D array (matrix)
        arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
        print(f"2D array:\n{arr_2d}\nShape: {arr_2d.shape}")
        
        # 3D array
        arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        print(f"3D array:\n{arr_3d}\nShape: {arr_3d.shape}")
        
        # Create special arrays
        zeros = np.zeros((3, 3))
        ones = np.ones((2, 4))
        identity = np.eye(3)
        random_arr = np.random.rand(2, 3)
        
        print(f"Zeros matrix:\n{zeros}")
        print(f"Ones matrix:\n{ones}")
        print(f"Identity matrix:\n{identity}")
        print(f"Random matrix:\n{random_arr}")
        
        return arr_1d, arr_2d, arr_3d, zeros, ones, identity
    
    def array_operations():
        """Demonstrate array operations."""
        
        # Matrix operations
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        print("Matrix A:")
        print(A)
        print("Matrix B:")
        print(B)
        
        # Element-wise operations
        addition = A + B
        multiplication = A * B  # Element-wise
        
        print(f"A + B:\n{addition}")
        print(f"A * B (element-wise):\n{multiplication}")
        
        # Matrix multiplication
        matrix_mult = np.dot(A, B)  # or A @ B
        print(f"A @ B (matrix multiplication):\n{matrix_mult}")
        
        # Other operations
        transpose = A.T
        determinant = np.linalg.det(A)
        inverse = np.linalg.inv(A)
        
        print(f"A transpose:\n{transpose}")
        print(f"A determinant: {determinant}")
        print(f"A inverse:\n{inverse}")
        
        return A, B, addition, matrix_mult
    
    def advanced_operations():
        """Advanced NumPy operations."""
        
        # Broadcasting
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        scalar = 10
        broadcasted = arr + scalar  # Adds 10 to each element
        
        print(f"Original array:\n{arr}")
        print(f"After adding {scalar}:\n{broadcasted}")
        
        # Indexing and slicing
        arr_3d = np.random.randint(0, 10, (3, 4, 5))
        print(f"3D array shape: {arr_3d.shape}")
        print(f"First 2D slice:\n{arr_3d[0]}")
        print(f"Element at [1, 2, 3]: {arr_3d[1, 2, 3]}")
        print(f"Subarray [0:2, 1:3, :]:\n{arr_3d[0:2, 1:3, :]}")
        
        # Reshaping
        flat_arr = arr_3d.flatten()  # 1D
        reshaped = arr_3d.reshape(6, 10)  # 2D
        
        print(f"Flattened shape: {flat_arr.shape}")
        print(f"Reshaped to 6×10: {reshaped.shape}")
        
        # Statistical operations
        print(f"Mean: {np.mean(arr_3d)}")
        print(f"Standard deviation: {np.std(arr_3d)}")
        print(f"Max along axis 0: {np.max(arr_3d, axis=0).shape}")
        
        return arr_3d, flat_arr, reshaped
    
    return create_arrays(), array_operations(), advanced_operations()
```

### 3D Arrays and Higher Dimensions

```python
def higher_dimensional_arrays():
    """Work with 3D and higher dimensional arrays."""
    
    def create_3d_array():
        """Create and manipulate 3D arrays."""
        
        # Create 3D array (depth × height × width)
        arr_3d = [[[i * j * k for k in range(3)] 
                   for j in range(4)] 
                   for i in range(2)]
        
        print("3D Array structure:")
        for i, layer in enumerate(arr_3d):
            print(f"Layer {i}:")
            for row in layer:
                print(f"  {row}")
            print()
        
        return arr_3d
    
    def access_3d_elements(arr_3d):
        """Demonstrate 3D array access patterns."""
        
        depth, height, width = len(arr_3d), len(arr_3d[0]), len(arr_3d[0][0])
        
        print("3D Array Access Patterns:")
        
        # Access specific element
        element = arr_3d[1][2][1]
        print(f"Element at [1][2][1]: {element}")
        
        # Access entire layer
        layer = arr_3d[0]
        print(f"Layer 0: {layer}")
        
        # Access column across all layers
        column = [arr_3d[i][1][2] for i in range(depth)]
        print(f"Column [*][1][2]: {column}")
        
        # Traverse all elements
        print("All elements:")
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    print(f"[{i}][{j}][{k}] = {arr_3d[i][j][k]}")
        
        return element, layer, column
    
    def tensor_operations():
        """Demonstrate tensor (4D+) operations."""
        
        # Create 4D tensor (batch × channels × height × width)
        # Common in deep learning for image batches
        def create_image_batch(batch_size, channels, height, width):
            """Create a batch of images."""
            return [[[[i + j + k + l for l in range(width)]
                      for k in range(height)]
                     for j in range(channels)]
                    for i in range(batch_size)]
        
        # Create mini-batch of 2 RGB images of size 3×3
        batch = create_image_batch(2, 3, 3, 3)
        
        print(f"Tensor shape: {len(batch)} × {len(batch[0])} × "
              f"{len(batch[0][0])} × {len(batch[0][0][0])}")
        
        # Access first image, red channel
        first_image_red = batch[0][0]
        print(f"First image, red channel:\n{first_image_red}")
        
        # Process each image in batch
        def process_batch(batch):
            """Process each image in the batch."""
            processed = []
            for img_idx, image in enumerate(batch):
                print(f"Processing image {img_idx}")
                
                # Process each channel
                processed_image = []
                for ch_idx, channel in enumerate(image):
                    # Apply some operation (e.g., add 1 to each pixel)
                    processed_channel = [[pixel + 1 for pixel in row] 
                                       for row in channel]
                    processed_image.append(processed_channel)
                
                processed.append(processed_image)
            
            return processed
        
        processed_batch = process_batch(batch)
        
        return batch, processed_batch
    
    return create_3d_array(), tensor_operations()
```

---

## 🎯 Common Patterns

### 1. Matrix Traversal Patterns

```python
def matrix_traversal_patterns():
    """Demonstrate common matrix traversal patterns."""
    
    def create_test_matrix(n):
        """Create n×n test matrix."""
        return [[i * n + j for j in range(n)] for i in range(n)]
    
    def row_major_traversal(matrix):
        """Traverse matrix row by row."""
        result = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                result.append(matrix[i][j])
        return result
    
    def column_major_traversal(matrix):
        """Traverse matrix column by column."""
        result = []
        for j in range(len(matrix[0])):
            for i in range(len(matrix)):
                result.append(matrix[i][j])
        return result
    
    def diagonal_traversal(matrix):
        """Traverse matrix diagonally."""
        n = len(matrix)
        result = []
        
        # Main diagonal
        for i in range(n):
            result.append(matrix[i][i])
        
        # Anti-diagonal
        for i in range(n):
            result.append(matrix[i][n - 1 - i])
        
        return result
    
    def zigzag_traversal(matrix):
        """Traverse matrix in zigzag pattern."""
        result = []
        for i, row in enumerate(matrix):
            if i % 2 == 0:
                result.extend(row)  # Left to right
            else:
                result.extend(reversed(row))  # Right to left
        return result
    
    def boundary_traversal(matrix):
        """Traverse only the boundary of matrix."""
        if not matrix:
            return []
        
        result = []
        rows, cols = len(matrix), len(matrix[0])
        
        if rows == 1:
            return matrix[0]
        if cols == 1:
            return [matrix[i][0] for i in range(rows)]
        
        # Top row
        result.extend(matrix[0])
        
        # Right column (excluding corners)
        for i in range(1, rows - 1):
            result.append(matrix[i][cols - 1])
        
        # Bottom row (reversed)
        result.extend(reversed(matrix[rows - 1]))
        
        # Left column (excluding corners, bottom to top)
        for i in range(rows - 2, 0, -1):
            result.append(matrix[i][0])
        
        return result
    
    # Test all patterns
    matrix = create_test_matrix(4)
    print("Test matrix:")
    for row in matrix:
        print(row)
    
    print(f"Row major: {row_major_traversal(matrix)}")
    print(f"Column major: {column_major_traversal(matrix)}")
    print(f"Diagonal: {diagonal_traversal(matrix)}")
    print(f"Zigzag: {zigzag_traversal(matrix)}")
    print(f"Boundary: {boundary_traversal(matrix)}")
    
    return (row_major_traversal, column_major_traversal, 
            diagonal_traversal, zigzag_traversal, boundary_traversal)

def dynamic_programming_matrices():
    """Use matrices for dynamic programming problems."""
    
    def longest_common_subsequence(str1, str2):
        """Find LCS using 2D DP table."""
        m, n = len(str1), len(str2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # Reconstruct LCS
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if str1[i - 1] == str2[j - 1]:
                lcs.append(str1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        
        return dp[m][n], ''.join(reversed(lcs))
    
    def edit_distance(str1, str2):
        """Calculate minimum edit distance using 2D DP."""
        m, n = len(str1), len(str2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i  # Delete all characters
        for j in range(n + 1):
            dp[0][j] = j  # Insert all characters
        
        # Fill the table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # No operation needed
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Delete
                        dp[i][j - 1],      # Insert
                        dp[i - 1][j - 1]   # Replace
                    )
        
        return dp[m][n]
    
    def knapsack_2d(weights, values, capacity):
        """Solve 0/1 knapsack using 2D DP table."""
        n = len(weights)
        
        # Create DP table
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        # Fill the table
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                # Don't take item i
                dp[i][w] = dp[i - 1][w]
                
                # Take item i if possible
                if weights[i - 1] <= w:
                    dp[i][w] = max(dp[i][w], 
                                   dp[i - 1][w - weights[i - 1]] + values[i - 1])
        
        # Reconstruct solution
        selected = []
        i, w = n, capacity
        while i > 0 and w > 0:
            if dp[i][w] != dp[i - 1][w]:
                selected.append(i - 1)
                w -= weights[i - 1]
            i -= 1
        
        return dp[n][capacity], selected
    
    # Test examples
    print("LCS Example:")
    lcs_length, lcs_str = longest_common_subsequence("ABCDGH", "AEDFHR")
    print(f"LCS length: {lcs_length}, LCS: '{lcs_str}'")
    
    print("\nEdit Distance Example:")
    edit_dist = edit_distance("kitten", "sitting")
    print(f"Edit distance: {edit_dist}")
    
    print("\nKnapsack Example:")
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    max_value, items = knapsack_2d(weights, values, capacity)
    print(f"Max value: {max_value}, Items: {items}")
    
    return longest_common_subsequence, edit_distance, knapsack_2d
```

### 2. Image Processing Applications

```python
def image_processing_examples():
    """Demonstrate image processing using 2D arrays."""
    
    def create_sample_image(width, height):
        """Create a sample grayscale image."""
        import random
        return [[random.randint(0, 255) for _ in range(width)] 
                for _ in range(height)]
    
    def apply_filter(image, kernel):
        """Apply convolution filter to image."""
        height, width = len(image), len(image[0])
        k_height, k_width = len(kernel), len(kernel[0])
        
        # Calculate padding
        pad_h, pad_w = k_height // 2, k_width // 2
        
        # Create result image
        result = [[0] * width for _ in range(height)]
        
        # Apply convolution
        for i in range(height):
            for j in range(width):
                pixel_sum = 0
                
                for ki in range(k_height):
                    for kj in range(k_width):
                        # Calculate image coordinates
                        img_i = i + ki - pad_h
                        img_j = j + kj - pad_w
                        
                        # Handle boundaries (zero padding)
                        if 0 <= img_i < height and 0 <= img_j < width:
                            pixel_sum += image[img_i][img_j] * kernel[ki][kj]
                
                result[i][j] = max(0, min(255, int(pixel_sum)))
        
        return result
    
    def edge_detection(image):
        """Apply Sobel edge detection."""
        # Sobel kernels
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        
        # Apply both filters
        edges_x = apply_filter(image, sobel_x)
        edges_y = apply_filter(image, sobel_y)
        
        # Combine gradients
        height, width = len(image), len(image[0])
        result = [[0] * width for _ in range(height)]
        
        for i in range(height):
            for j in range(width):
                gradient_magnitude = (edges_x[i][j]**2 + edges_y[i][j]**2)**0.5
                result[i][j] = min(255, int(gradient_magnitude))
        
        return result
    
    def blur_image(image, kernel_size=3):
        """Apply Gaussian blur to image."""
        # Create Gaussian kernel
        kernel = [[1/(kernel_size**2) for _ in range(kernel_size)] 
                  for _ in range(kernel_size)]
        
        return apply_filter(image, kernel)
    
    def rotate_image_90(image):
        """Rotate image 90 degrees clockwise."""
        height, width = len(image), len(image[0])
        result = [[0] * height for _ in range(width)]
        
        for i in range(height):
            for j in range(width):
                result[j][height - 1 - i] = image[i][j]
        
        return result
    
    def histogram(image):
        """Calculate histogram of image."""
        hist = [0] * 256
        
        for row in image:
            for pixel in row:
                hist[pixel] += 1
        
        return hist
    
    # Example usage
    image = create_sample_image(8, 8)
    print("Original image (8×8):")
    for row in image:
        print(' '.join(f'{x:3}' for x in row))
    
    blurred = blur_image(image)
    print("\nBlurred image:")
    for row in blurred:
        print(' '.join(f'{x:3}' for x in row))
    
    edges = edge_detection(image)
    print("\nEdge detection:")
    for row in edges:
        print(' '.join(f'{x:3}' for x in row))
    
    return apply_filter, edge_detection, blur_image, rotate_image_90
```

---

## 🚀 Performance Optimization

### Memory Layout Optimization

```python
def memory_layout_optimization():
    """Demonstrate memory layout considerations."""
    
    import time
    
    def cache_friendly_access():
        """Demonstrate cache-friendly access patterns."""
        
        # Create large matrix
        size = 1000
        matrix = [[i * size + j for j in range(size)] for i in range(size)]
        
        def row_major_sum(matrix):
            """Sum using row-major order (cache-friendly)."""
            total = 0
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    total += matrix[i][j]
            return total
        
        def column_major_sum(matrix):
            """Sum using column-major order (cache-unfriendly)."""
            total = 0
            for j in range(len(matrix[0])):
                for i in range(len(matrix)):
                    total += matrix[i][j]
            return total
        
        # Time both approaches
        start = time.time()
        row_sum = row_major_sum(matrix)
        row_time = time.time() - start
        
        start = time.time()
        col_sum = column_major_sum(matrix)
        col_time = time.time() - start
        
        print(f"Row-major time: {row_time:.4f}s")
        print(f"Column-major time: {col_time:.4f}s")
        print(f"Speedup: {col_time/row_time:.2f}x")
        
        return row_time, col_time
    
    def memory_efficient_operations():
        """Demonstrate memory-efficient matrix operations."""
        
        def in_place_transpose(matrix):
            """Transpose square matrix in-place."""
            n = len(matrix)
            for i in range(n):
                for j in range(i + 1, n):
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
            return matrix
        
        def block_matrix_multiply(A, B, block_size=64):
            """Multiply matrices using block algorithm for better cache usage."""
            n = len(A)
            C = [[0] * n for _ in range(n)]
            
            for i in range(0, n, block_size):
                for j in range(0, n, block_size):
                    for k in range(0, n, block_size):
                        # Multiply blocks
                        for ii in range(i, min(i + block_size, n)):
                            for jj in range(j, min(j + block_size, n)):
                                for kk in range(k, min(k + block_size, n)):
                                    C[ii][jj] += A[ii][kk] * B[kk][jj]
            
            return C
        
        def sparse_matrix_operations():
            """Handle sparse matrices efficiently."""
            
            # Coordinate list format (COO)
            class SparseMatrix:
                def __init__(self, rows, cols):
                    self.rows = rows
                    self.cols = cols
                    self.data = []  # List of (row, col, value) tuples
                
                def set(self, row, col, value):
                    if value != 0:
                        self.data.append((row, col, value))
                
                def get(self, row, col):
                    for r, c, v in self.data:
                        if r == row and c == col:
                            return v
                    return 0
                
                def multiply_vector(self, vector):
                    """Multiply sparse matrix by vector."""
                    if len(vector) != self.cols:
                        raise ValueError("Vector size mismatch")
                    
                    result = [0] * self.rows
                    for row, col, value in self.data:
                        result[row] += value * vector[col]
                    
                    return result
                
                def to_dense(self):
                    """Convert to dense matrix."""
                    dense = [[0] * self.cols for _ in range(self.rows)]
                    for row, col, value in self.data:
                        dense[row][col] = value
                    return dense
            
            # Create sparse matrix (mostly zeros)
            sparse = SparseMatrix(5, 5)
            sparse.set(0, 0, 1)
            sparse.set(1, 2, 3)
            sparse.set(3, 4, 7)
            sparse.set(4, 1, 2)
            
            print("Sparse matrix:")
            dense = sparse.to_dense()
            for row in dense:
                print(row)
            
            # Multiply by vector
            vector = [1, 2, 3, 4, 5]
            result = sparse.multiply_vector(vector)
            print(f"Matrix × vector: {result}")
            
            return sparse
        
        return in_place_transpose, block_matrix_multiply, sparse_matrix_operations()
    
    return cache_friendly_access(), memory_efficient_operations()
```

---

## 🎯 When to Use Multidimensional Arrays

### ✅ Best Use Cases

1. **Mathematical Computations**: Linear algebra, scientific computing
2. **Image Processing**: 2D/3D image manipulation and computer vision
3. **Game Development**: 2D grids, 3D worlds, game boards
4. **Data Analysis**: Matrices, tensors, multi-dimensional datasets
5. **Machine Learning**: Feature matrices, neural network weights
6. **Simulation**: Physical simulations, cellular automata

### ❌ Limitations

1. **Memory Usage**: Can consume large amounts of memory
2. **Cache Performance**: Poor cache locality for certain access patterns
3. **Complexity**: More complex indexing and iteration
4. **Fixed Structure**: Difficult to resize efficiently
5. **Sparse Data**: Inefficient for mostly empty data

### Performance Considerations

| Dimension | Memory Usage | Access Speed | Use Case |
|-----------|--------------|--------------|----------|
| 2D | O(n²) | Fast for row-major | Matrices, images |
| 3D | O(n³) | Moderate | Volumes, videos |
| 4D+ | O(n⁴⁺) | Slower | Deep learning, simulations |

---

## 🔗 Related Topics

- **[Static Arrays](static-arrays.md)**: Foundation for multidimensional arrays
- **[Dynamic Arrays](dynamic-arrays.md)**: For resizable collections
- **[Matrix Algorithms](../../algorithms/matrix/index.md)**: Advanced matrix operations
- **[Image Processing](../../../computer-vision/image-processing.md)**: Real-world applications
- **[Linear Algebra](../../math/linear-algebra.md)**: Mathematical foundations

---

*Ready to practice with array problems? Start with [Easy Problems](easy-problems.md)!*
