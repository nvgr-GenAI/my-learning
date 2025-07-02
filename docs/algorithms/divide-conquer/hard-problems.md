# Divide and Conquer - Hard Problems

## Problem Categories

### 1. Advanced Mathematical Algorithms
- Fast Fourier Transform
- Karatsuba multiplication
- Matrix operations

### 2. Complex Geometric Problems
- Convex hull algorithms
- Voronoi diagrams
- Line segment intersection

### 3. Advanced Data Structure Problems
- Persistent data structures
- Range queries
- Multi-dimensional problems

---

## 1. Fast Fourier Transform (FFT)

**Problem**: Compute the Discrete Fourier Transform of a sequence efficiently.

**Application**: Polynomial multiplication, signal processing, image compression.

**Solution**:
```python
import cmath
import math

def fft(coefficients):
    """
    Fast Fourier Transform using divide and conquer.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n log n) - recursion stack
    """
    n = len(coefficients)
    
    # Base case
    if n <= 1:
        return coefficients
    
    # Ensure n is power of 2 (pad with zeros if necessary)
    if n & (n - 1) != 0:
        next_power = 1 << (n - 1).bit_length()
        coefficients.extend([0] * (next_power - n))
        n = next_power
    
    # Divide: separate even and odd indexed elements
    even = [coefficients[i] for i in range(0, n, 2)]
    odd = [coefficients[i] for i in range(1, n, 2)]
    
    # Conquer: recursively compute FFT of even and odd parts
    fft_even = fft(even)
    fft_odd = fft(odd)
    
    # Combine: merge the results
    result = [0] * n
    for i in range(n // 2):
        # Calculate twiddle factor
        omega = cmath.exp(-2j * cmath.pi * i / n)
        
        # Combine even and odd parts
        result[i] = fft_even[i] + omega * fft_odd[i]
        result[i + n // 2] = fft_even[i] - omega * fft_odd[i]
    
    return result

def ifft(coefficients):
    """
    Inverse Fast Fourier Transform.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n log n)
    """
    n = len(coefficients)
    
    # Conjugate the complex numbers
    conjugated = [complex(c.real, -c.imag) for c in coefficients]
    
    # Apply FFT
    result = fft(conjugated)
    
    # Conjugate again and divide by n
    return [complex(c.real / n, -c.imag / n) for c in result]

def polynomial_multiply(poly1, poly2):
    """
    Multiply two polynomials using FFT.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    # Determine result size
    result_size = len(poly1) + len(poly2) - 1
    
    # Pad to next power of 2
    fft_size = 1
    while fft_size < result_size:
        fft_size <<= 1
    
    # Pad polynomials
    poly1_padded = poly1 + [0] * (fft_size - len(poly1))
    poly2_padded = poly2 + [0] * (fft_size - len(poly2))
    
    # Convert to frequency domain
    fft1 = fft(poly1_padded)
    fft2 = fft(poly2_padded)
    
    # Pointwise multiplication in frequency domain
    product_fft = [fft1[i] * fft2[i] for i in range(fft_size)]
    
    # Convert back to time domain
    product = ifft(product_fft)
    
    # Extract real parts and trim to actual result size
    result = [int(round(c.real)) for c in product[:result_size]]
    return result

# Test
poly1 = [1, 2, 3]  # represents 1 + 2x + 3x^2
poly2 = [4, 5]     # represents 4 + 5x
print(polynomial_multiply(poly1, poly2))
# Output: [4, 13, 22, 15] representing 4 + 13x + 22x^2 + 15x^3
```

**Key Points**:
- Reduces polynomial multiplication from O(n²) to O(n log n)
- Uses complex roots of unity as evaluation points
- Divide step separates even and odd coefficients
- Combine step uses twiddle factors to merge results

---

## 2. Karatsuba Multiplication

**Problem**: Multiply two large integers efficiently.

**Example**:
```
Input: x = 1234, y = 5678
Output: 7006652
```

**Solution**:
```python
def karatsuba_multiply(x, y):
    """
    Karatsuba algorithm for fast integer multiplication.
    
    Time Complexity: O(n^1.585) where n is number of digits
    Space Complexity: O(log n) - recursion depth
    """
    # Convert to strings for easier digit manipulation
    str_x, str_y = str(x), str(y)
    
    def karatsuba_helper(x_str, y_str):
        # Base case: single digit multiplication
        if len(x_str) == 1 or len(y_str) == 1:
            return int(x_str) * int(y_str)
        
        # Make both strings same length by padding with zeros
        max_len = max(len(x_str), len(y_str))
        x_str = x_str.zfill(max_len)
        y_str = y_str.zfill(max_len)
        
        # If length is odd, make it even
        if max_len % 2 == 1:
            max_len += 1
            x_str = x_str.zfill(max_len)
            y_str = y_str.zfill(max_len)
        
        # Divide: split into high and low parts
        mid = max_len // 2
        x_high, x_low = x_str[:mid], x_str[mid:]
        y_high, y_low = y_str[:mid], y_str[mid:]
        
        # Conquer: three recursive multiplications
        z0 = karatsuba_helper(x_low, y_low)    # low * low
        z2 = karatsuba_helper(x_high, y_high)  # high * high
        
        # Calculate (x_high + x_low) * (y_high + y_low)
        x_sum = str(int(x_high) + int(x_low))
        y_sum = str(int(y_high) + int(y_low))
        z1_full = karatsuba_helper(x_sum, y_sum)
        
        # z1 = z1_full - z2 - z0
        z1 = z1_full - z2 - z0
        
        # Combine: result = z2 * 10^(2*mid) + z1 * 10^mid + z0
        return z2 * (10 ** (2 * mid)) + z1 * (10 ** mid) + z0
    
    return karatsuba_helper(str_x, str_y)

# Optimized version for very large numbers
def karatsuba_large_numbers(x, y):
    """
    Optimized Karatsuba for very large numbers using bit operations.
    """
    def bit_length(n):
        return n.bit_length()
    
    def karatsuba_rec(x, y):
        # Base case threshold
        if x < 10 or y < 10:
            return x * y
        
        # Find the size of the numbers
        max_bits = max(bit_length(x), bit_length(y))
        half_bits = max_bits // 2
        
        # Split the numbers
        shift = 1 << half_bits
        x_high, x_low = divmod(x, shift)
        y_high, y_low = divmod(y, shift)
        
        # Three multiplications
        z0 = karatsuba_rec(x_low, y_low)
        z2 = karatsuba_rec(x_high, y_high)
        z1 = karatsuba_rec(x_high + x_low, y_high + y_low) - z2 - z0
        
        # Combine results
        return (z2 << (2 * half_bits)) + (z1 << half_bits) + z0
    
    return karatsuba_rec(x, y)

# Test
print(karatsuba_multiply(1234, 5678))  # Output: 7006652
```

**Key Points**:
- Reduces multiplication complexity from O(n²) to O(n^1.585)
- Uses three multiplications instead of four: (a+b)(c+d) = ac + bd + (a+b)(c+d) - ac - bd
- Recursive structure naturally fits divide and conquer paradigm

---

## 3. Strassen's Matrix Multiplication

**Problem**: Multiply two n×n matrices efficiently.

**Solution**:
```python
def strassen_multiply(A, B):
    """
    Strassen's algorithm for matrix multiplication.
    
    Time Complexity: O(n^2.807)
    Space Complexity: O(n^2)
    """
    n = len(A)
    
    # Base case: small matrices use standard multiplication
    if n <= 2:
        return standard_multiply(A, B)
    
    # Ensure matrices are of even dimension
    if n % 2 != 0:
        A = pad_matrix(A)
        B = pad_matrix(B)
        n += 1
    
    # Divide matrices into quadrants
    mid = n // 2
    
    A11 = [[A[i][j] for j in range(mid)] for i in range(mid)]
    A12 = [[A[i][j] for j in range(mid, n)] for i in range(mid)]
    A21 = [[A[i][j] for j in range(mid)] for i in range(mid, n)]
    A22 = [[A[i][j] for j in range(mid, n)] for i in range(mid, n)]
    
    B11 = [[B[i][j] for j in range(mid)] for i in range(mid)]
    B12 = [[B[i][j] for j in range(mid, n)] for i in range(mid)]
    B21 = [[B[i][j] for j in range(mid)] for i in range(mid, n)]
    B22 = [[B[i][j] for j in range(mid, n)] for i in range(mid, n)]
    
    # Compute the 7 Strassen products
    M1 = strassen_multiply(matrix_add(A11, A22), matrix_add(B11, B22))
    M2 = strassen_multiply(matrix_add(A21, A22), B11)
    M3 = strassen_multiply(A11, matrix_subtract(B12, B22))
    M4 = strassen_multiply(A22, matrix_subtract(B21, B11))
    M5 = strassen_multiply(matrix_add(A11, A12), B22)
    M6 = strassen_multiply(matrix_subtract(A21, A11), matrix_add(B11, B12))
    M7 = strassen_multiply(matrix_subtract(A12, A22), matrix_add(B21, B22))
    
    # Compute result quadrants
    C11 = matrix_add(matrix_subtract(matrix_add(M1, M4), M5), M7)
    C12 = matrix_add(M3, M5)
    C21 = matrix_add(M2, M4)
    C22 = matrix_add(matrix_subtract(matrix_add(M1, M3), M2), M6)
    
    # Combine quadrants into result matrix
    C = [[0] * n for _ in range(n)]
    for i in range(mid):
        for j in range(mid):
            C[i][j] = C11[i][j]
            C[i][j + mid] = C12[i][j]
            C[i + mid][j] = C21[i][j]
            C[i + mid][j + mid] = C22[i][j]
    
    return C

def matrix_add(A, B):
    """Add two matrices."""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def matrix_subtract(A, B):
    """Subtract two matrices."""
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def standard_multiply(A, B):
    """Standard O(n³) matrix multiplication."""
    n = len(A)
    C = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

def pad_matrix(matrix):
    """Pad matrix to make it even-sized."""
    n = len(matrix)
    padded = [[0] * (n + 1) for _ in range(n + 1)]
    
    for i in range(n):
        for j in range(n):
            padded[i][j] = matrix[i][j]
    
    return padded

# Cache-friendly version for practical use
def strassen_multiply_optimized(A, B, threshold=64):
    """
    Optimized Strassen with threshold for switching to standard multiplication.
    """
    n = len(A)
    
    # Use standard multiplication for small matrices
    if n <= threshold:
        return standard_multiply(A, B)
    
    # Continue with Strassen's algorithm...
    return strassen_multiply(A, B)
```

**Key Points**:
- Uses 7 multiplications instead of 8 for matrix quadrants
- Asymptotically faster than standard O(n³) algorithm
- Practical threshold needed due to constant factor overhead
- Cache-friendly optimizations important for real-world performance

---

## 4. Convex Hull (Graham Scan with D&C optimization)

**Problem**: Find the convex hull of a set of 2D points.

**Solution**:
```python
import math

def convex_hull_divide_conquer(points):
    """
    Convex hull using divide and conquer approach.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def merge_hulls(left_hull, right_hull):
        """Merge two convex hulls."""
        # Find rightmost point of left hull and leftmost point of right hull
        left_most_right = max(range(len(left_hull)), key=lambda i: left_hull[i][0])
        right_most_left = min(range(len(right_hull)), key=lambda i: right_hull[i][0])
        
        # Find upper tangent
        upper_left, upper_right = left_most_right, right_most_left
        
        while True:
            changed = False
            
            # Check if we can move clockwise on left hull
            next_left = (upper_left + 1) % len(left_hull)
            if cross_product(right_hull[upper_right], left_hull[upper_left], left_hull[next_left]) < 0:
                upper_left = next_left
                changed = True
            
            # Check if we can move counter-clockwise on right hull
            prev_right = (upper_right - 1) % len(right_hull)
            if cross_product(left_hull[upper_left], right_hull[upper_right], right_hull[prev_right]) > 0:
                upper_right = prev_right
                changed = True
            
            if not changed:
                break
        
        # Find lower tangent
        lower_left, lower_right = left_most_right, right_most_left
        
        while True:
            changed = False
            
            # Check if we can move counter-clockwise on left hull
            prev_left = (lower_left - 1) % len(left_hull)
            if cross_product(right_hull[lower_right], left_hull[lower_left], left_hull[prev_left]) > 0:
                lower_left = prev_left
                changed = True
            
            # Check if we can move clockwise on right hull
            next_right = (lower_right + 1) % len(right_hull)
            if cross_product(left_hull[lower_left], right_hull[lower_right], right_hull[next_right]) < 0:
                lower_right = next_right
                changed = True
            
            if not changed:
                break
        
        # Combine hulls
        result = []
        
        # Add points from left hull (upper to lower)
        i = upper_left
        while True:
            result.append(left_hull[i])
            if i == lower_left:
                break
            i = (i + 1) % len(left_hull)
        
        # Add points from right hull (lower to upper)
        i = lower_right
        while True:
            result.append(right_hull[i])
            if i == upper_right:
                break
            i = (i + 1) % len(right_hull)
        
        return result
    
    def hull_recursive(points):
        n = len(points)
        
        # Base case: small number of points
        if n <= 3:
            if n < 3:
                return points
            
            # Check if points are collinear
            if cross_product(points[0], points[1], points[2]) == 0:
                return [min(points), max(points)]
            
            # Order points counter-clockwise
            if cross_product(points[0], points[1], points[2]) < 0:
                return [points[0], points[2], points[1]]
            else:
                return points
        
        # Divide
        mid = n // 2
        left_hull = hull_recursive(points[:mid])
        right_hull = hull_recursive(points[mid:])
        
        # Conquer
        return merge_hulls(left_hull, right_hull)
    
    if len(points) < 3:
        return points
    
    # Sort points by x-coordinate
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    
    return hull_recursive(sorted_points)

# More efficient Graham scan for comparison
def graham_scan(points):
    """
    Graham scan algorithm for convex hull.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Find the bottom-most point (and left-most in case of tie)
    start = min(points, key=lambda p: (p[1], p[0]))
    
    # Sort points by polar angle with respect to start point
    def polar_angle_key(p):
        dx, dy = p[0] - start[0], p[1] - start[1]
        return math.atan2(dy, dx)
    
    sorted_points = sorted([p for p in points if p != start], key=polar_angle_key)
    
    # Build convex hull
    hull = [start]
    
    for point in sorted_points:
        # Remove points that create clockwise turn
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    
    return hull
```

**Key Points**:
- Divide points by x-coordinate and recursively find hulls
- Merge step finds upper and lower tangents between hulls
- Graham scan is typically more efficient in practice
- Complex merge operation makes this more of a theoretical interest

---

## 5. Median of Two Sorted Arrays

**Problem**: Find the median of two sorted arrays in O(log(min(m,n))) time.

**Solution**:
```python
def find_median_sorted_arrays(nums1, nums2):
    """
    Find median of two sorted arrays using binary search.
    
    Time Complexity: O(log(min(m, n)))
    Space Complexity: O(1)
    """
    # Ensure nums1 is the smaller array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    
    def find_kth_element(k):
        """Find the kth smallest element in merged arrays."""
        if m == 0:
            return nums2[k - 1]
        if n == 0:
            return nums1[k - 1]
        if k == 1:
            return min(nums1[0], nums2[0])
        
        # Binary search on the smaller array
        left, right = 0, m
        
        while left <= right:
            partition1 = (left + right) // 2
            partition2 = k - partition1
            
            # Handle edge cases
            if partition2 < 0:
                right = partition1 - 1
                continue
            if partition2 > n:
                left = partition1 + 1
                continue
            
            # Get elements around partitions
            max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
            max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
            
            min_right1 = float('inf') if partition1 == m else nums1[partition1]
            min_right2 = float('inf') if partition2 == n else nums2[partition2]
            
            # Check if we found the correct partition
            if max_left1 <= min_right2 and max_left2 <= min_right1:
                return max(max_left1, max_left2)
            elif max_left1 > min_right2:
                right = partition1 - 1
            else:
                left = partition1 + 1
        
        raise ValueError("Input arrays are not sorted")
    
    total_length = m + n
    
    if total_length % 2 == 1:
        # Odd total length
        return float(find_kth_element(total_length // 2 + 1))
    else:
        # Even total length
        median1 = find_kth_element(total_length // 2)
        median2 = find_kth_element(total_length // 2 + 1)
        return (median1 + median2) / 2.0

# Alternative approach with cleaner partition logic
def find_median_sorted_arrays_v2(nums1, nums2):
    """
    Alternative implementation with cleaner logic.
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    total = m + n
    half = total // 2
    
    left, right = 0, m
    
    while left <= right:
        i = (left + right) // 2  # partition for nums1
        j = half - i             # partition for nums2
        
        # Elements on left side of partition
        nums1_left = float('-inf') if i == 0 else nums1[i - 1]
        nums2_left = float('-inf') if j == 0 else nums2[j - 1]
        
        # Elements on right side of partition
        nums1_right = float('inf') if i == m else nums1[i]
        nums2_right = float('inf') if j == n else nums2[j]
        
        # Check if partition is correct
        if nums1_left <= nums2_right and nums2_left <= nums1_right:
            # Found correct partition
            if total % 2 == 1:
                return min(nums1_right, nums2_right)
            else:
                return (max(nums1_left, nums2_left) + min(nums1_right, nums2_right)) / 2
        elif nums1_left > nums2_right:
            right = i - 1
        else:
            left = i + 1
    
    raise ValueError("Input arrays are not sorted")

# Test
print(find_median_sorted_arrays([1, 3], [2]))        # Output: 2.0
print(find_median_sorted_arrays([1, 2], [3, 4]))    # Output: 2.5
```

**Key Points**:
- Use binary search on the smaller array for efficiency
- Partition both arrays such that left partition has half the total elements
- Ensure all elements in left partition ≤ all elements in right partition
- Handle odd/even total length cases for median calculation

These hard problems showcase the most sophisticated applications of divide and conquer, involving complex mathematical algorithms, advanced geometric computations, and highly optimized solutions that push the boundaries of algorithmic efficiency.
