# Array Searching

Searching is one of the most fundamental operations performed on arrays. It involves finding a specific element or determining if an element with certain properties exists within the array.

## Basic Search Algorithms for Arrays

### Linear Search

Linear search is the simplest searching algorithm that checks each element of the array sequentially until a match is found or the entire array is traversed.

```python
def linear_search(arr, target):
    """
    Search for target in array using linear search.
    
    Args:
        arr: List of elements to search in
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

```java
// Java implementation
public int linearSearch(int[] arr, int target) {
    for (int i = 0; i < arr.length; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}
```

**When to use**: 
- Small arrays where the overhead of more complex algorithms isn't justified
- Unsorted arrays where binary search isn't applicable
- When the array elements are accessed sequentially regardless (e.g., performing an operation on all elements)

### Binary Search

Binary search is an efficient algorithm for finding an element in a sorted array. It works by repeatedly dividing the search interval in half.

```python
def binary_search(arr, target):
    """
    Search for target in a sorted array using binary search.
    
    Args:
        arr: Sorted list of elements to search in
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoids integer overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1
```

```java
// Java implementation
public int binarySearch(int[] arr, int target) {
    int left = 0;
    int right = arr.length - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}
```

**When to use**:
- Sorted arrays
- When search operations are frequent compared to insertions/deletions
- Large data sets where efficiency is crucial

#### Recursive Binary Search

Binary search can also be implemented recursively:

```python
def binary_search_recursive(arr, target, left, right):
    """
    Recursive implementation of binary search.
    
    Args:
        arr: Sorted list of elements to search in
        target: Element to search for
        left: Left boundary of search space
        right: Right boundary of search space
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log n)
    Space Complexity: O(log n) due to recursion stack
    """
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# Usage
def search(arr, target):
    return binary_search_recursive(arr, target, 0, len(arr) - 1)
```

### Interpolation Search

Interpolation search is an improved variant of binary search that works better for uniformly distributed data.

```python
def interpolation_search(arr, target):
    """
    Search for target in a sorted array using interpolation search.
    
    Args:
        arr: Sorted list of elements to search in
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: 
        - Average case: O(log log n) for uniformly distributed data
        - Worst case: O(n)
    Space Complexity: O(1)
    """
    low, high = 0, len(arr) - 1
    
    while low <= high and arr[low] <= target <= arr[high]:
        # Calculate the probable position using linear interpolation
        if arr[high] == arr[low]:  # Avoid division by zero
            pos = low
        else:
            pos = low + ((target - arr[low]) * (high - low)) // (arr[high] - arr[low])
        
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
            
    return -1
```

**When to use**:
- Sorted arrays with uniformly distributed values
- When you want to potentially improve on binary search for specific data distributions

### Jump Search

Jump search is a searching algorithm for sorted arrays that works by jumping ahead by fixed steps and then performing a linear search.

```python
import math

def jump_search(arr, target):
    """
    Search for target in a sorted array using jump search.
    
    Args:
        arr: Sorted list of elements to search in
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(√n)
    Space Complexity: O(1)
    """
    n = len(arr)
    step = int(math.sqrt(n))
    
    # Finding the block where the element is present
    prev = 0
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    # Linear search in the identified block
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return -1
    
    if arr[prev] == target:
        return prev
    
    return -1
```

**When to use**:
- Sorted arrays
- When binary search might be overkill but linear search is too slow
- When the cost of jumping ahead (e.g., in external storage) is less than the cost of checking each element

## Searching in Multidimensional Arrays

### Row-wise and Column-wise Sorted Matrix Search

When a 2D matrix is sorted both row-wise and column-wise, we can use a more efficient search algorithm than checking each element.

```python
def search_in_sorted_matrix(matrix, target):
    """
    Search for target in a matrix where each row and each column is sorted.
    
    Args:
        matrix: 2D array where rows and columns are sorted
        target: Element to search for
        
    Returns:
        Tuple (row, col) if found, (-1, -1) otherwise
        
    Time Complexity: O(m + n) where m is rows and n is columns
    Space Complexity: O(1)
    """
    if not matrix or not matrix[0]:
        return (-1, -1)
    
    rows, cols = len(matrix), len(matrix[0])
    row, col = 0, cols - 1  # Start from top-right corner
    
    while row < rows and col >= 0:
        if matrix[row][col] == target:
            return (row, col)
        elif matrix[row][col] > target:
            col -= 1  # Move left
        else:
            row += 1  # Move down
    
    return (-1, -1)
```

### Binary Search in 2D Arrays

For a completely sorted 2D array (where the last element of a row is smaller than the first element of the next row), we can use binary search by treating it as a 1D array.

```python
def search_in_flattened_sorted_matrix(matrix, target):
    """
    Search for target in a matrix that is completely sorted
    (i.e., can be flattened into a sorted 1D array).
    
    Args:
        matrix: Completely sorted 2D array
        target: Element to search for
        
    Returns:
        Tuple (row, col) if found, (-1, -1) otherwise
        
    Time Complexity: O(log(m*n))
    Space Complexity: O(1)
    """
    if not matrix or not matrix[0]:
        return (-1, -1)
    
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        # Convert 1D index to 2D coordinates
        mid_row, mid_col = mid // cols, mid % cols
        
        if matrix[mid_row][mid_col] == target:
            return (mid_row, mid_col)
        elif matrix[mid_row][mid_col] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return (-1, -1)
```

## Advanced Search Techniques

### Exponential Search

Exponential search combines a galloping phase with binary search, making it suitable for unbounded arrays or when the target is likely to be near the beginning.

```python
def exponential_search(arr, target):
    """
    Search for target using exponential search.
    
    Args:
        arr: Sorted list of elements to search in
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log i) where i is the position of target
    Space Complexity: O(1)
    """
    n = len(arr)
    
    # If target is at first position
    if arr[0] == target:
        return 0
    
    # Find range for binary search by repeated doubling
    i = 1
    while i < n and arr[i] <= target:
        i = i * 2
    
    # Call binary search for the found range
    return binary_search(arr, target, i // 2, min(i, n - 1))

def binary_search(arr, target, left, right):
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1
```

**When to use**:
- Sorted arrays where the target is likely to be near the beginning
- Unbounded/infinite arrays where you don't know the size in advance

### Fibonacci Search

Fibonacci search is a comparison-based search algorithm that uses Fibonacci numbers to divide the array.

```python
def fibonacci_search(arr, target):
    """
    Search for target using Fibonacci search.
    
    Args:
        arr: Sorted list of elements to search in
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    n = len(arr)
    
    # Initialize Fibonacci numbers
    fibMMm2 = 0  # (m-2)'th Fibonacci number
    fibMMm1 = 1  # (m-1)'th Fibonacci number
    fibM = fibMMm2 + fibMMm1  # m'th Fibonacci number
    
    # Find the smallest Fibonacci number greater than or equal to n
    while fibM < n:
        fibMMm2 = fibMMm1
        fibMMm1 = fibM
        fibM = fibMMm2 + fibMMm1
    
    # Marks the eliminated range from front
    offset = -1
    
    # While there are elements to be inspected
    while fibM > 1:
        # Check if fibMMm2 is a valid location
        i = min(offset + fibMMm2, n - 1)
        
        # If target is greater than the value at index i,
        # cut the array from offset to i
        if arr[i] < target:
            fibM = fibMMm1
            fibMMm1 = fibMMm2
            fibMMm2 = fibM - fibMMm1
            offset = i
        # If target is less than the value at index i,
        # cut the array after i+1
        elif arr[i] > target:
            fibM = fibMMm2
            fibMMm1 = fibMMm1 - fibMMm2
            fibMMm2 = fibM - fibMMm1
        # Element found
        else:
            return i
    
    # Compare the last element
    if fibMMm1 and arr[offset + 1] == target:
        return offset + 1
    
    # Element not found
    return -1
```

**When to use**:
- When binary search needs to be optimized for arrays stored on magnetic tapes or similar sequential-access storage
- When the cost of comparisons is higher than the cost of addition/subtraction operations

### Ternary Search

Ternary search divides the array into three parts rather than two, which can be faster in certain scenarios.

```python
def ternary_search(arr, target):
    """
    Search for target using ternary search.
    
    Args:
        arr: Sorted list of elements to search in
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log_3 n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        # Calculate the positions of the two mid points
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2
        
        if target < arr[mid1]:
            right = mid1 - 1
        elif target > arr[mid2]:
            left = mid2 + 1
        else:
            left = mid1 + 1
            right = mid2 - 1
            
    return -1
```

**When to use**:
- When comparison operations are expensive compared to other operations
- In situations where dividing the search space into three parts might yield better results (rare in practice)

## Optimizations and Considerations

### Early Termination

For certain applications, you might want to stop the search as soon as you find the first occurrence of the target, especially when duplicates are present.

```python
def find_first_occurrence(arr, target):
    """
    Find the first occurrence of target in a sorted array.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid  # Save this result
            right = mid - 1  # Continue searching on the left side
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return result
```

### Finding Bounds

When dealing with sorted arrays with duplicates, you might want to find the lower and upper bounds of a range.

```python
def find_bounds(arr, target):
    """
    Find the lower and upper bounds of target in a sorted array.
    
    Returns:
        Tuple (first_occurrence, last_occurrence)
        If target is not found, returns (-1, -1)
        
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    first = find_first_occurrence(arr, target)
    
    if first == -1:
        return (-1, -1)
        
    last = find_last_occurrence(arr, target)
    
    return (first, last)

def find_first_occurrence(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return result

def find_last_occurrence(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            left = mid + 1
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return result
```

### Using Bitwise Operations for Optimization

In performance-critical applications, using bitwise operations can make a small difference:

```cpp
// C++ optimization for binary search
int binarySearchOptimized(std::vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + ((right - left) >> 1);  // Faster than division by 2
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}
```

## Common Search Patterns and Applications

### Finding the Closest Element

When the exact target isn't in the array, you might want to find the closest element.

```python
def find_closest(arr, target):
    """
    Find the element in arr that is closest to target.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    # Edge cases
    if len(arr) == 0:
        return -1
    if len(arr) == 1:
        return 0
        
    # Binary search
    left, right = 0, len(arr) - 1
    
    # If target is outside the range of array
    if target <= arr[left]:
        return left
    if target >= arr[right]:
        return right
        
    # Binary search to find the closest element
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    # At this point, left > right
    # Choose the closer one between arr[right] and arr[left]
    if abs(arr[right] - target) <= abs(arr[left] - target):
        return right
    else:
        return left
```

### Finding a Peak Element

A peak element is an element that is greater than its neighbors.

```python
def find_peak(arr):
    """
    Find a peak element in the array.
    A peak element is an element that is greater than its neighbors.
    For edge elements, we only compare with one neighbor.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    n = len(arr)
    left, right = 0, n - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] > arr[mid + 1]:
            # Peak is in the left half (including mid)
            right = mid
        else:
            # Peak is in the right half
            left = mid + 1
    
    # When left == right, we've found a peak
    return left
```

### Search in Rotated Sorted Array

A common interview problem is searching in a sorted array that has been rotated.

```python
def search_rotated(arr, target):
    """
    Search for target in a rotated sorted array.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
            
        # Check if the left half is sorted
        if arr[left] <= arr[mid]:
            # Check if target is in the left half
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            # Check if target is in the right half
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
                
    return -1
```

## Conclusion

Array searching is a fundamental operation with various algorithms optimized for different scenarios. The choice of search algorithm depends on several factors:

- Is the array sorted?
- How large is the array?
- What's the distribution of the data?
- Are there duplicates?
- Is the array static or dynamic?

Understanding the characteristics and trade-offs of different search algorithms allows you to select the most appropriate one for your specific use case, optimizing both performance and code clarity.

## Practice Problems

1. Implement binary search to find the first and last occurrence of an element in a sorted array.
2. Search in a 2D matrix where each row and column is sorted.
3. Find the minimum element in a rotated sorted array.
4. Find the square root of an integer using binary search.
5. Search for an element in an infinite sorted array (you don't know the size in advance).
