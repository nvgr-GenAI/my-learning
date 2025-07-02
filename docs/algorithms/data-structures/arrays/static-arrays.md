# Static Arrays

## 🔍 Overview

Static arrays are arrays with a fixed size determined at compile time or creation. They offer excellent performance characteristics due to their predictable memory layout and are commonly used in systems programming and performance-critical applications.

---

## 📊 Characteristics

### Key Properties

- **Fixed Size**: Size determined at creation and cannot be changed
- **Contiguous Memory**: Elements stored in consecutive memory locations
- **Type Homogeneous**: All elements must be of the same data type
- **Direct Access**: O(1) access time using index
- **Memory Efficient**: No overhead for dynamic resizing

### Memory Layout

```text
Static Array: [10, 20, 30, 40, 50]
Index:         0   1   2   3   4
Memory:       ┌────┬────┬────┬────┬────┐
              │ 10 │ 20 │ 30 │ 40 │ 50 │
              └────┴────┴────┴────┴────┘
Address:     1000 1004 1008 1012 1016
```

---

## ⏱️ Time Complexities

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| **Access** | O(1) | Direct index calculation |
| **Search** | O(n) | Linear search required |
| **Insert** | O(n) | Shift elements (if allowed) |
| **Delete** | O(n) | Shift elements (if allowed) |
| **Update** | O(1) | Direct access by index |

---

## 💻 Implementations

### Python Array Module

```python
import array

def static_array_examples():
    """Demonstrate static array usage with Python's array module."""
    
    # Create typed arrays
    int_array = array.array('i', [1, 2, 3, 4, 5])  # Integer array
    float_array = array.array('f', [1.1, 2.2, 3.3])  # Float array
    char_array = array.array('u', 'hello')  # Unicode character array
    
    print(f"Integer array: {int_array}")
    print(f"Float array: {float_array}")
    print(f"Character array: {char_array}")
    
    # Memory efficiency comparison
    import sys
    python_list = [1, 2, 3, 4, 5]
    
    print(f"Python list size: {sys.getsizeof(python_list)} bytes")
    print(f"Array module size: {sys.getsizeof(int_array)} bytes")
    
    return int_array, float_array, char_array

def array_operations():
    """Basic operations on static arrays."""
    
    # Create and initialize
    arr = array.array('i', [10, 20, 30, 40, 50])
    
    # Access elements
    print(f"Element at index 2: {arr[2]}")
    print(f"First element: {arr[0]}")
    print(f"Last element: {arr[-1]}")
    
    # Modify elements
    arr[1] = 25
    print(f"After modification: {list(arr)}")
    
    # Search for element
    try:
        index = arr.index(30)
        print(f"Element 30 found at index: {index}")
    except ValueError:
        print("Element not found")
    
    # Count occurrences
    arr.append(30)  # Add duplicate
    count = arr.count(30)
    print(f"Element 30 appears {count} times")
    
    return arr
```

### NumPy Arrays (Scientific Computing)

```python
import numpy as np

def numpy_static_arrays():
    """Demonstrate NumPy static arrays for numerical computing."""
    
    # Create arrays of different types
    int_arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    float_arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    bool_arr = np.array([True, False, True], dtype=np.bool_)
    
    print(f"Integer array: {int_arr}, dtype: {int_arr.dtype}")
    print(f"Float array: {float_arr}, dtype: {float_arr.dtype}")
    print(f"Boolean array: {bool_arr}, dtype: {bool_arr.dtype}")
    
    # Multi-dimensional arrays
    matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    print(f"2D array shape: {matrix.shape}")
    print(f"2D array:\n{matrix}")
    
    # Memory layout information
    print(f"Array strides: {matrix.strides}")
    print(f"Memory usage: {matrix.nbytes} bytes")
    
    return int_arr, float_arr, matrix

def numpy_operations():
    """Advanced operations with NumPy static arrays."""
    
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    
    # Vectorized operations (much faster than loops)
    squared = arr ** 2
    doubled = arr * 2
    sum_all = np.sum(arr)
    
    print(f"Original: {arr}")
    print(f"Squared: {squared}")
    print(f"Doubled: {doubled}")
    print(f"Sum: {sum_all}")
    
    # Statistical operations
    print(f"Mean: {np.mean(arr)}")
    print(f"Standard deviation: {np.std(arr)}")
    print(f"Min: {np.min(arr)}, Max: {np.max(arr)}")
    
    # Array manipulation
    reshaped = arr.reshape(5, 1)  # Column vector
    print(f"Reshaped to column:\n{reshaped}")
    
    return arr, squared, doubled
```

### C-Style Arrays (Memory Layout)

```python
import ctypes

def c_style_arrays():
    """Demonstrate C-style arrays using ctypes."""
    
    # Create C-style integer array
    IntArray5 = ctypes.c_int * 5
    c_array = IntArray5(1, 2, 3, 4, 5)
    
    print("C-style array contents:")
    for i in range(5):
        print(f"c_array[{i}] = {c_array[i]}")
    
    # Memory address calculation
    print(f"\nMemory addresses:")
    for i in range(5):
        addr = ctypes.addressof(c_array) + i * ctypes.sizeof(ctypes.c_int)
        print(f"c_array[{i}] address: {hex(addr)}")
    
    # Direct memory manipulation
    c_array[2] = 99
    print(f"\nAfter modification: {list(c_array)}")
    
    return c_array
```

---

## 🎯 Common Patterns

### 1. Fixed-Size Buffer

```python
def fixed_size_buffer_example():
    """Implement a fixed-size circular buffer."""
    
    class CircularBuffer:
        def __init__(self, size):
            self.buffer = array.array('i', [0] * size)
            self.size = size
            self.head = 0
            self.tail = 0
            self.count = 0
        
        def push(self, value):
            """Add element to buffer."""
            if self.count < self.size:
                self.buffer[self.tail] = value
                self.tail = (self.tail + 1) % self.size
                self.count += 1
            else:
                # Overwrite oldest element
                self.buffer[self.tail] = value
                self.tail = (self.tail + 1) % self.size
                self.head = (self.head + 1) % self.size
        
        def pop(self):
            """Remove and return oldest element."""
            if self.count == 0:
                raise IndexError("Buffer is empty")
            
            value = self.buffer[self.head]
            self.head = (self.head + 1) % self.size
            self.count -= 1
            return value
        
        def peek(self):
            """Return oldest element without removing."""
            if self.count == 0:
                raise IndexError("Buffer is empty")
            return self.buffer[self.head]
        
        def is_full(self):
            return self.count == self.size
        
        def is_empty(self):
            return self.count == 0
    
    # Usage example
    buffer = CircularBuffer(3)
    for i in range(5):
        buffer.push(i)
        print(f"Pushed {i}, buffer state: {list(buffer.buffer)}")
    
    return buffer
```

### 2. Memory Pool Allocation

```python
def memory_pool_example():
    """Implement a simple memory pool using static array."""
    
    class MemoryPool:
        def __init__(self, block_size, num_blocks):
            self.block_size = block_size
            self.num_blocks = num_blocks
            self.pool = array.array('B', [0] * (block_size * num_blocks))
            self.free_blocks = list(range(num_blocks))  # Track free blocks
        
        def allocate(self):
            """Allocate a block from the pool."""
            if not self.free_blocks:
                raise MemoryError("No free blocks available")
            
            block_index = self.free_blocks.pop()
            start_addr = block_index * self.block_size
            return start_addr, block_index
        
        def deallocate(self, block_index):
            """Return a block to the pool."""
            if block_index in self.free_blocks:
                raise ValueError("Block already free")
            
            self.free_blocks.append(block_index)
            
            # Clear the block (optional)
            start = block_index * self.block_size
            end = start + self.block_size
            for i in range(start, end):
                self.pool[i] = 0
        
        def write_block(self, block_index, data):
            """Write data to a specific block."""
            if len(data) > self.block_size:
                raise ValueError("Data too large for block")
            
            start = block_index * self.block_size
            for i, byte in enumerate(data):
                self.pool[start + i] = byte
        
        def read_block(self, block_index):
            """Read data from a specific block."""
            start = block_index * self.block_size
            end = start + self.block_size
            return self.pool[start:end]
    
    # Usage example
    pool = MemoryPool(64, 10)  # 10 blocks of 64 bytes each
    
    # Allocate blocks
    addr1, block1 = pool.allocate()
    addr2, block2 = pool.allocate()
    
    # Write data
    pool.write_block(block1, b"Hello, World!")
    pool.write_block(block2, b"Static arrays rock!")
    
    # Read data
    data1 = bytes(pool.read_block(block1)[:13])  # First 13 bytes
    data2 = bytes(pool.read_block(block2)[:19])  # First 19 bytes
    
    print(f"Block {block1}: {data1}")
    print(f"Block {block2}: {data2}")
    
    # Deallocate
    pool.deallocate(block1)
    pool.deallocate(block2)
    
    return pool
```

---

## 🚀 Performance Optimization

### Cache-Friendly Access Patterns

```python
def cache_optimization_examples():
    """Demonstrate cache-friendly array access patterns."""
    
    import time
    
    def create_large_matrix(rows, cols):
        """Create a large 2D matrix."""
        return [[i * cols + j for j in range(cols)] for i in range(rows)]
    
    def row_major_sum(matrix):
        """Sum matrix elements in row-major order (cache-friendly)."""
        total = 0
        for row in matrix:
            for element in row:
                total += element
        return total
    
    def column_major_sum(matrix):
        """Sum matrix elements in column-major order (cache-unfriendly)."""
        total = 0
        rows, cols = len(matrix), len(matrix[0])
        for j in range(cols):
            for i in range(rows):
                total += matrix[i][j]
        return total
    
    # Performance comparison
    matrix = create_large_matrix(1000, 1000)
    
    # Row-major access (faster)
    start_time = time.time()
    row_sum = row_major_sum(matrix)
    row_time = time.time() - start_time
    
    # Column-major access (slower)
    start_time = time.time()
    col_sum = column_major_sum(matrix)
    col_time = time.time() - start_time
    
    print(f"Row-major sum: {row_sum}, Time: {row_time:.4f}s")
    print(f"Column-major sum: {col_sum}, Time: {col_time:.4f}s")
    print(f"Row-major is {col_time/row_time:.2f}x faster")
    
    return row_time, col_time

def prefetching_example():
    """Demonstrate memory prefetching benefits."""
    
    import array
    
    def sequential_access(arr):
        """Access array elements sequentially."""
        total = 0
        for i in range(len(arr)):
            total += arr[i]
        return total
    
    def random_access(arr, indices):
        """Access array elements randomly."""
        total = 0
        for i in indices:
            total += arr[i]
        return total
    
    # Create large array
    size = 1000000
    arr = array.array('i', range(size))
    
    # Generate random indices
    import random
    random_indices = list(range(size))
    random.shuffle(random_indices)
    
    # Time sequential vs random access
    import time
    
    start_time = time.time()
    seq_sum = sequential_access(arr)
    seq_time = time.time() - start_time
    
    start_time = time.time()
    rand_sum = random_access(arr, random_indices)
    rand_time = time.time() - start_time
    
    print(f"Sequential access: {seq_sum}, Time: {seq_time:.4f}s")
    print(f"Random access: {rand_sum}, Time: {rand_time:.4f}s")
    print(f"Sequential is {rand_time/seq_time:.2f}x faster")
    
    return seq_time, rand_time
```

---

## 🎯 When to Use Static Arrays

### ✅ Best Use Cases

1. **Known Fixed Size**: When array size is determined at compile time
2. **Performance Critical**: Maximum performance with minimal overhead
3. **Memory Constrained**: Embedded systems with limited memory
4. **Scientific Computing**: NumPy arrays for numerical operations
5. **System Programming**: Low-level operations requiring direct memory control

### ❌ Limitations

1. **Fixed Size**: Cannot grow or shrink dynamically
2. **Memory Waste**: May allocate more space than needed
3. **No Built-in Safety**: Potential for buffer overflows
4. **Limited Flexibility**: Less convenient than dynamic arrays

---

## 🔗 Related Topics

- **[Dynamic Arrays](dynamic-arrays.md)**: For variable-size collections
- **[Multidimensional Arrays](multidimensional-arrays.md)**: For matrices and tensors
- **[Easy Problems](easy-problems.md)**: Practice with static array problems
- **[Memory Management](../../../systems/memory-management.md)**: Understanding memory layout

---

*Ready to explore dynamic behavior? Check out [Dynamic Arrays](dynamic-arrays.md) next!*
