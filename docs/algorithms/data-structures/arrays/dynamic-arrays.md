# Dynamic Arrays

## 🔍 Overview

Dynamic arrays are resizable arrays that can grow and shrink during runtime. They provide the convenience of automatic memory management while maintaining the performance characteristics of arrays for most operations.

---

## 📊 Characteristics

### Key Properties

- **Variable Size**: Can grow and shrink during runtime
- **Automatic Resizing**: Handles memory allocation automatically
- **Contiguous Memory**: Elements still stored consecutively
- **Amortized Performance**: Most operations have excellent average-case performance
- **Flexibility**: Easy to use with built-in methods

### Memory Management

```text
Dynamic Array Growth:
Initial: [1, 2, 3] (capacity: 3, size: 3)
         ┌───┬───┬───┐
         │ 1 │ 2 │ 3 │
         └───┴───┴───┘

After append(4): [1, 2, 3, 4, _, _] (capacity: 6, size: 4)
                 ┌───┬───┬───┬───┬───┬───┐
                 │ 1 │ 2 │ 3 │ 4 │   │   │
                 └───┴───┴───┴───┴───┴───┘
```

---

## ⏱️ Time Complexities

| Operation | Average Case | Worst Case | Notes |
|-----------|--------------|------------|-------|
| **Access** | O(1) | O(1) | Direct index access |
| **Search** | O(n) | O(n) | Linear search |
| **Insert (end)** | O(1) | O(n) | Amortized, may trigger resize |
| **Insert (middle)** | O(n) | O(n) | Shift elements |
| **Delete (end)** | O(1) | O(1) | Simple removal |
| **Delete (middle)** | O(n) | O(n) | Shift elements |

### Amortized Analysis

The key insight is that while individual operations might be expensive (O(n) for resize), the **amortized cost** over many operations is O(1) for append operations.

---

## 💻 Implementations

### Python Lists

```python
def python_list_examples():
    """Demonstrate Python list operations and behavior."""
    
    # Create and initialize
    dynamic_arr = []
    print(f"Initial list: {dynamic_arr}")
    
    # Append elements (amortized O(1))
    for i in range(5):
        dynamic_arr.append(i)
        print(f"After append({i}): {dynamic_arr}")
    
    # Access elements (O(1))
    print(f"Element at index 2: {dynamic_arr[2]}")
    print(f"Last element: {dynamic_arr[-1]}")
    
    # Insert at specific position (O(n))
    dynamic_arr.insert(2, 99)
    print(f"After insert(2, 99): {dynamic_arr}")
    
    # Remove elements
    dynamic_arr.remove(99)  # Remove by value (O(n))
    print(f"After remove(99): {dynamic_arr}")
    
    popped = dynamic_arr.pop()  # Remove from end (O(1))
    print(f"Popped: {popped}, List: {dynamic_arr}")
    
    popped_index = dynamic_arr.pop(1)  # Remove from index (O(n))
    print(f"Popped from index 1: {popped_index}, List: {dynamic_arr}")
    
    return dynamic_arr

def list_comprehensions():
    """Demonstrate list comprehensions and advanced operations."""
    
    # Basic list comprehension
    squares = [x**2 for x in range(10)]
    print(f"Squares: {squares}")
    
    # Conditional list comprehension
    evens = [x for x in range(20) if x % 2 == 0]
    print(f"Even numbers: {evens}")
    
    # Nested list comprehension
    matrix = [[i*j for j in range(3)] for i in range(3)]
    print(f"Matrix: {matrix}")
    
    # List comprehension with function
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in range(2, 50) if is_prime(x)]
    print(f"Primes: {primes}")
    
    return squares, evens, matrix, primes

def advanced_list_operations():
    """Demonstrate advanced list operations."""
    
    # Slicing operations
    arr = list(range(10))
    print(f"Original: {arr}")
    print(f"arr[2:5]: {arr[2:5]}")  # Slice
    print(f"arr[:5]: {arr[:5]}")    # Start to index
    print(f"arr[5:]: {arr[5:]}")    # Index to end
    print(f"arr[::2]: {arr[::2]}")  # Every 2nd element
    print(f"arr[::-1]: {arr[::-1]}")  # Reverse
    
    # List concatenation and repetition
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    concatenated = list1 + list2
    repeated = list1 * 3
    
    print(f"Concatenation: {concatenated}")
    print(f"Repetition: {repeated}")
    
    # In-place operations
    arr = [1, 2, 3, 4, 5]
    arr.extend([6, 7, 8])  # Extend in-place
    print(f"After extend: {arr}")
    
    arr.reverse()  # Reverse in-place
    print(f"After reverse: {arr}")
    
    arr.sort()  # Sort in-place
    print(f"After sort: {arr}")
    
    return arr
```

### Custom Dynamic Array Implementation

```python
class DynamicArray:
    """Custom implementation of a dynamic array."""
    
    def __init__(self, initial_capacity=4):
        self._capacity = initial_capacity
        self._size = 0
        self._data = [None] * self._capacity
    
    def __len__(self):
        """Return the size of the array."""
        return self._size
    
    def __getitem__(self, index):
        """Get element at given index."""
        if not 0 <= index < self._size:
            raise IndexError("Array index out of range")
        return self._data[index]
    
    def __setitem__(self, index, value):
        """Set element at given index."""
        if not 0 <= index < self._size:
            raise IndexError("Array index out of range")
        self._data[index] = value
    
    def append(self, value):
        """Add element to the end of the array."""
        if self._size >= self._capacity:
            self._resize()
        
        self._data[self._size] = value
        self._size += 1
    
    def insert(self, index, value):
        """Insert element at given index."""
        if not 0 <= index <= self._size:
            raise IndexError("Array index out of range")
        
        if self._size >= self._capacity:
            self._resize()
        
        # Shift elements to the right
        for i in range(self._size, index, -1):
            self._data[i] = self._data[i - 1]
        
        self._data[index] = value
        self._size += 1
    
    def remove(self, value):
        """Remove first occurrence of value."""
        for i in range(self._size):
            if self._data[i] == value:
                self._remove_at(i)
                return
        raise ValueError("Value not found in array")
    
    def pop(self, index=-1):
        """Remove and return element at given index."""
        if self._size == 0:
            raise IndexError("Pop from empty array")
        
        if index < 0:
            index = self._size + index
        
        if not 0 <= index < self._size:
            raise IndexError("Array index out of range")
        
        value = self._data[index]
        self._remove_at(index)
        return value
    
    def _remove_at(self, index):
        """Remove element at given index."""
        # Shift elements to the left
        for i in range(index, self._size - 1):
            self._data[i] = self._data[i + 1]
        
        self._size -= 1
        self._data[self._size] = None  # Clear reference
        
        # Shrink if necessary
        if self._size <= self._capacity // 4 and self._capacity > 4:
            self._shrink()
    
    def _resize(self):
        """Double the capacity of the array."""
        old_capacity = self._capacity
        self._capacity *= 2
        old_data = self._data
        self._data = [None] * self._capacity
        
        # Copy old data
        for i in range(old_capacity):
            self._data[i] = old_data[i]
    
    def _shrink(self):
        """Halve the capacity of the array."""
        old_capacity = self._capacity
        self._capacity //= 2
        old_data = self._data
        self._data = [None] * self._capacity
        
        # Copy data
        for i in range(self._size):
            self._data[i] = old_data[i]
    
    def __str__(self):
        """String representation of the array."""
        return f"[{', '.join(str(self._data[i]) for i in range(self._size))}]"
    
    def __repr__(self):
        return f"DynamicArray({self})"
    
    def capacity(self):
        """Return current capacity."""
        return self._capacity
    
    def is_empty(self):
        """Check if array is empty."""
        return self._size == 0

def test_dynamic_array():
    """Test the custom dynamic array implementation."""
    
    arr = DynamicArray()
    print(f"Initial: {arr}, capacity: {arr.capacity()}")
    
    # Test append
    for i in range(10):
        arr.append(i)
        print(f"After append({i}): {arr}, capacity: {arr.capacity()}")
    
    # Test access
    print(f"Element at index 5: {arr[5]}")
    
    # Test insert
    arr.insert(3, 99)
    print(f"After insert(3, 99): {arr}")
    
    # Test remove
    arr.remove(99)
    print(f"After remove(99): {arr}")
    
    # Test pop
    popped = arr.pop()
    print(f"Popped: {popped}, Array: {arr}")
    
    # Test shrinking
    for _ in range(7):
        arr.pop()
        print(f"After pop: {arr}, capacity: {arr.capacity()}")
    
    return arr
```

---

## 🎯 Common Patterns

### 1. Dynamic Window/Buffer

```python
def sliding_window_dynamic():
    """Implement sliding window with dynamic size."""
    
    class SlidingWindow:
        def __init__(self):
            self.window = []
            self.sum = 0
        
        def add(self, value):
            """Add value to window."""
            self.window.append(value)
            self.sum += value
        
        def remove_front(self):
            """Remove element from front of window."""
            if self.window:
                removed = self.window.pop(0)
                self.sum -= removed
                return removed
            return None
        
        def get_average(self):
            """Get average of current window."""
            if not self.window:
                return 0
            return self.sum / len(self.window)
        
        def get_max(self):
            """Get maximum in current window."""
            return max(self.window) if self.window else None
        
        def size(self):
            return len(self.window)
    
    # Usage example
    window = SlidingWindow()
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for i, value in enumerate(data):
        window.add(value)
        
        # Maintain window size of 3
        if window.size() > 3:
            window.remove_front()
        
        print(f"Step {i+1}: Window {window.window}, "
              f"Avg: {window.get_average():.2f}, Max: {window.get_max()}")
    
    return window

def dynamic_stack():
    """Implement stack using dynamic array."""
    
    class DynamicStack:
        def __init__(self):
            self.items = []
        
        def push(self, item):
            """Push item onto stack."""
            self.items.append(item)
        
        def pop(self):
            """Pop item from stack."""
            if self.is_empty():
                raise IndexError("Pop from empty stack")
            return self.items.pop()
        
        def peek(self):
            """Peek at top item without removing."""
            if self.is_empty():
                raise IndexError("Peek from empty stack")
            return self.items[-1]
        
        def is_empty(self):
            return len(self.items) == 0
        
        def size(self):
            return len(self.items)
        
        def __str__(self):
            return f"Stack({self.items})"
    
    # Usage example
    stack = DynamicStack()
    
    # Push elements
    for i in range(5):
        stack.push(i)
        print(f"After push({i}): {stack}")
    
    # Pop elements
    while not stack.is_empty():
        popped = stack.pop()
        print(f"Popped {popped}: {stack}")
    
    return stack
```

### 2. Auto-Resizing Containers

```python
def auto_resizing_containers():
    """Demonstrate auto-resizing container patterns."""
    
    class AutoResizingMatrix:
        """Matrix that automatically resizes when accessed."""
        
        def __init__(self, initial_rows=1, initial_cols=1):
            self.rows = initial_rows
            self.cols = initial_cols
            self.data = [[0 for _ in range(initial_cols)] 
                        for _ in range(initial_rows)]
        
        def get(self, row, col):
            """Get element, expanding matrix if necessary."""
            self._ensure_capacity(row + 1, col + 1)
            return self.data[row][col]
        
        def set(self, row, col, value):
            """Set element, expanding matrix if necessary."""
            self._ensure_capacity(row + 1, col + 1)
            self.data[row][col] = value
        
        def _ensure_capacity(self, min_rows, min_cols):
            """Ensure matrix has at least min_rows x min_cols."""
            
            # Expand rows if needed
            while self.rows < min_rows:
                self.data.append([0] * self.cols)
                self.rows += 1
            
            # Expand columns if needed
            while self.cols < min_cols:
                for row in self.data:
                    row.append(0)
                self.cols += 1
        
        def shape(self):
            return (self.rows, self.cols)
        
        def __str__(self):
            return '\n'.join([' '.join(f'{x:3}' for x in row) 
                            for row in self.data])
    
    # Usage example
    matrix = AutoResizingMatrix(2, 2)
    print(f"Initial matrix shape: {matrix.shape()}")
    print(matrix)
    print()
    
    # Set element that requires expansion
    matrix.set(4, 6, 42)
    print(f"After set(4, 6, 42), shape: {matrix.shape()}")
    print(matrix)
    
    return matrix

def dynamic_hash_table():
    """Implement hash table with dynamic resizing."""
    
    class DynamicHashTable:
        def __init__(self, initial_capacity=8):
            self.capacity = initial_capacity
            self.size = 0
            self.buckets = [[] for _ in range(self.capacity)]
            self.load_factor_threshold = 0.75
        
        def _hash(self, key):
            """Simple hash function."""
            return hash(key) % self.capacity
        
        def put(self, key, value):
            """Insert or update key-value pair."""
            index = self._hash(key)
            bucket = self.buckets[index]
            
            # Update existing key
            for i, (k, v) in enumerate(bucket):
                if k == key:
                    bucket[i] = (key, value)
                    return
            
            # Add new key-value pair
            bucket.append((key, value))
            self.size += 1
            
            # Check if resize is needed
            if self.size > self.capacity * self.load_factor_threshold:
                self._resize()
        
        def get(self, key):
            """Get value for key."""
            index = self._hash(key)
            bucket = self.buckets[index]
            
            for k, v in bucket:
                if k == key:
                    return v
            
            raise KeyError(key)
        
        def delete(self, key):
            """Delete key-value pair."""
            index = self._hash(key)
            bucket = self.buckets[index]
            
            for i, (k, v) in enumerate(bucket):
                if k == key:
                    del bucket[i]
                    self.size -= 1
                    return v
            
            raise KeyError(key)
        
        def _resize(self):
            """Double the capacity and rehash all elements."""
            old_buckets = self.buckets
            self.capacity *= 2
            self.size = 0
            self.buckets = [[] for _ in range(self.capacity)]
            
            # Rehash all elements
            for bucket in old_buckets:
                for key, value in bucket:
                    self.put(key, value)
        
        def load_factor(self):
            """Calculate current load factor."""
            return self.size / self.capacity
        
        def __str__(self):
            items = []
            for bucket in self.buckets:
                for key, value in bucket:
                    items.append(f"{key}: {value}")
            return "{" + ", ".join(items) + "}"
    
    # Usage example
    ht = DynamicHashTable(4)
    
    # Insert elements
    for i in range(10):
        ht.put(f"key{i}", f"value{i}")
        print(f"After put(key{i}, value{i}): "
              f"size={ht.size}, capacity={ht.capacity}, "
              f"load_factor={ht.load_factor():.2f}")
    
    return ht
```

---

## 🚀 Performance Optimization

### Optimization Techniques

```python
def memory_optimization():
    """Demonstrate memory optimization techniques."""
    
    import sys
    
    def compare_memory_usage():
        """Compare memory usage of different approaches."""
        
        # Method 1: Frequent append (many reallocations)
        list1 = []
        for i in range(1000):
            list1.append(i)
        
        # Method 2: Pre-allocate (single allocation)
        list2 = [0] * 1000
        for i in range(1000):
            list2[i] = i
        
        # Method 3: List comprehension (optimized)
        list3 = [i for i in range(1000)]
        
        print(f"Method 1 (append): {sys.getsizeof(list1)} bytes")
        print(f"Method 2 (pre-allocate): {sys.getsizeof(list2)} bytes")
        print(f"Method 3 (comprehension): {sys.getsizeof(list3)} bytes")
        
        return list1, list2, list3
    
    def memory_efficient_operations():
        """Show memory-efficient operations."""
        
        # Use generators for large datasets
        def sum_squares_list(n):
            """Memory-intensive approach."""
            squares = [i**2 for i in range(n)]
            return sum(squares)
        
        def sum_squares_generator(n):
            """Memory-efficient approach."""
            squares = (i**2 for i in range(n))
            return sum(squares)
        
        n = 1000000
        print(f"Sum of squares (1 to {n}): {sum_squares_generator(n)}")
        
        # Use itertools for memory-efficient operations
        import itertools
        
        # Memory-efficient chunking
        def chunked(iterable, chunk_size):
            """Yield successive chunks from iterable."""
            iterator = iter(iterable)
            while True:
                chunk = list(itertools.islice(iterator, chunk_size))
                if not chunk:
                    break
                yield chunk
        
        # Process large dataset in chunks
        large_data = range(1000000)
        chunk_sums = []
        
        for chunk in chunked(large_data, 1000):
            chunk_sums.append(sum(chunk))
        
        total_sum = sum(chunk_sums)
        print(f"Total sum using chunking: {total_sum}")
        
        return sum_squares_list, sum_squares_generator, chunked
    
    return compare_memory_usage(), memory_efficient_operations()
```

---

## 🎯 When to Use Dynamic Arrays

### ✅ Best Use Cases

1. **Unknown Size**: When you don't know the final size in advance
2. **Frequent Appends**: When you need to add elements frequently
3. **Random Access**: When you need O(1) access to elements
4. **General Purpose**: Most common use case for collections
5. **Built-in Methods**: When you need rich built-in functionality

### ❌ Limitations

1. **Memory Overhead**: Extra capacity maintained for growth
2. **Insertion Cost**: O(n) for middle insertions
3. **Deletion Cost**: O(n) for middle deletions
4. **Memory Fragmentation**: Frequent resizing can cause fragmentation
5. **Worst-case Performance**: Occasional O(n) operations due to resizing

### Performance Comparison

| Operation | Static Array | Dynamic Array | Notes |
|-----------|--------------|---------------|-------|
| Access | O(1) | O(1) | Same performance |
| Append | N/A | O(1) amortized | Dynamic arrays excel |
| Insert Middle | O(n) | O(n) | Same performance |
| Memory Usage | Exact | 25-50% overhead | Static arrays more efficient |

---

## 🔗 Related Topics

- **[Static Arrays](static-arrays.md)**: For fixed-size, high-performance arrays
- **[Multidimensional Arrays](multidimensional-arrays.md)**: For matrices and tensors
- **[Easy Problems](easy-problems.md)**: Practice with dynamic array problems
- **[Algorithm Patterns](../../patterns/index.md)**: Common algorithmic patterns

---

*Ready to explore multi-dimensional data? Check out [Multidimensional Arrays](multidimensional-arrays.md) next!*
