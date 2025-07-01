# Arrays

## 📚 What are Arrays?

Arrays are fundamental data structures that store elements of the same type in contiguous memory locations. Each element can be accessed directly using its index, making arrays one of the most basic and important data structures in computer science.

## 🔑 Key Concepts

### Static vs Dynamic Arrays

**Static Arrays**
- Fixed size determined at compile time
- Memory allocated on stack (usually)
- Cannot grow or shrink during runtime
- Examples: C arrays, Java arrays

**Dynamic Arrays** 
- Size can change during runtime
- Memory allocated on heap
- Automatically resize when needed
- Examples: Python lists, C++ vectors, Java ArrayList

### Memory Layout
```
Array: [10, 20, 30, 40, 50]
Memory: [10][20][30][40][50]
Indices: 0   1   2   3   4
```

## ⏱️ Time & Space Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Access    | O(1)           | O(1)             |
| Search    | O(n)           | O(1)             |
| Insert    | O(n)           | O(1)             |
| Delete    | O(n)           | O(1)             |
| Append    | O(1) amortized | O(1)             |

**Space Complexity**: O(n) where n is the number of elements

## 🛠️ Common Operations

### 1. Array Creation and Initialization

```python
# Different ways to create arrays
arr1 = [1, 2, 3, 4, 5]                    # List literal
arr2 = [0] * 10                           # Initialize with zeros
arr3 = list(range(1, 11))                 # Range-based
arr4 = [i**2 for i in range(5)]          # List comprehension

# 2D Arrays
matrix = [[0 for _ in range(3)] for _ in range(3)]
```

### 2. Array Traversal

```python
def traverse_array(arr):
    """Different ways to traverse an array"""
    
    # Method 1: Index-based
    for i in range(len(arr)):
        print(f"Index {i}: {arr[i]}")
    
    # Method 2: Direct iteration
    for element in arr:
        print(element)
    
    # Method 3: With enumerate
    for index, value in enumerate(arr):
        print(f"Index {index}: {value}")
    
    # Method 4: Reverse traversal
    for i in range(len(arr) - 1, -1, -1):
        print(arr[i])
```

### 3. Searching Algorithms

```python
def linear_search(arr, target):
    """Linear search - O(n)"""
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

def binary_search(arr, target):
    """Binary search - O(log n) - requires sorted array"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def find_all_occurrences(arr, target):
    """Find all indices where target appears"""
    indices = []
    for i, val in enumerate(arr):
        if val == target:
            indices.append(i)
    return indices
```

### 4. Array Manipulation

```python
def insert_element(arr, index, value):
    """Insert element at specific index"""
    arr.insert(index, value)
    return arr

def delete_element(arr, index):
    """Delete element at specific index"""
    if 0 <= index < len(arr):
        return arr.pop(index)
    return None

def rotate_array(arr, k):
    """Rotate array to the right by k positions"""
    n = len(arr)
    k = k % n  # Handle k > n
    return arr[-k:] + arr[:-k]

def reverse_array(arr):
    """Reverse array in-place"""
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    return arr
```

### 5. Array Algorithms

```python
def two_sum(nums, target):
    """Find two numbers that add up to target"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

def max_subarray_sum(arr):
    """Kadane's algorithm for maximum subarray sum"""
    max_sum = current_sum = arr[0]
    
    for i in range(1, len(arr)):
        current_sum = max(arr[i], current_sum + arr[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

def merge_sorted_arrays(arr1, arr2):
    """Merge two sorted arrays"""
    result = []
    i = j = 0
    
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result

def remove_duplicates(arr):
    """Remove duplicates from sorted array"""
    if not arr:
        return 0
    
    write_index = 1
    for read_index in range(1, len(arr)):
        if arr[read_index] != arr[read_index - 1]:
            arr[write_index] = arr[read_index]
            write_index += 1
    
    return write_index
```

## 🎯 Common Array Problems

### Easy Level
1. **Two Sum** - Find pair that sums to target
2. **Best Time to Buy and Sell Stock** - Maximum profit
3. **Contains Duplicate** - Check for duplicates
4. **Maximum Subarray** - Kadane's algorithm
5. **Merge Sorted Array** - Merge in-place

### Medium Level  
6. **3Sum** - Find triplets that sum to zero
7. **Product of Array Except Self** - Without division
8. **Rotate Array** - Rotate by k positions
9. **Find All Duplicates** - In array of size n
10. **Spiral Matrix** - Traverse in spiral order

### Hard Level
11. **Median of Two Sorted Arrays** - O(log(min(m,n)))
12. **First Missing Positive** - O(n) time, O(1) space
13. **Trapping Rain Water** - Two pointer technique
14. **Sliding Window Maximum** - Using deque

## 💡 Tips and Tricks

### 1. Two Pointer Technique
```python
def is_palindrome(s):
    """Check if array represents palindrome"""
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

### 2. Sliding Window
```python
def max_sum_subarray(arr, k):
    """Maximum sum of subarray of size k"""
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(len(arr) - k):
        window_sum = window_sum - arr[i] + arr[i + k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

### 3. Prefix Sum
```python
def range_sum_query(arr, queries):
    """Answer range sum queries efficiently"""
    prefix_sum = [0]
    for num in arr:
        prefix_sum.append(prefix_sum[-1] + num)
    
    results = []
    for left, right in queries:
        results.append(prefix_sum[right + 1] - prefix_sum[left])
    
    return results
```

## 🔧 When to Use Arrays

**Use Arrays When:**
- Need fast random access to elements (O(1))
- Memory efficiency is important
- Cache performance matters
- Working with mathematical operations
- Size is relatively fixed

**Avoid Arrays When:**
- Frequent insertions/deletions in middle
- Size varies dramatically
- Need to maintain sorted order with insertions

## 📊 Arrays vs Other Data Structures

| Feature | Array | Linked List | Dynamic Array |
|---------|-------|-------------|---------------|
| Access | O(1) | O(n) | O(1) |
| Search | O(n) | O(n) | O(n) |
| Insert | O(n) | O(1) | O(1) amortized |
| Delete | O(n) | O(1) | O(1) amortized |
| Memory | Contiguous | Scattered | Contiguous |
| Cache | Excellent | Poor | Excellent |

## 🎓 Practice Problems

### Beginner
- [ ] Two Sum (LeetCode 1)
- [ ] Best Time to Buy and Sell Stock (LeetCode 121)
- [ ] Contains Duplicate (LeetCode 217)
- [ ] Maximum Subarray (LeetCode 53)

### Intermediate
- [ ] Product of Array Except Self (LeetCode 238)
- [ ] Rotate Array (LeetCode 189)
- [ ] Find All Numbers Disappeared in an Array (LeetCode 448)
- [ ] Move Zeroes (LeetCode 283)

### Advanced
- [ ] Median of Two Sorted Arrays (LeetCode 4)
- [ ] First Missing Positive (LeetCode 41)
- [ ] Trapping Rain Water (LeetCode 42)
- [ ] Sliding Window Maximum (LeetCode 239)

## 📚 Additional Resources

- **Books**: "Introduction to Algorithms" by CLRS
- **Online**: GeeksforGeeks Array section
- **Practice**: LeetCode Array tag
- **Visualization**: VisuAlgo Array section
