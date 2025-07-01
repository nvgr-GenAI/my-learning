# Arrays

## Overview
Arrays are fundamental data structures that store elements in contiguous memory locations.

## Key Concepts
- **Static vs Dynamic Arrays**
- **Time Complexity**: O(1) access, O(n) search
- **Space Complexity**: O(n)

## Common Operations

### Array Traversal
```python
def traverse_array(arr):
    for i in range(len(arr)):
        print(f"Index {i}: {arr[i]}")
```

### Array Search
```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

### Array Insertion
```python
def insert_at_index(arr, index, value):
    arr.insert(index, value)
    return arr
```

## Common Problems
1. **Two Sum Problem**
2. **Maximum Subarray (Kadane's Algorithm)**
3. **Rotate Array**
4. **Merge Sorted Arrays**
5. **Remove Duplicates**

## Practice Problems
- [ ] Two Sum
- [ ] Best Time to Buy and Sell Stock
- [ ] Contains Duplicate
- [ ] Product of Array Except Self
- [ ] Maximum Subarray

## Time Complexities
| Operation | Best | Average | Worst |
|-----------|------|---------|-------|
| Access    | O(1) | O(1)    | O(1)  |
| Search    | O(1) | O(n)    | O(n)  |
| Insertion | O(1) | O(n)    | O(n)  |
| Deletion  | O(1) | O(n)    | O(n)  |
