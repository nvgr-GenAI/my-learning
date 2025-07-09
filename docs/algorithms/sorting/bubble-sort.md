# Bubble Sort

Bubble Sort is one of the simplest sorting algorithms. It repeatedly steps through the list, compares adjacent elements and swaps them if they're in the wrong order. The pass through the list is repeated until the list is sorted.

=== "📋 Algorithm Overview"

    ## Key Characteristics
    
    - **Time Complexity**: 
      - Best Case: O(n) - when the array is already sorted
      - Average/Worst Case: O(n²)
    - **Space Complexity**: O(1)
    - **Stable**: Yes - maintains the relative order of equal elements
    - **In-place**: Yes - requires only constant extra space
    - **Adaptive**: Yes - performs better when array is partially sorted
    
    ## When to Use
    
    - Educational purposes - to understand basic sorting concepts
    - Small data sets where simplicity is preferred over efficiency
    - Nearly sorted data where few swaps are needed
    - When memory usage is a concern (since it's in-place)
    - When stability is required
    
    ## Advantages and Disadvantages
    
    ### Advantages
    - Simple implementation
    - Stable sorting algorithm
    - In-place (requires no additional memory)
    - Works well for small or nearly sorted arrays
    
    ### Disadvantages
    - Inefficient for large data sets with O(n²) time complexity
    - Generally performs worse than insertion sort
    - Excessive swapping operations

=== "🔄 How It Works"

    ## Algorithm Steps
    
    1. Start at the beginning of the array
    2. Compare adjacent elements (arr[j] and arr[j+1])
    3. Swap them if they're in the wrong order
    4. Continue to the end of the unsorted portion
    5. After each pass, the largest unsorted element "bubbles up" to its correct position
    6. Repeat until no swaps are needed in a pass
    
    ## Optimization
    
    - Track whether any swaps occurred during a pass
    - If no swaps occur, the array is already sorted and we can exit early
    - This optimization gives bubble sort O(n) performance for already sorted arrays
    
    ## Visual Example
    
    Initial array: `[64, 34, 25, 12, 22, 11, 90]`
    
    **Pass 1:**
    
    ```
    [64, 34, 25, 12, 22, 11, 90] → [34, 64, 25, 12, 22, 11, 90] (swap)
    [34, 64, 25, 12, 22, 11, 90] → [34, 25, 64, 12, 22, 11, 90] (swap)
    [34, 25, 64, 12, 22, 11, 90] → [34, 25, 12, 64, 22, 11, 90] (swap)
    [34, 25, 12, 64, 22, 11, 90] → [34, 25, 12, 22, 64, 11, 90] (swap)
    [34, 25, 12, 22, 64, 11, 90] → [34, 25, 12, 22, 11, 64, 90] (swap)
    ```
    
    After Pass 1: `[34, 25, 12, 22, 11, 64, 90]` (90 is in correct position)
    
    **Pass 2:**
    
    ```
    [34, 25, 12, 22, 11, 64] → [25, 34, 12, 22, 11, 64] (swap)
    [25, 34, 12, 22, 11, 64] → [25, 12, 34, 22, 11, 64] (swap)
    [25, 12, 34, 22, 11, 64] → [25, 12, 22, 34, 11, 64] (swap)
    [25, 12, 22, 34, 11, 64] → [25, 12, 22, 11, 34, 64] (swap)
    ```
    
    After Pass 2: `[25, 12, 22, 11, 34, 64, 90]` (64 and 90 are in correct positions)
    
    **Passes 3-6:**
    
    Continue until the array is fully sorted: `[11, 12, 22, 25, 34, 64, 90]`

=== "💻 Implementation"

    ## Basic Bubble Sort
    
    ```python
    def bubble_sort_basic(arr):
        """
        Basic bubble sort without optimization
        Always performs O(n²) comparisons
        """
        n = len(arr)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        
        return arr
    ```
    
    ## Optimized Bubble Sort
    
    ```python
    def bubble_sort(arr):
        """
        Bubble Sort implementation with early termination
        Time: O(n²), Space: O(1)
        Stable, in-place, adaptive
        """
        n = len(arr)
        
        for i in range(n):
            # Flag to optimize for already sorted arrays
            swapped = False
            
            # Last i elements are already sorted
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            
            # If no swapping occurred, array is sorted
            if not swapped:
                break
        
        return arr
    ```
    
    ## Bidirectional Bubble Sort (Cocktail Sort)
    
    ```python
    def cocktail_sort(arr):
        """
        Bidirectional bubble sort (cocktail sort)
        Bubbles values in both directions
        """
        n = len(arr)
        swapped = True
        start = 0
        end = n - 1
        
        while swapped:
            # Reset swapped flag for forward pass
            swapped = False
            
            # Forward pass (like regular bubble sort)
            for i in range(start, end):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
            
            # If nothing was swapped, array is sorted
            if not swapped:
                break
                
            # Move end point back as largest element is at end
            end -= 1
            
            # Reset swapped flag for backward pass
            swapped = False
            
            # Backward pass (bubbles smallest element to beginning)
            for i in range(end - 1, start - 1, -1):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
            
            # Move start point forward as smallest element is at beginning
            start += 1
            
        return arr
    ```
    
    ## Example Usage
    
    ```python
    # Test with sample array
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", test_arr)
    
    sorted_arr = bubble_sort(test_arr.copy())
    print("Sorted array:", sorted_arr)
    
    # Performance test
    import time
    import random
    
    # Test with larger array
    large_arr = [random.randint(1, 1000) for _ in range(1000)]
    
    start_time = time.time()
    bubble_sort(large_arr.copy())
    end_time = time.time()
    
    print(f"Time taken for 1000 elements: {end_time - start_time:.4f} seconds")
    ```

=== "📊 Performance Analysis"

    ## Time Complexity
    
    - **Best Case**: O(n) - When array is already sorted (with optimization)
    - **Average Case**: O(n²)
    - **Worst Case**: O(n²) - When array is sorted in reverse order
    
    ## Space Complexity
    
    - O(1) - Only requires constant extra space regardless of input size
    
    ## Comparison with Other Simple Sorts
    
    | Algorithm | Best | Average | Worst | Space | Stable | In-Place |
    |-----------|------|---------|-------|-------|--------|----------|
    | Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |
    | Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No | Yes |
    | Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |
    
    ## Performance Factors
    
    1. **Swapping Overhead** - Bubble sort typically performs more swaps than other algorithms
    2. **Cache Performance** - Good locality of reference when working with adjacent elements
    3. **Array Size Impact** - Dramatic slowdown as array size increases due to O(n²) complexity

=== "💡 Applications & Variations"

    ## Real-world Applications
    
    - **Educational tool** - Used to teach basic sorting concepts
    - **Small datasets** - When simplicity is more important than efficiency
    - **Nearly sorted data** - Can perform well when most elements are in the right place
    - **Embedded systems** - When memory is extremely limited and code simplicity is valued
    
    ## Variations
    
    ### 1. Cocktail Shaker Sort (Bidirectional Bubble Sort)
    
    Sorts in both directions, moving the smallest elements to the beginning and largest to the end:
    
    ```python
    def cocktail_sort(arr):
        n = len(arr)
        swapped = True
        start = 0
        end = n - 1
        
        while swapped:
            swapped = False
            
            # Forward pass (move largest to the end)
            for i in range(start, end):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
            
            if not swapped:
                break
            
            end -= 1
            swapped = False
            
            # Backward pass (move smallest to the beginning)
            for i in range(end - 1, start - 1, -1):
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
            
            start += 1
    ```
    
    ### 2. Odd-Even Sort
    
    A parallel version of bubble sort that alternates between comparing odd/even indexed pairs and even/odd indexed pairs:
    
    ```python
    def odd_even_sort(arr):
        n = len(arr)
        sorted = False
        
        while not sorted:
            sorted = True
            
            # Odd-even pairs
            for i in range(1, n-1, 2):
                if arr[i] > arr[i+1]:
                    arr[i], arr[i+1] = arr[i+1], arr[i]
                    sorted = False
            
            # Even-odd pairs
            for i in range(0, n-1, 2):
                if arr[i] > arr[i+1]:
                    arr[i], arr[i+1] = arr[i+1], arr[i]
                    sorted = False
    ```
    
    ### 3. Comb Sort
    
    An improvement over bubble sort that eliminates "turtles" (small values near the end of the list):
    
    ```python
    def comb_sort(arr):
        n = len(arr)
        gap = n
        shrink = 1.3
        sorted = False
        
        while not sorted:
            # Update gap
            gap = int(gap / shrink)
            if gap <= 1:
                gap = 1
                sorted = True
            
            i = 0
            while i + gap < n:
                if arr[i] > arr[i + gap]:
                    arr[i], arr[i + gap] = arr[i + gap], arr[i]
                    sorted = False
                i += 1
    ```

=== "🎯 Practice Problems"

    ## Beginner Problems
    
    1. **Bubble Sort Implementation**
       - Implement basic bubble sort from scratch
       - Add the optimization for early termination
       
    2. **Sort an Array of Strings**
       - Use bubble sort to arrange strings alphabetically
       
    3. **Descending Order Sort**
       - Modify bubble sort to sort in descending order
    
    ## Intermediate Problems
    
    1. **Custom Comparator**
       - Implement bubble sort with a custom comparison function
       - Example: Sort objects by specific property
    
    2. **Implement Cocktail Sort**
       - Create a bidirectional bubble sort implementation
       - Compare its performance with standard bubble sort
    
    3. **Optimize Swaps**
       - Reduce the number of swap operations in bubble sort
       - Example: Use temporary variable more efficiently
    
    ## Application Problems
    
    1. **Sort Linked List**
       - Apply bubble sort algorithm to a linked list
       - Compare the approach with array-based bubble sort
    
    2. **Bubble Sort Visualization**
       - Create a step-by-step visualization of bubble sort
       - Show how elements "bubble up" to their final positions
    
    3. **Partially Sorted Data Analysis**
       - Compare bubble sort performance on various degrees of sorted data
       - Measure how the optimization performs as sortedness increases

Pass 2:
[34, 25, 12, 22, 11, 64, 90] → [25, 34, 12, 22, 11, 64, 90]
[25, 34, 12, 22, 11, 64, 90] → [25, 12, 34, 22, 11, 64, 90]
[25, 12, 34, 22, 11, 64, 90] → [25, 12, 22, 34, 11, 64, 90]
[25, 12, 22, 34, 11, 64, 90] → [25, 12, 22, 11, 34, 64, 90]
[25, 12, 22, 11, 34, 64, 90] → [25, 12, 22, 11, 34, 64, 90] (no swap)

After Pass 2: [25, 12, 22, 11, 34, 64, 90] (64 is in correct position)

... (continues until fully sorted)

## When to Use

**Good for:**

- Educational purposes (easy to understand)
- Very small datasets
- Nearly sorted data (with optimization)
- When simplicity is more important than efficiency

**Not good for:**

- Large datasets
- Performance-critical applications
- Production systems

## Advantages

- Simple to understand and implement
- No additional memory space needed
- Stable sorting algorithm
- Can detect if list is already sorted (optimized version)
- Works well on small datasets

## Disadvantages

- Poor time complexity O(n²)
- More comparisons and swaps compared to other algorithms
- Not suitable for large datasets
- Generally slower than other O(n²) algorithms like insertion sort

## Variants

### Cocktail Shaker Sort (Bidirectional Bubble Sort)

```python
def cocktail_shaker_sort(arr):
    """
    Cocktail shaker sort - bubble sort that works in both directions
    Slightly better performance than regular bubble sort
    """
    n = len(arr)
    start = 0
    end = n - 1
    swapped = True
    
    while swapped:
        swapped = False
        
        # Forward pass
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        
        if not swapped:
            break
        
        end -= 1
        swapped = False
        
        # Backward pass
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        
        start += 1
    
    return arr
```

## Practice Problems

1. **Bubble Sort Count**: Count the number of swaps needed to sort an array
2. **Bubble Sort Visualization**: Implement a step-by-step visualization
3. **Optimized Bubble Sort**: Implement various optimizations
4. **Bubble Sort Comparison**: Compare performance with other sorting algorithms

---

*Bubble sort may not be the most efficient, but it's an excellent starting point to understand sorting algorithms!*
