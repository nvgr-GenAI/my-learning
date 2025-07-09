# Quick Sort

## Overview

Quick Sort is a divide and conquer algorithm that works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays according to whether they are less than or greater than the pivot. The sub-arrays are then recursively sorted.

## Algorithm

1. **Select a pivot** from the array (various strategies exist)
2. **Partition** the array around the pivot:
   - Move all elements smaller than the pivot to the left side
   - Move all elements greater than the pivot to the right side
3. **Recursively sort** the sub-arrays

## Implementation

### Python Implementation

```python
def quick_sort(arr):
    """
    Sorts an array using the Quick Sort algorithm.
    
    Args:
        arr: The array to be sorted
        
    Returns:
        Sorted array
    """
    if len(arr) <= 1:
        return arr
    
    # Select pivot (here we simply choose the first element)
    pivot = arr[0]
    
    # Partition the array
    less = [x for x in arr[1:] if x <= pivot]
    greater = [x for x in arr[1:] if x > pivot]
    
    # Recursively sort sub-arrays and combine the result
    return quick_sort(less) + [pivot] + quick_sort(greater)

# In-place implementation
def quick_sort_in_place(arr, low=0, high=None):
    """
    Sorts an array in-place using the Quick Sort algorithm.
    
    Args:
        arr: The array to be sorted
        low: Starting index of the segment to sort
        high: Ending index of the segment to sort
        
    Returns:
        None (the array is sorted in-place)
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition the array and get the pivot index
        pivot_idx = partition(arr, low, high)
        
        # Recursively sort the sub-arrays
        quick_sort_in_place(arr, low, pivot_idx - 1)
        quick_sort_in_place(arr, pivot_idx + 1, high)

def partition(arr, low, high):
    """
    Partitions an array around a pivot.
    
    Args:
        arr: The array to partition
        low: Starting index
        high: Ending index
        
    Returns:
        Index of the pivot after partitioning
    """
    # Choose the rightmost element as the pivot
    pivot = arr[high]
    
    # Index of the smaller element
    i = low - 1
    
    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            # Increment index of smaller element
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # Place pivot in its correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    
    # Return the pivot's index
    return i + 1

# Example usage
arr = [38, 27, 43, 3, 9, 82, 10]
print(f"Original array: {arr}")

sorted_arr = quick_sort(arr)
print(f"Sorted array (functional): {sorted_arr}")

arr2 = [38, 27, 43, 3, 9, 82, 10]
quick_sort_in_place(arr2)
print(f"Sorted array (in-place): {arr2}")

# Randomized Quick Sort (better average case performance)
import random

def randomized_quick_sort(arr, low=0, high=None):
    """
    Sorts an array in-place using the Randomized Quick Sort algorithm.
    
    Args:
        arr: The array to be sorted
        low: Starting index of the segment to sort
        high: Ending index of the segment to sort
        
    Returns:
        None (the array is sorted in-place)
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Choose a random pivot
        pivot_idx = random_partition(arr, low, high)
        
        # Recursively sort the sub-arrays
        randomized_quick_sort(arr, low, pivot_idx - 1)
        randomized_quick_sort(arr, pivot_idx + 1, high)

def random_partition(arr, low, high):
    """
    Randomly selects a pivot and partitions the array.
    
    Args:
        arr: The array to partition
        low: Starting index
        high: Ending index
        
    Returns:
        Index of the pivot after partitioning
    """
    # Choose a random pivot
    random_idx = random.randint(low, high)
    
    # Swap the random pivot with the last element
    arr[random_idx], arr[high] = arr[high], arr[random_idx]
    
    # Use the standard partition function
    return partition(arr, low, high)

# Example usage of randomized quick sort
arr3 = [38, 27, 43, 3, 9, 82, 10]
randomized_quick_sort(arr3)
print(f"Sorted array (randomized): {arr3}")
```

### Java Implementation

```java
import java.util.Arrays;
import java.util.Random;

public class QuickSort {
    
    // Simple quick sort implementation that creates new arrays
    public static int[] quickSort(int[] arr) {
        if (arr.length <= 1) {
            return arr;
        }
        
        // Select pivot (first element in this case)
        int pivot = arr[0];
        
        // Count elements for each sub-array
        int lessCount = 0, greaterCount = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] <= pivot) {
                lessCount++;
            } else {
                greaterCount++;
            }
        }
        
        // Create sub-arrays
        int[] less = new int[lessCount];
        int[] greater = new int[greaterCount];
        
        // Fill sub-arrays
        int l = 0, g = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] <= pivot) {
                less[l++] = arr[i];
            } else {
                greater[g++] = arr[i];
            }
        }
        
        // Recursively sort sub-arrays
        less = quickSort(less);
        greater = quickSort(greater);
        
        // Combine the result
        int[] result = new int[arr.length];
        System.arraycopy(less, 0, result, 0, less.length);
        result[less.length] = pivot;
        System.arraycopy(greater, 0, result, less.length + 1, greater.length);
        
        return result;
    }
    
    // In-place quick sort implementation
    public static void quickSortInPlace(int[] arr) {
        quickSortInPlace(arr, 0, arr.length - 1);
    }
    
    private static void quickSortInPlace(int[] arr, int low, int high) {
        if (low < high) {
            // Partition the array and get the pivot index
            int pivotIdx = partition(arr, low, high);
            
            // Recursively sort the sub-arrays
            quickSortInPlace(arr, low, pivotIdx - 1);
            quickSortInPlace(arr, pivotIdx + 1, high);
        }
    }
    
    private static int partition(int[] arr, int low, int high) {
        // Choose the rightmost element as the pivot
        int pivot = arr[high];
        
        // Index of the smaller element
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            // If current element is smaller than or equal to pivot
            if (arr[j] <= pivot) {
                // Increment index of smaller element
                i++;
                
                // Swap arr[i] and arr[j]
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        
        // Place pivot in its correct position
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        
        // Return the pivot's index
        return i + 1;
    }
    
    // Randomized Quick Sort implementation
    public static void randomizedQuickSort(int[] arr) {
        randomizedQuickSort(arr, 0, arr.length - 1);
    }
    
    private static void randomizedQuickSort(int[] arr, int low, int high) {
        if (low < high) {
            // Choose a random pivot
            int pivotIdx = randomPartition(arr, low, high);
            
            // Recursively sort the sub-arrays
            randomizedQuickSort(arr, low, pivotIdx - 1);
            randomizedQuickSort(arr, pivotIdx + 1, high);
        }
    }
    
    private static int randomPartition(int[] arr, int low, int high) {
        // Choose a random pivot
        Random rand = new Random();
        int randomIdx = low + rand.nextInt(high - low + 1);
        
        // Swap the random pivot with the last element
        int temp = arr[randomIdx];
        arr[randomIdx] = arr[high];
        arr[high] = temp;
        
        // Use the standard partition function
        return partition(arr, low, high);
    }
    
    public static void main(String[] args) {
        int[] arr = {38, 27, 43, 3, 9, 82, 10};
        
        System.out.println("Original array: " + Arrays.toString(arr));
        
        // Using non-in-place quick sort
        int[] sortedArr = quickSort(Arrays.copyOf(arr, arr.length));
        System.out.println("Sorted array (functional): " + Arrays.toString(sortedArr));
        
        // Using in-place quick sort
        int[] arr2 = Arrays.copyOf(arr, arr.length);
        quickSortInPlace(arr2);
        System.out.println("Sorted array (in-place): " + Arrays.toString(arr2));
        
        // Using randomized quick sort
        int[] arr3 = Arrays.copyOf(arr, arr.length);
        randomizedQuickSort(arr3);
        System.out.println("Sorted array (randomized): " + Arrays.toString(arr3));
    }
}
```

## Complexity Analysis

- **Time Complexity**:
  - **Best Case**: O(n log n) - When the pivot divides the array into roughly equal halves
  - **Average Case**: O(n log n) - With good pivot selection
  - **Worst Case**: O(n²) - When the pivot is always the smallest or largest element (this can be mitigated with good pivot selection strategies)
  
- **Space Complexity**:
  - **Functional Implementation**: O(n) - Due to the creation of new sub-arrays
  - **In-place Implementation**: O(log n) - For the recursion stack in the average case

## Advantages and Disadvantages

### Advantages

- Excellent average-case performance
- In-place implementation requires minimal extra memory
- Cache-friendly due to good locality of reference
- Can be easily parallelized
- Works well for virtual memory environments

### Disadvantages

- Worst-case performance is O(n²)
- Not stable (equal elements may change their relative order)
- Performance depends heavily on the pivot selection strategy
- Recursive nature can lead to stack overflow for very large arrays

## Pivot Selection Strategies

1. **First Element**: Simple but can lead to worst-case performance for sorted or nearly sorted arrays
2. **Last Element**: Also simple but has similar issues as the first element
3. **Middle Element**: Better for sorted or nearly sorted arrays
4. **Random Element**: Provides good average-case performance and is simple to implement
5. **Median-of-Three**: Chooses the median of the first, middle, and last elements
6. **Median-of-Medians**: A more complex but deterministic approach that guarantees O(n log n) worst-case

## Variations

1. **Dual-Pivot Quick Sort**: Uses two pivots instead of one (used in Java's `Arrays.sort()`)
2. **Three-Way Quick Sort**: Handles equal elements efficiently by partitioning into "less than", "equal to", and "greater than"
3. **Quick Select**: Uses the partitioning idea to find the kth smallest/largest element
4. **Introsort**: Hybrid algorithm that combines quick sort with heap sort to guarantee O(n log n) worst-case
5. **External Quick Sort**: Adaptation for datasets that don't fit in memory

## Applications

1. **General-purpose sorting**: Used in many standard library implementations
2. **Quick Select Algorithm**: Finding the kth smallest/largest element
3. **Numeric stability tests**: Used in numerical analysis
4. **Java's Arrays.sort()**: Uses a dual-pivot quick sort

## Practice Problems

1. [Quick Sort](https://leetcode.com/problems/sort-an-array/) - Implement quick sort
2. [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) - Uses quick select, a variation of quick sort
3. [Sort Colors](https://leetcode.com/problems/sort-colors/) - A variation of the Dutch national flag problem, which is similar to quick sort's partitioning
4. [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) - Can be solved using quick select

## References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
2. Sedgewick, R., & Wayne, K. (2011). Algorithms (4th ed.). Addison-Wesley Professional.
3. Hoare, C. A. R. (1962). "Quicksort". The Computer Journal. 5 (1): 10–16.
4. Bentley, J. L., & McIlroy, M. D. (1993). "Engineering a Sort Function". Software: Practice and Experience. 23 (11): 1249–1265.
