# Merge Sort

## Overview

Merge Sort is a divide and conquer algorithm that divides the input array into two halves, recursively sorts them, and then merges the sorted halves to produce a sorted output.

## Algorithm

1. **Divide**: Divide the unsorted array into two halves
2. **Conquer**: Recursively sort the two halves
3. **Combine**: Merge the sorted halves to form a single sorted array

## Implementation

### Python Implementation

```python
def merge_sort(arr):
    """
    Sorts an array using the Merge Sort algorithm.
    
    Args:
        arr: The array to be sorted
        
    Returns:
        Sorted array
    """
    if len(arr) <= 1:
        return arr
    
    # Divide the array into two halves
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    
    # Recursively sort the two halves
    left = merge_sort(left)
    right = merge_sort(right)
    
    # Merge the sorted halves
    return merge(left, right)

def merge(left, right):
    """
    Merges two sorted arrays into a single sorted array.
    
    Args:
        left: First sorted array
        right: Second sorted array
        
    Returns:
        Merged sorted array
    """
    result = []
    i = j = 0
    
    # Compare elements from both arrays and add the smaller one to the result
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add any remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# Example usage
arr = [38, 27, 43, 3, 9, 82, 10]
sorted_arr = merge_sort(arr)
print(f"Original array: {arr}")
print(f"Sorted array: {sorted_arr}")

# In-place implementation (less intuitive but more memory efficient)
def merge_sort_in_place(arr, start=0, end=None):
    """
    Sorts an array in-place using the Merge Sort algorithm.
    
    Args:
        arr: The array to be sorted
        start: Start index of the array segment to sort
        end: End index (exclusive) of the array segment to sort
        
    Returns:
        None (the array is sorted in-place)
    """
    if end is None:
        end = len(arr)
    
    if end - start <= 1:
        return
    
    # Divide the array into two halves
    mid = (start + end) // 2
    
    # Recursively sort the two halves
    merge_sort_in_place(arr, start, mid)
    merge_sort_in_place(arr, mid, end)
    
    # Merge the sorted halves
    merge_in_place(arr, start, mid, end)

def merge_in_place(arr, start, mid, end):
    """
    Merges two sorted segments of an array in-place.
    
    Args:
        arr: The array containing the segments to merge
        start: Start index of the first segment
        mid: Start index of the second segment
        end: End index (exclusive) of the second segment
        
    Returns:
        None (the array is merged in-place)
    """
    # Create temporary arrays
    left = arr[start:mid]
    right = arr[mid:end]
    
    # Merge back into arr[start:end]
    i = j = 0
    k = start
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    
    # Add any remaining elements
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1

# Example usage of in-place implementation
arr2 = [38, 27, 43, 3, 9, 82, 10]
merge_sort_in_place(arr2)
print(f"Original array: {[38, 27, 43, 3, 9, 82, 10]}")
print(f"Sorted array (in-place): {arr2}")
```

### Java Implementation

```java
import java.util.Arrays;

public class MergeSort {
    
    public static void mergeSort(int[] arr) {
        if (arr.length <= 1) {
            return;
        }
        
        // Create a temporary array for merging
        int[] temp = new int[arr.length];
        
        // Call the recursive helper function
        mergeSortHelper(arr, temp, 0, arr.length - 1);
    }
    
    private static void mergeSortHelper(int[] arr, int[] temp, int left, int right) {
        if (left < right) {
            // Find the middle point
            int mid = left + (right - left) / 2;
            
            // Sort first and second halves
            mergeSortHelper(arr, temp, left, mid);
            mergeSortHelper(arr, temp, mid + 1, right);
            
            // Merge the sorted halves
            merge(arr, temp, left, mid, right);
        }
    }
    
    private static void merge(int[] arr, int[] temp, int left, int mid, int right) {
        // Copy data to temporary arrays
        for (int i = left; i <= right; i++) {
            temp[i] = arr[i];
        }
        
        // Merge the two arrays back into arr[left..right]
        int i = left;      // Initial index of first subarray
        int j = mid + 1;   // Initial index of second subarray
        int k = left;      // Initial index of merged array
        
        while (i <= mid && j <= right) {
            if (temp[i] <= temp[j]) {
                arr[k] = temp[i];
                i++;
            } else {
                arr[k] = temp[j];
                j++;
            }
            k++;
        }
        
        // Copy the remaining elements of left array, if any
        while (i <= mid) {
            arr[k] = temp[i];
            i++;
            k++;
        }
        
        // Copy the remaining elements of right array, if any
        // (not necessary, as they are already in place)
    }
    
    // Alternative implementation that creates new arrays
    public static int[] mergeSortCreateNew(int[] arr) {
        if (arr.length <= 1) {
            return arr;
        }
        
        // Divide the array into two halves
        int mid = arr.length / 2;
        int[] left = Arrays.copyOfRange(arr, 0, mid);
        int[] right = Arrays.copyOfRange(arr, mid, arr.length);
        
        // Recursively sort the two halves
        left = mergeSortCreateNew(left);
        right = mergeSortCreateNew(right);
        
        // Merge the sorted halves
        return mergeArrays(left, right);
    }
    
    private static int[] mergeArrays(int[] left, int[] right) {
        int[] result = new int[left.length + right.length];
        int i = 0, j = 0, k = 0;
        
        while (i < left.length && j < right.length) {
            if (left[i] <= right[j]) {
                result[k++] = left[i++];
            } else {
                result[k++] = right[j++];
            }
        }
        
        // Copy the remaining elements
        while (i < left.length) {
            result[k++] = left[i++];
        }
        
        while (j < right.length) {
            result[k++] = right[j++];
        }
        
        return result;
    }
    
    public static void main(String[] args) {
        int[] arr = {38, 27, 43, 3, 9, 82, 10};
        
        System.out.println("Original array: " + Arrays.toString(arr));
        
        // Using in-place merge sort
        int[] arr1 = Arrays.copyOf(arr, arr.length);
        mergeSort(arr1);
        System.out.println("Sorted array (in-place): " + Arrays.toString(arr1));
        
        // Using merge sort that creates new arrays
        int[] arr2 = Arrays.copyOf(arr, arr.length);
        arr2 = mergeSortCreateNew(arr2);
        System.out.println("Sorted array (create new): " + Arrays.toString(arr2));
    }
}
```

## Complexity Analysis

- **Time Complexity**: O(n log n) - The array is divided into two halves in each step (log n levels) and merging n elements takes O(n) time
- **Space Complexity**: O(n) - Additional space is needed for the temporary arrays during merging

## Advantages and Disadvantages

### Advantages

- Stable sort (preserves the relative order of equal elements)
- Guaranteed O(n log n) time complexity regardless of the input
- Efficient for large datasets
- Well-suited for external sorting (sorting data that doesn't fit in memory)

### Disadvantages

- Requires additional O(n) space for merging
- Not an in-place sorting algorithm in its standard implementation
- For small arrays, the overhead of recursive calls can make it slower than simpler algorithms like Insertion Sort

## Divide and Conquer Analysis

Merge Sort is a classic example of the divide and conquer paradigm:

1. **Divide**: The problem is broken down into smaller subproblems (dividing the array into halves)
2. **Conquer**: The subproblems are solved recursively (sorting the halves)
3. **Combine**: The solutions to the subproblems are combined to form the solution to the original problem (merging the sorted halves)

## Variations

1. **Bottom-up Merge Sort**: An iterative version that starts with single-element subarrays and progressively merges them
2. **Natural Merge Sort**: Takes advantage of existing order in the input
3. **Timsort**: A hybrid sorting algorithm derived from merge sort and insertion sort, used in Python and Java
4. **Parallel Merge Sort**: Divides the work among multiple processors
5. **External Merge Sort**: Used when the data doesn't fit in memory

## Applications

1. **Database Systems**: For sorting large datasets
2. **External Sorting**: When data doesn't fit in memory
3. **Inversion Count Problem**: To count inversions in an array
4. **Custom Sort Orders**: When stability is important

## Practice Problems

1. [Merge Sort](https://leetcode.com/problems/sort-an-array/) - Implement merge sort
2. [Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/) - A problem that can be solved using merge sort
3. [Count Inversions](https://practice.geeksforgeeks.org/problems/inversion-of-array-1587115620/1) - Count inversions in an array using merge sort
4. [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) - Apply merge sort concept to merge k sorted lists

## References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
2. Sedgewick, R., & Wayne, K. (2011). Algorithms (4th ed.). Addison-Wesley Professional.
3. Knuth, D. E. (1998). The Art of Computer Programming, Volume 3: Sorting and Searching (2nd ed.). Addison-Wesley Professional.
