# Sorting Algorithms - Medium Problems

## 🎯 Learning Objectives

Master intermediate sorting concepts and hybrid algorithms:

- Advanced sorting applications and modifications
- Custom comparators and multi-key sorting
- Sorting with constraints and optimizations
- Merge operations and divide-and-conquer techniques
- Counting and bucket sort applications

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Sort Colors (Dutch Flag) | 3-way Partitioning | Medium | O(n) | O(1) |
    | 2 | Merge Intervals | Sorting + Merging | Medium | O(n log n) | O(n) |
    | 3 | Largest Number | Custom Comparator | Medium | O(n log n) | O(n) |
    | 4 | Top K Frequent Elements | Heap/Bucket Sort | Medium | O(n log k) | O(k) |
    | 5 | Sort Array by Frequency | Frequency + Stability | Medium | O(n log n) | O(n) |
    | 6 | Meeting Rooms II | Sweep Line/Heap | Medium | O(n log n) | O(n) |
    | 7 | Car Pool | Difference Array | Medium | O(n log n) | O(n) |
    | 8 | Non-overlapping Intervals | Greedy + Sorting | Medium | O(n log n) | O(1) |
    | 9 | Insert Interval | Sorted Insertion | Medium | O(n) | O(n) |
    | 10 | Queue Reconstruction by Height | Greedy Insertion | Medium | O(n²) | O(n) |
    | 11 | Relative Ranks | Index Sorting | Medium | O(n log n) | O(n) |
    | 12 | Custom Sort String | Counting + Custom Order | Medium | O(n + m) | O(1) |
    | 13 | Sort Characters by Frequency | Frequency Sorting | Medium | O(n log n) | O(n) |
    | 14 | Wiggle Sort II | Median + Partitioning | Medium | O(n log n) | O(n) |
    | 15 | H-Index | Sorting + Linear Scan | Medium | O(n log n) | O(1) |

=== "🎯 Core Patterns"

    **🎨 Custom Sorting:**
    - Implement custom comparators for complex objects
    - Multi-key sorting with tie-breaking rules
    - Stability requirements in sorting
    
    **🔄 Merge Operations:**
    - Merging overlapping intervals
    - Combining sorted sequences
    - Event-based sorting (sweep line)
    
    **📊 Frequency-based Sorting:**
    - Count frequencies and sort by frequency
    - Bucket sort for limited value ranges
    - Top-K problems with heaps
    
    **🎯 Constrained Sorting:**
    - In-place sorting with limited space
    - Sorting with specific patterns (wiggle sort)
    - Reconstruction problems

=== "💡 Solutions"

    === "Sort Colors (Dutch Flag)"
        ```python
        def sortColors(nums):
            """
            Sort array with values 0, 1, 2 in-place
            Dutch National Flag algorithm
            """
            left = curr = 0
            right = len(nums) - 1
            
            while curr <= right:
                if nums[curr] == 0:
                    nums[left], nums[curr] = nums[curr], nums[left]
                    left += 1
                    curr += 1
                elif nums[curr] == 2:
                    nums[curr], nums[right] = nums[right], nums[curr]
                    right -= 1
                    # Don't increment curr, need to check swapped element
                else:  # nums[curr] == 1
                    curr += 1
        ```
    
    === "Merge Intervals"
        ```python
        def merge(intervals):
            """
            Merge overlapping intervals
            """
            if not intervals:
                return []
            
            # Sort by start time
            intervals.sort(key=lambda x: x[0])
            merged = [intervals[0]]
            
            for current in intervals[1:]:
                last_merged = merged[-1]
                
                if current[0] <= last_merged[1]:
                    # Overlapping, merge them
                    last_merged[1] = max(last_merged[1], current[1])
                else:
                    # No overlap, add new interval
                    merged.append(current)
            
            return merged
        ```
    
    === "Largest Number"
        ```python
        from functools import cmp_to_key
        
        def largestNumber(nums):
            """
            Arrange numbers to form largest possible number
            """
            def compare(x, y):
                # Compare xy vs yx
                if x + y > y + x:
                    return -1
                elif x + y < y + x:
                    return 1
                else:
                    return 0
            
            # Convert to strings
            str_nums = list(map(str, nums))
            
            # Sort with custom comparator
            str_nums.sort(key=cmp_to_key(compare))
            
            # Handle edge case of all zeros
            result = ''.join(str_nums)
            return '0' if result[0] == '0' else result
        ```
    
    === "Top K Frequent Elements"
        ```python
        import heapq
        from collections import Counter
        
        def topKFrequent(nums, k):
            """
            Find k most frequent elements
            """
            # Count frequencies
            counter = Counter(nums)
            
            # Use heap to get top k
            return heapq.nlargest(k, counter.keys(), key=counter.get)
            
            # Alternative: Bucket sort approach
            # buckets = [[] for _ in range(len(nums) + 1)]
            # for num, freq in counter.items():
            #     buckets[freq].append(num)
            # 
            # result = []
            # for i in range(len(buckets) - 1, -1, -1):
            #     result.extend(buckets[i])
            #     if len(result) >= k:
            #         break
            # return result[:k]
        ```
    
    === "Meeting Rooms II"
        ```python
        import heapq
        
        def minMeetingRooms(intervals):
            """
            Find minimum number of meeting rooms needed
            """
            if not intervals:
                return 0
            
            # Sort by start time
            intervals.sort(key=lambda x: x[0])
            
            # Min heap to track end times
            min_heap = []
            
            for interval in intervals:
                # If earliest ending meeting has ended, reuse room
                if min_heap and min_heap[0] <= interval[0]:
                    heapq.heappop(min_heap)
                
                # Add current meeting's end time
                heapq.heappush(min_heap, interval[1])
            
            return len(min_heap)
        ```
    
    === "Queue Reconstruction"
        ```python
        def reconstructQueue(people):
            """
            Reconstruct queue based on height and position
            """
            # Sort by height (desc), then by position (asc)
            people.sort(key=lambda x: (-x[0], x[1]))
            
            result = []
            for person in people:
                # Insert at the position specified by k value
                result.insert(person[1], person)
            
            return result
        ```

=== "📊 Key Insights"

    **🔧 Algorithm Selection:**
    - **Custom Comparator**: When standard ordering doesn't apply
    - **Counting Sort**: Limited range of values (0-k)
    - **Bucket Sort**: Uniform distribution of values
    - **Heap**: Top-K problems, priority-based sorting
    - **Two-pointer**: In-place partitioning problems
    
    **⚡ Optimization Techniques:**
    - **Stable vs Unstable**: Choose based on requirements
    - **In-place vs Extra Space**: Trade-off between space and time
    - **Preprocessing**: Sort once, query multiple times
    - **Early Termination**: Stop when enough elements found
    
    **🎯 Pattern Recognition:**
    - **Interval problems**: Usually need sorting by start/end times
    - **Frequency problems**: Count first, then sort by frequency
    - **Custom ordering**: Define comparison function carefully
    - **Reconstruction**: Often requires sorting by multiple criteria

=== "🚀 Advanced Tips"

    **💡 Interview Strategy:**
    1. **Clarify Requirements**: Stable? In-place? Custom order?
    2. **Choose Algorithm**: Based on input size and constraints
    3. **Handle Edge Cases**: Empty arrays, duplicates, invalid input
    4. **Optimize**: Consider time/space trade-offs
    5. **Test**: Verify with edge cases and examples
    
    **🔍 Common Pitfalls:**
    - **Stability**: When order of equal elements matters
    - **Custom Comparators**: Ensure transitivity and consistency
    - **Integer Overflow**: In problems like "Largest Number"
    - **Edge Cases**: Single element, all same elements
    
    **🏆 Best Practices:**
    - Use built-in sort when possible (highly optimized)
    - Custom comparators for complex sorting logic
    - Consider counting/bucket sort for limited ranges
    - Use heaps for Top-K problems with large datasets

## 📝 Summary

These medium sorting problems focus on:

- **Custom Comparators** for complex sorting logic
- **Merge Operations** for interval and sequence problems  
- **Frequency-based Sorting** with counting and bucketing
- **Constrained Sorting** with space/pattern limitations
- **Hybrid Approaches** combining sorting with other techniques

Master these patterns to handle any sorting challenge in interviews and real-world applications!
