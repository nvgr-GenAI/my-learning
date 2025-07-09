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

## Sort Colors (Dutch Flag)

=== "🔍 Problem Statement"
    
        Given an array `nums` with `n` objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.
        
        We will use the integers 0, 1, and 2 to represent the colors red, white, and blue, respectively.
        
        **Example 1:**
        ```
        Input: nums = [2,0,2,1,1,0]
        Output: [0,0,1,1,2,2]
        ```
        
        **Example 2:**
        ```
        Input: nums = [2,0,1]
        Output: [0,1,2]
        ```
        
        **Constraints:**
        - n == nums.length
        - 1 <= n <= 300
        - nums[i] is either 0, 1, or 2
        
        **Follow-up:** Could you come up with a one-pass algorithm using only constant extra space?
    
    === "💡 Solution Approach"
    
        **Key Insight:**
        
        This is the Dutch National Flag problem, introduced by Edsger W. Dijkstra. It requires sorting an array containing only three distinct values in a single pass using constant space.
        
        **Step-by-step:**
        
        1. Use three pointers to divide the array into four regions:
           - `[0, left)`: all 0s (red)
           - `[left, curr)`: all 1s (white)
           - `[curr, right]`: unprocessed elements
           - `(right, end]`: all 2s (blue)
        
        2. Process each element at the `curr` pointer:
           - If it's 0, swap with the `left` boundary and increment both `left` and `curr`
           - If it's 1, just increment `curr`
           - If it's 2, swap with the `right` boundary and decrement `right` (don't increment `curr` as we need to check the swapped element)
        
        3. Continue until `curr` exceeds `right`
        
        **Why it works:**
        
        The algorithm maintains three regions of the array and gradually expands them as we process each element. By the end, all 0s are at the beginning, followed by all 1s, and then all 2s.
    
    === "💻 Implementation"
    
        ```python
        def sortColors(nums):
            """
            Sort array with values 0, 1, 2 in-place
            Dutch National Flag algorithm
            
            Time: O(n) - single pass through the array
            Space: O(1) - constant extra space
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
    
    === "🔄 Alternative Approaches"
    
        **Approach 1: Counting Sort (Two-Pass)**
        
        ```python
        def sortColors(nums):
            """
            Time: O(n) - two-pass solution
            Space: O(1) - constant extra space
            """
            # Count occurrences of each color
            counts = [0, 0, 0]
            for num in nums:
                counts[num] += 1
            
            # Fill the array with the right colors
            index = 0
            for color in range(3):
                for _ in range(counts[color]):
                    nums[index] = color
                    index += 1
        ```
        
        **Approach 2: Using Standard Sort**
        
        ```python
        def sortColors(nums):
            """
            Time: O(n log n)
            Space: O(log n) to O(n) depending on sort implementation
            """
            nums.sort()
        ```
    
    === "💭 Tips & Insights"
    
        **Interview Tips:**
        
        1. **One-pass vs. Two-pass:** The Dutch National Flag algorithm is the optimal solution as it meets the follow-up requirement of a one-pass solution with constant space.
        
        2. **Common Mistake:** Not handling the swapped element when swapping with the right pointer. After swapping with `right`, we need to check the new element at `curr` rather than moving forward.
        
        3. **Generalizing:** This approach can be extended to sort arrays with more than three distinct values using the "rainbow sort" technique.
        
        4. **Similar Problems:**
           - Sort Array By Parity (two-way partitioning)
           - Partition List (with linked lists)
           - Sort an array of 0s, 1s, 2s, and 3s
        
        **Real-world Applications:**
        
        - Image processing where pixels need to be organized by color
        - Memory defragmentation
        - Efficient partitioning in quicksort implementations
    
    ## Merge Intervals

=== "🔍 Problem Statement"

    Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.
    
    **Example 1:**
    ```
    Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]
    Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
    ```
    
    **Example 2:**
    ```
    Input: intervals = [[1,4],[4,5]]
    Output: [[1,5]]
    Explanation: Intervals [1,4] and [4,5] are considered overlapping.
    ```
    
    **Constraints:**
    - 1 <= intervals.length <= 10^4
    - intervals[i].length == 2
    - 0 <= starti <= endi <= 10^4
    
    === "💡 Solution Approach"
    
        **Key Insight:**
        
        The key to solving this problem is to first sort the intervals by their start times. After sorting, we can merge overlapping intervals by comparing the end time of the current merged interval with the start time of the next interval.
        
        **Step-by-step:**
        
        1. Sort intervals based on the start time
        2. Initialize a result list with the first interval
        3. Iterate through the remaining intervals:
           - If the current interval's start time is less than or equal to the end time of the last interval in the result, merge them by updating the end time
           - Otherwise, add the current interval to the result list
        
        **Code Implementation:**
        
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
    
    === "🔄 Alternative Approaches"
    
        **Approach: Using a Sweep Line Algorithm**
        
        ```python
        def merge(intervals):
            """
            Time: O(n log n)
            Space: O(n)
            """
            events = []
            
            # Create events for start and end points
            for start, end in intervals:
                events.append((start, 1))  # 1 for start event
                events.append((end, -1))   # -1 for end event
            
            # Sort events by position, with ties broken by event type (end before start)
            events.sort(key=lambda x: (x[0], x[1]))
            
            result = []
            count = 0
            start = None
            
            # Process events in order
            for pos, evt in events:
                if count == 0 and evt == 1:
                    # Start of a new merged interval
                    start = pos
                
                count += evt
                
                if count == 0:
                    # All intervals ended, add merged interval
                    result.append([start, pos])
            
            return result
        ```
    
    === "💭 Tips & Insights"
    
        **Interview Tips:**
        
        1. **Sorting is Key:** The problem becomes much simpler after sorting. Always consider sorting as a first step when dealing with interval problems.
        
        2. **Common Mistake:** Forgetting to update the end value when merging intervals. Make sure to take the maximum of the current end and the new interval's end.
        
        3. **Edge Cases:** 
           - Empty input array
           - Single interval
           - All intervals overlap
           - No intervals overlap
        
        4. **Follow-up Questions:**
           - How would you handle streaming intervals?
           - What if the intervals are already sorted?
           - What if the input doesn't fit in memory?
        
        **Real-world Applications:**
        
        - Calendar scheduling and meeting room allocation
        - Network packet coalescence
        - Memory allocation and defragmentation
        - Time interval analysis in logs and monitoring systems
    
    ## Largest Number

=== "🔍 Problem Statement"

    Given a list of non-negative integers `nums`, arrange them such that they form the largest number and return it.
    
    Since the result may be very large, so you need to return a string instead of an integer.
    
    **Example 1:**
    ```
    Input: nums = [10,2]
    Output: "210"
    ```
    
    **Example 2:**
    ```
    Input: nums = [3,30,34,5,9]
    Output: "9534330"
    ```
    
    **Constraints:**
    - 1 <= nums.length <= 100
    - 0 <= nums[i] <= 10^9
    
    === "💡 Solution Approach"
    
        **Key Insight:**
        
        The key observation is that for any two numbers, their contribution to the final result depends on their order. For example, should "34" come before "3" or after? We can determine this by comparing "343" vs "334" and seeing which is larger.
        
        **Step-by-step:**
        
        1. Convert all numbers to strings
        2. Create a custom comparator that compares concatenation of strings in different orders
        3. Sort the strings using this custom comparator
        4. Join the sorted strings to form the result
        5. Handle edge case of all zeros
        
        **Code Implementation:**
        
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
    
    === "🔄 Alternative Approaches"
    
        **Approach: Using a Custom Class with __lt__ Method**
        
        ```python
        def largestNumber(nums):
            """
            Time: O(n log n)
            Space: O(n)
            """
            # Define custom comparison class
            class LargerNumKey(str):
                def __lt__(self, other):
                    return self + other > other + self
            
            # Convert numbers to strings and sort
            str_nums = map(str, nums)
            result = ''.join(sorted(str_nums, key=LargerNumKey))
            
            # Handle edge case of all zeros
            return '0' if result[0] == '0' else result
        ```
    
    === "💭 Tips & Insights"
    
        **Interview Tips:**
        
        1. **Custom Comparator:** This problem is a great example of when to use a custom comparator for sorting. The comparison logic here is not intuitive but makes perfect sense for this problem.
        
        2. **Key Insight:** For any two numbers a and b, we need to determine whether a should come before b by comparing the concatenated strings "ab" and "ba".
        
        3. **Edge Cases:** 
           - All zeros (output should be "0" not "000...")
           - Mixed numbers with zeros (e.g., [0, 0, 30])
           - Single element array
        
        4. **Language-Specific Notes:**
           - In Python 3, the `cmp` parameter for sort is removed, so we use `cmp_to_key` from functools
           - In Java, you can implement a custom Comparator
           - In JavaScript, you can provide a comparison function to sort()
        
        **Real-world Applications:**
        
        - Number formatting in user interfaces
        - Sorting for maximum revenue in advertising (placing highest-value ads first)
        - Priority queuing where concatenation order matters
    
    ## Top K Frequent Elements

=== "🔍 Problem Statement"

    Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in any order.
    
    **Example 1:**
    ```
    Input: nums = [1,1,1,2,2,3], k = 2
    Output: [1,2]
    ```
    
    **Example 2:**
    ```
    Input: nums = [1], k = 1
    Output: [1]
    ```
    
    **Constraints:**
    - 1 <= nums.length <= 10^5
    - -10^4 <= nums[i] <= 10^4
    - k is in the range [1, the number of unique elements in the array]
    - It is guaranteed that the answer is unique
    
    **Follow up:** Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
    
    === "💡 Solution Approach"
    
        **Key Insight:**
        
        We can use a hash map to count frequencies and then use a heap to efficiently find the top k elements. Alternatively, we can use bucket sort for an O(n) solution.
        
        **Step-by-step:**
        
        1. Use a Counter to count the frequency of each element
        2. Use a heap to get the k most frequent elements based on their counts
        
        **Code Implementation:**
        
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
    
    === "🔄 Alternative Approaches"
    
        **Approach: Bucket Sort (O(n) solution)**
        
        ```python
        from collections import Counter
        
        def topKFrequent(nums, k):
            """
            Time: O(n)
            Space: O(n)
            """
            counter = Counter(nums)
            
            # Create buckets - index is frequency, value is list of numbers with that frequency
            buckets = [[] for _ in range(len(nums) + 1)]
            
            # Place numbers in appropriate buckets based on their frequency
            for num, freq in counter.items():
                buckets[freq].append(num)
            
            # Collect results starting from the highest frequency
            result = []
            for i in range(len(buckets) - 1, -1, -1):
                if buckets[i]:
                    result.extend(buckets[i])
                if len(result) >= k:
                    return result[:k]
                    
            return result  # Should never reach here given the constraints
        ```
        
        **Approach: Quick Select**
        
        ```python
        from collections import Counter
        import random
        
        def topKFrequent(nums, k):
            """
            Time: Average O(n), Worst O(n²)
            Space: O(n)
            """
            count = Counter(nums)
            unique = list(count.keys())
            
            def partition(left, right, pivot_index):
                pivot_freq = count[unique[pivot_index]]
                # Move pivot to end
                unique[pivot_index], unique[right] = unique[right], unique[pivot_index]
                
                # Move all less frequent elements to the left
                store_index = left
                for i in range(left, right):
                    if count[unique[i]] < pivot_freq:
                        unique[store_index], unique[i] = unique[i], unique[store_index]
                        store_index += 1
                
                # Move pivot to its final place
                unique[right], unique[store_index] = unique[store_index], unique[right]
                
                return store_index
            
            def quick_select(left, right, k_smallest):
                if left == right:
                    return
                
                # Select a random pivot
                pivot_index = random.randint(left, right)
                
                # Find the pivot position in a sorted array
                pivot_index = partition(left, right, pivot_index)
                
                # If the pivot is in its final sorted position
                if k_smallest == pivot_index:
                    return
                elif k_smallest < pivot_index:
                    # Go left
                    quick_select(left, pivot_index - 1, k_smallest)
                else:
                    # Go right
                    quick_select(pivot_index + 1, right, k_smallest)
            
            n = len(unique)
            # Find the kth least frequent element
            quick_select(0, n - 1, n - k)
            
            # Return the k most frequent elements
            return unique[n - k:]
        ```
    
    === "💭 Tips & Insights"
    
        **Interview Tips:**
        
        1. **Frequency Counting:** Almost all solutions start with counting frequencies using a hash map.
        
        2. **Multiple Solutions:** Be prepared to discuss the trade-offs between:
           - Heap-based solution: O(n log k) time
           - Bucket sort: O(n) time but requires extra space
           - Quick select: Average O(n) time but worst case O(n²)
        
        3. **Follow-up:** If asked about O(n log n) solutions, you should mention that sorting all elements by frequency would work but is not optimal.
        
        4. **Edge Cases:**
           - All elements appear with equal frequency
           - k equals the number of unique elements
           - Array contains only one unique element
        
        **Real-world Applications:**
        
        - Search engine result ranking
        - Cache eviction policies (LFU - Least Frequently Used)
        - Data compression algorithms
        - Recommendation systems based on user behavior frequency
        - Text analysis for identifying key terms
    
    ## Meeting Rooms II

=== "🔍 Problem Statement"

    Given an array of meeting time intervals where intervals[i] = [starti, endi], find the minimum number of conference rooms required.
    
    **Example 1:**
    ```
    Input: intervals = [[0,30],[5,10],[15,20]]
    Output: 2
    Explanation: We need two meeting rooms:
    - Room 1: [0,30]
    - Room 2: [5,10], [15,20]
    ```
    
    **Example 2:**
    ```
    Input: intervals = [[7,10],[2,4]]
    Output: 1
    Explanation: We need only one meeting room since the meetings don't overlap.
    ```
    
    **Constraints:**
    - 1 <= intervals.length <= 10^4
    - 0 <= starti < endi <= 10^6
    
    === "💡 Solution Approach"
    
        **Key Insight:**
        
        The key insight is to track the number of ongoing meetings at each point in time. We need a new room when a new meeting starts while other meetings are still ongoing.
        
        **Step-by-step:**
        
        1. Sort the meetings by their start time
        2. Use a min heap to keep track of end times of ongoing meetings
        3. For each meeting:
           - If the earliest ending meeting has already ended (end time <= current meeting's start time), reuse that room (pop from heap)
           - Add the current meeting's end time to the heap
        4. The size of the heap at the end represents the minimum number of rooms needed
        
        **Code Implementation:**
        
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
    
    === "🔄 Alternative Approaches"
    
        **Approach: Using Chronological Ordering (Sweep Line)**
        
        ```python
        def minMeetingRooms(intervals):
            """
            Time: O(n log n)
            Space: O(n)
            """
            # Create start and end time arrays
            start_times = sorted([interval[0] for interval in intervals])
            end_times = sorted([interval[1] for interval in intervals])
            
            rooms = 0
            max_rooms = 0
            start_ptr = 0
            end_ptr = 0
            
            # Process events chronologically
            while start_ptr < len(intervals):
                if start_times[start_ptr] < end_times[end_ptr]:
                    # A meeting starts
                    rooms += 1
                    start_ptr += 1
                else:
                    # A meeting ends
                    rooms -= 1
                    end_ptr += 1
                
                max_rooms = max(max_rooms, rooms)
                
            return max_rooms
        ```
        
        **Approach: Using a Priority Queue for Both Start and End Times**
        
        ```python
        import heapq
        
        def minMeetingRooms(intervals):
            """
            Time: O(n log n)
            Space: O(n)
            """
            if not intervals:
                return 0
                
            # Create events with type (0 for start, 1 for end)
            events = []
            for start, end in intervals:
                events.append((start, 0))  # Start event
                events.append((end, 1))    # End event
                
            # Sort by time, then by event type (end before start if same time)
            events.sort()
            
            rooms = 0
            max_rooms = 0
            
            for time, event_type in events:
                if event_type == 0:  # Start event
                    rooms += 1
                else:  # End event
                    rooms -= 1
                
                max_rooms = max(max_rooms, rooms)
                
            return max_rooms
        ```
    
    === "💭 Tips & Insights"
    
        **Interview Tips:**
        
        1. **Problem Essence:** This is fundamentally a resource allocation problem that requires tracking overlapping intervals.
        
        2. **Key Insight:** The minimum number of rooms needed equals the maximum number of overlapping meetings at any point in time.
        
        3. **Algorithm Choice:**
           - Min heap approach is intuitive and straightforward to explain
           - Sweep line approach is more elegant but requires careful implementation
        
        4. **Edge Cases:**
           - No meetings
           - All meetings at different times (no overlaps)
           - All meetings at the same time (maximum overlap)
           - Back-to-back meetings (exactly touching)
        
        5. **Follow-up Questions:**
           - How would you assign specific rooms to meetings?
           - How would you handle real-time meeting requests?
           - What if meetings have different priorities?
        
        **Real-world Applications:**
        
        - Conference room scheduling systems
        - Resource allocation in operating systems
        - Airport runway/gate allocation
        - Time slot allocation in broadcasting
        - Processor task scheduling
    
    ## Queue Reconstruction by Height

=== "🔍 Problem Statement"

    You are given an array of people, people, which are the attributes of some people in a queue (not necessarily in order). Each people[i] = [hi, ki] represents the ith person of height hi with exactly ki other people in front who have a height greater than or equal to hi.

    Reconstruct and return the queue that is represented by the input array people. The returned queue should be formatted as an array queue, where queue[j] = [hj, kj] is the attributes of the jth person in the queue (queue[0] is the person at the front of the queue).

    **Example 1:**
    ```
    Input: people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
    Output: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
    Explanation:
    Person 0: [5,0] has height 5 with no one taller in front.
    Person 1: [7,0] has height 7 with no one taller in front.
    Person 2: [5,2] has height 5 with two persons taller in front (Person 1 and 3).
    Person 3: [6,1] has height 6 with one person taller in front (Person 1).
    Person 4: [4,4] has height 4 with four people taller in front (Person 0, 1, 2, and 3).
    Person 5: [7,1] has height 7 with one person taller in front (Person 1).
    ```

    **Example 2:**
    ```
    Input: people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
    Output: [[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]
    ```

    **Constraints:**
    - 1 <= people.length <= 2000
    - 0 <= hi <= 10^6
    - 0 <= ki < people.length

=== "💡 Implementation"

    ```python
    def reconstructQueue(people):
        """
        Reconstruct queue based on height and position
        
        Time Complexity: O(n²) due to the insertion operation
        Space Complexity: O(n) for the result array and sorting
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
