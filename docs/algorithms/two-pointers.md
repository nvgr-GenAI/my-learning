# Two Pointers Technique

## 🎯 Overview

The Two Pointers technique is a powerful algorithmic approach that uses two pointers to traverse data structures, typically arrays or linked lists. This technique is particularly effective for solving problems involving pairs, subarrays, or when you need to compare elements at different positions.

## 📋 Core Patterns

### **1. Opposite Direction (Converging)**
```python
def two_sum_sorted(arr, target):
    """Find pair that sums to target in sorted array"""
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return [-1, -1]
```

### **2. Same Direction (Fast & Slow)**
```python
def remove_duplicates(arr):
    """Remove duplicates from sorted array in-place"""
    if len(arr) <= 1:
        return len(arr)
    
    slow = 0  # Position for next unique element
    
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]
    
    return slow + 1
```

### **3. Sliding Window**
```python
def max_sum_subarray(arr, k):
    """Find maximum sum of subarray of size k"""
    if len(arr) < k:
        return -1
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

## 🏆 Classic Problems

### **Valid Palindrome**
```python
def is_palindrome(s):
    """Check if string is palindrome (ignoring non-alphanumeric)"""
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True
```

### **Three Sum**
```python
def three_sum(nums):
    """Find all unique triplets that sum to zero"""
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates for second and third numbers
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    
    return result
```

### **Container With Most Water**
```python
def max_area(height):
    """Find container that holds most water"""
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        # Calculate current area
        width = right - left
        current_height = min(height[left], height[right])
        current_area = width * current_height
        max_water = max(max_water, current_area)
        
        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water
```

## 🎯 When to Use Two Pointers

### **Problem Indicators**
- ✅ Array or string processing
- ✅ Looking for pairs or subarrays
- ✅ Need to compare elements at different positions
- ✅ Sorted array optimization opportunities
- ✅ In-place operations required

### **Common Patterns**
1. **Pair Finding** - Two Sum variations
2. **Palindrome Checking** - Compare from ends
3. **Array Partitioning** - Dutch National Flag
4. **Cycle Detection** - Floyd's algorithm
5. **Window Problems** - Sliding window technique

## 📊 Complexity Benefits

### **Time Complexity**
- **Before**: O(n²) nested loops
- **After**: O(n) single pass with two pointers

### **Space Complexity**
- **Before**: O(n) for hash tables or additional arrays
- **After**: O(1) constant extra space

## 🚀 Advanced Applications

### **Cycle Detection (Floyd's Algorithm)**
```python
def has_cycle(head):
    """Detect cycle in linked list using tortoise and hare"""
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while fast and fast.next:
        if slow == fast:
            return True
        slow = slow.next
        fast = fast.next.next
    
    return False
```

### **Dutch National Flag**
```python
def sort_colors(nums):
    """Sort array of 0s, 1s, and 2s in-place"""
    left = 0      # Next position for 0
    right = len(nums) - 1  # Next position for 2
    current = 0   # Current element being processed
    
    while current <= right:
        if nums[current] == 0:
            nums[left], nums[current] = nums[current], nums[left]
            left += 1
            current += 1
        elif nums[current] == 2:
            nums[current], nums[right] = nums[right], nums[current]
            right -= 1
            # Don't increment current (need to check swapped element)
        else:  # nums[current] == 1
            current += 1
```

## 💡 Pro Tips

### **1. Pointer Movement Strategy**
```python
# Move the pointer that gives you more information
if condition_A:
    left += 1
elif condition_B:
    right -= 1
else:
    # Move both or the one that makes sense
    left += 1
    right -= 1
```

### **2. Handling Duplicates**
```python
# Skip duplicates to avoid redundant work
while left < right and nums[left] == nums[left + 1]:
    left += 1
while left < right and nums[right] == nums[right - 1]:
    right -= 1
```

### **3. Boundary Conditions**
```python
# Always check bounds before accessing
while left < right:  # Prevent pointers from crossing
    # Process elements
    if some_condition:
        left += 1
    else:
        right -= 1
```

---

**Master the Two Pointers technique and you'll solve array and string problems with elegance and efficiency!**
