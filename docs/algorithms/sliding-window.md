# Sliding Window Technique

## 🎯 Overview

The Sliding Window technique is used to perform required operations on a specific window size of an array or string. This technique is particularly useful for problems involving subarrays or substrings with certain properties, reducing time complexity from O(n²) to O(n).

## 📋 Core Patterns

### **1. Fixed Window Size**
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

### **2. Variable Window Size**
```python
def longest_substring_k_distinct(s, k):
    """Find longest substring with at most k distinct characters"""
    if k == 0:
        return 0
    
    char_count = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Expand window
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Contract window if necessary
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        # Update max length
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

### **3. Shrinking Window**
```python
def min_window_substring(s, t):
    """Find minimum window substring containing all characters of t"""
    if not s or not t:
        return ""
    
    # Count characters in t
    t_count = {}
    for char in t:
        t_count[char] = t_count.get(char, 0) + 1
    
    left = 0
    min_length = float('inf')
    min_start = 0
    formed = 0  # Number of unique chars in window with desired frequency
    window_count = {}
    
    for right in range(len(s)):
        # Add character to window
        char = s[right]
        window_count[char] = window_count.get(char, 0) + 1
        
        if char in t_count and window_count[char] == t_count[char]:
            formed += 1
        
        # Try to contract window
        while left <= right and formed == len(t_count):
            # Update minimum window
            if right - left + 1 < min_length:
                min_length = right - left + 1
                min_start = left
            
            # Remove character from left
            left_char = s[left]
            window_count[left_char] -= 1
            if left_char in t_count and window_count[left_char] < t_count[left_char]:
                formed -= 1
            
            left += 1
    
    return "" if min_length == float('inf') else s[min_start:min_start + min_length]
```

## 🏆 Classic Problems

### **Longest Substring Without Repeating Characters**
```python
def length_of_longest_substring(s):
    """Find length of longest substring without repeating characters"""
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Shrink window until no duplicates
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        # Add current character
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

### **Fruits Into Baskets**
```python
def total_fruit(fruits):
    """Maximum fruits you can pick with 2 baskets (each basket holds one type)"""
    basket = {}
    left = 0
    max_fruits = 0
    
    for right in range(len(fruits)):
        # Add fruit to basket
        basket[fruits[right]] = basket.get(fruits[right], 0) + 1
        
        # If more than 2 types, shrink window
        while len(basket) > 2:
            basket[fruits[left]] -= 1
            if basket[fruits[left]] == 0:
                del basket[fruits[left]]
            left += 1
        
        max_fruits = max(max_fruits, right - left + 1)
    
    return max_fruits
```

### **Permutation in String**
```python
def check_inclusion(s1, s2):
    """Check if any permutation of s1 is substring of s2"""
    if len(s1) > len(s2):
        return False
    
    # Count characters in s1
    s1_count = {}
    for char in s1:
        s1_count[char] = s1_count.get(char, 0) + 1
    
    window_size = len(s1)
    window_count = {}
    
    # Initialize first window
    for i in range(window_size):
        char = s2[i]
        window_count[char] = window_count.get(char, 0) + 1
    
    # Check first window
    if window_count == s1_count:
        return True
    
    # Slide window
    for i in range(window_size, len(s2)):
        # Add new character
        new_char = s2[i]
        window_count[new_char] = window_count.get(new_char, 0) + 1
        
        # Remove old character
        old_char = s2[i - window_size]
        window_count[old_char] -= 1
        if window_count[old_char] == 0:
            del window_count[old_char]
        
        # Check if permutation found
        if window_count == s1_count:
            return True
    
    return False
```

## 🎯 Window Types

### **1. Fast Expand, Slow Contract**
```python
def max_consecutive_ones_iii(nums, k):
    """Max consecutive 1s after flipping at most k zeros"""
    left = 0
    zeros_count = 0
    max_length = 0
    
    for right in range(len(nums)):
        if nums[right] == 0:
            zeros_count += 1
        
        # Contract window if too many zeros
        while zeros_count > k:
            if nums[left] == 0:
                zeros_count -= 1
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

### **2. Grow Until Invalid**
```python
def character_replacement(s, k):
    """Longest substring with same characters after k replacements"""
    char_count = {}
    left = 0
    max_freq = 0
    max_length = 0
    
    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        max_freq = max(max_freq, char_count[s[right]])
        
        # Window size - most frequent char > k means invalid
        window_size = right - left + 1
        if window_size - max_freq > k:
            char_count[s[left]] -= 1
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

## 📊 Complexity Analysis

### **Time Complexity**
- **Fixed Window**: O(n) - each element processed once
- **Variable Window**: O(n) - each element added/removed at most once
- **Optimization**: From O(n²) brute force to O(n)

### **Space Complexity**
- **Hash Map**: O(k) where k is unique elements in window
- **Array**: O(1) for character counting (fixed alphabet size)

## 🔧 Implementation Patterns

### **Template for Fixed Window**
```python
def fixed_window_template(arr, k):
    if len(arr) < k:
        return []
    
    # Process first window
    window_value = process_initial_window(arr, k)
    result = [window_value]
    
    # Slide window
    for i in range(k, len(arr)):
        # Remove element going out of window
        remove_element(arr[i - k])
        # Add element coming into window
        add_element(arr[i])
        # Update result
        result.append(get_current_window_value())
    
    return result
```

### **Template for Variable Window**
```python
def variable_window_template(arr, condition):
    left = 0
    result = initialize_result()
    
    for right in range(len(arr)):
        # Expand window
        add_to_window(arr[right])
        
        # Contract window while condition violated
        while window_invalid(condition):
            remove_from_window(arr[left])
            left += 1
        
        # Update result with current valid window
        update_result(result, right - left + 1)
    
    return result
```

## 💡 Pro Tips

### **1. Two Pointers vs Sliding Window**
- **Two Pointers**: Compare elements at different positions
- **Sliding Window**: Maintain a range/subarray with specific properties

### **2. When to Use Sliding Window**
- ✅ Contiguous subarray/substring problems
- ✅ Find optimal window size
- ✅ All subarrays of size k
- ✅ Longest/shortest subarray with condition

### **3. Common Mistakes**
- ❌ Forgetting to update window boundaries
- ❌ Not handling empty windows
- ❌ Incorrect expansion/contraction logic

### **4. Optimization Tricks**
```python
# Use arrays for character counting (faster than hash maps)
char_count = [0] * 26  # for lowercase letters
char_count[ord(char) - ord('a')] += 1

# Early termination
if current_window_optimal:
    return result

# Batch processing for fixed windows
def batch_slide_window(arr, k, step_size):
    # Process multiple elements at once
    pass
```

---

**Master the Sliding Window technique to efficiently solve subarray and substring problems!**
