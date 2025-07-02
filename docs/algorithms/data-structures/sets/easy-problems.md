# Set Easy Problems 🟢

## 🎯 Overview

Practice fundamental set operations and algorithms. These problems focus on basic set manipulations, membership testing, and simple applications.

## 🔧 Problem Categories

### 1. Basic Set Operations
- Intersection and union of arrays
- Duplicate detection  
- Membership testing

### 2. Hash Set Applications
- Unique character checking
- Simple deduplication
- Fast lookups

### 3. Set-based Counting
- Counting unique elements
- Finding missing numbers
- Simple frequency problems

---

## 📝 Problems

### Problem 1: Contains Duplicate

**Problem**: Given an integer array, return true if any value appears at least twice.

```python
def contains_duplicate(nums):
    """
    Check if array contains duplicates
    Time: O(n), Space: O(n)
    """
    seen = set()
    
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    
    return False

# Alternative one-liner
def contains_duplicate_oneliner(nums):
    return len(nums) != len(set(nums))

# Test
print(contains_duplicate([1, 2, 3, 1]))      # True
print(contains_duplicate([1, 2, 3, 4]))      # False
```

### Problem 2: Intersection of Two Arrays

**Problem**: Find the intersection of two arrays.

```python
def intersection(nums1, nums2):
    """
    Find intersection without duplicates
    Time: O(n + m), Space: O(min(n, m))
    """
    set1 = set(nums1)
    result = set()
    
    for num in nums2:
        if num in set1:
            result.add(num)
    
    return list(result)

def intersection_ii(nums1, nums2):
    """
    Find intersection with duplicates allowed
    Time: O(n + m), Space: O(min(n, m))
    """
    from collections import Counter
    
    count1 = Counter(nums1)
    result = []
    
    for num in nums2:
        if count1[num] > 0:
            result.append(num)
            count1[num] -= 1
    
    return result

# Test
nums1 = [1, 2, 2, 1]
nums2 = [2, 2]
print(intersection(nums1, nums2))        # [2]
print(intersection_ii(nums1, nums2))     # [2, 2]
```

### Problem 3: Single Number

**Problem**: Find the number that appears only once while others appear twice.

```python
def single_number(nums):
    """
    Find single number using XOR
    Time: O(n), Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result

def single_number_set(nums):
    """
    Find single number using set
    Time: O(n), Space: O(n)
    """
    seen = set()
    
    for num in nums:
        if num in seen:
            seen.remove(num)
        else:
            seen.add(num)
    
    return seen.pop()

# Test
print(single_number([2, 2, 1]))          # 1
print(single_number([4, 1, 2, 1, 2]))    # 4
```

### Problem 4: Happy Number

**Problem**: Determine if a number is happy (repeatedly replace by sum of squares of digits until it becomes 1).

```python
def is_happy(n):
    """
    Check if number is happy using set to detect cycles
    Time: O(log n), Space: O(log n)
    """
    def get_sum_of_squares(num):
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total
    
    seen = set()
    
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_sum_of_squares(n)
    
    return n == 1

# Test
print(is_happy(19))  # True (19 -> 82 -> 68 -> 100 -> 1)
print(is_happy(2))   # False
```

### Problem 5: Missing Number

**Problem**: Find the missing number in array containing distinct numbers from 0 to n.

```python
def missing_number_set(nums):
    """
    Find missing number using set
    Time: O(n), Space: O(n)
    """
    n = len(nums)
    expected = set(range(n + 1))
    actual = set(nums)
    
    return (expected - actual).pop()

def missing_number_math(nums):
    """
    Find missing number using math formula
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum

def missing_number_xor(nums):
    """
    Find missing number using XOR
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    result = n
    
    for i, num in enumerate(nums):
        result ^= i ^ num
    
    return result

# Test
print(missing_number_set([3, 0, 1]))     # 2
print(missing_number_math([0, 1]))       # 2
print(missing_number_xor([9,6,4,2,3,5,7,0,1]))  # 8
```

### Problem 6: Unique Characters in String

**Problem**: Check if a string has all unique characters.

```python
def is_unique_chars(s):
    """
    Check if string has all unique characters
    Time: O(n), Space: O(k) where k = unique chars
    """
    char_set = set()
    
    for char in s:
        if char in char_set:
            return False
        char_set.add(char)
    
    return True

def is_unique_chars_oneliner(s):
    """One-liner solution"""
    return len(s) == len(set(s))

def is_unique_chars_no_extra_space(s):
    """
    Without extra space (assuming ASCII)
    Time: O(n), Space: O(1)
    """
    if len(s) > 128:  # ASCII has 128 characters
        return False
    
    char_flags = [False] * 128
    
    for char in s:
        char_code = ord(char)
        if char_flags[char_code]:
            return False
        char_flags[char_code] = True
    
    return True

# Test
print(is_unique_chars("abcdef"))         # True
print(is_unique_chars("hello"))          # False
```

### Problem 7: First Unique Character

**Problem**: Find the first non-repeating character in a string.

```python
def first_unique_char(s):
    """
    Find index of first unique character
    Time: O(n), Space: O(1) - limited alphabet
    """
    from collections import Counter
    
    char_count = Counter(s)
    
    for i, char in enumerate(s):
        if char_count[char] == 1:
            return i
    
    return -1

def first_unique_char_two_pass(s):
    """
    Two-pass solution with set
    Time: O(n), Space: O(1)
    """
    # First pass: mark duplicates
    seen_once = set()
    seen_multiple = set()
    
    for char in s:
        if char in seen_once:
            seen_multiple.add(char)
        else:
            seen_once.add(char)
    
    # Second pass: find first unique
    for i, char in enumerate(s):
        if char not in seen_multiple:
            return i
    
    return -1

# Test
print(first_unique_char("leetcode"))     # 0 ('l')
print(first_unique_char("loveleetcode")) # 2 ('v')
```

### Problem 8: Jewels and Stones

**Problem**: Count how many stones are jewels.

```python
def num_jewels_in_stones(jewels, stones):
    """
    Count jewels in stones
    Time: O(j + s), Space: O(j)
    """
    jewel_set = set(jewels)
    count = 0
    
    for stone in stones:
        if stone in jewel_set:
            count += 1
    
    return count

def num_jewels_in_stones_oneliner(jewels, stones):
    """One-liner solution"""
    return sum(stone in set(jewels) for stone in stones)

# Test
print(num_jewels_in_stones("aA", "aAAbbbb"))     # 3
print(num_jewels_in_stones("z", "ZZ"))           # 0
```

### Problem 9: Buddy Strings

**Problem**: Check if you can swap exactly two letters to make two strings equal.

```python
def buddy_strings(s, goal):
    """
    Check if two strings can be made equal with one swap
    Time: O(n), Space: O(1)
    """
    if len(s) != len(goal):
        return False
    
    if s == goal:
        # Check if any character appears more than once
        return len(set(s)) < len(s)
    
    # Find differences
    diffs = []
    for i in range(len(s)):
        if s[i] != goal[i]:
            diffs.append(i)
    
    # Must have exactly 2 differences that can be swapped
    return (len(diffs) == 2 and 
            s[diffs[0]] == goal[diffs[1]] and 
            s[diffs[1]] == goal[diffs[0]])

# Test
print(buddy_strings("ab", "ba"))         # True
print(buddy_strings("ab", "ab"))         # False
print(buddy_strings("aa", "aa"))         # True
```

### Problem 10: Unique Email Addresses

**Problem**: Count unique email addresses after applying rules.

```python
def num_unique_emails(emails):
    """
    Count unique email addresses
    Time: O(n * m), Space: O(n * m)
    """
    unique_emails = set()
    
    for email in emails:
        local, domain = email.split('@')
        
        # Remove everything after +
        if '+' in local:
            local = local[:local.index('+')]
        
        # Remove dots from local name
        local = local.replace('.', '')
        
        # Add cleaned email to set
        unique_emails.add(local + '@' + domain)
    
    return len(unique_emails)

# Test
emails = [
    "test.email+tag@leetcode.com",
    "test.e.mail+tag.one@leetcode.com",
    "testemail+tag@leetcode.com"
]
print(num_unique_emails(emails))  # 2
```

## 🎯 Key Patterns

### Pattern 1: Duplicate Detection
```python
# Use set to track seen elements
seen = set()
for item in items:
    if item in seen:
        return True  # Found duplicate
    seen.add(item)
```

### Pattern 2: Intersection/Union
```python
# Convert to sets for fast operations
set1, set2 = set(arr1), set(arr2)
intersection = set1 & set2
union = set1 | set2
difference = set1 - set2
```

### Pattern 3: Unique Counting
```python
# Count unique elements
unique_count = len(set(array))

# Find missing/extra elements
expected = set(range(n))
actual = set(array)
missing = expected - actual
```

## 📈 Time Complexities

- **Hash Set Operations**: O(1) average, O(n) worst case
- **Set Creation**: O(n) where n = number of elements
- **Set Operations**: O(min(len(set1), len(set2))) for intersection
- **Membership Testing**: O(1) average in hash set

## 🚀 Next Level

Ready for more challenging problems? Try:
- **[Medium Problems](medium-problems.md)** - Set algorithms and optimizations
- **[Hard Problems](hard-problems.md)** - Complex set applications
