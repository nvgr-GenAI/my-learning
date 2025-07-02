# Set Medium Problems 🟡

## 🎯 Overview

Advance your set skills with more complex algorithms and optimization techniques. These problems require deeper understanding of set operations and their applications.

## 🔧 Problem Categories

### 1. Advanced Set Operations
- Complex intersection and union scenarios
- Set-based graph algorithms
- Multi-set problems

### 2. Optimization with Sets  
- Space-time trade-offs
- Sliding window with sets
- Set-based dynamic programming

### 3. String Processing with Sets
- Anagram problems
- Pattern matching with sets
- Substring problems

---

## 📝 Problems

### Problem 1: Longest Substring Without Repeating Characters

**Problem**: Find the length of the longest substring without repeating characters.

```python
def length_of_longest_substring(s):
    """
    Sliding window with set
    Time: O(n), Space: O(min(m,n)) where m = charset size
    """
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

def length_of_longest_substring_optimized(s):
    """
    Optimized sliding window with hash map
    Time: O(n), Space: O(min(m,n))
    """
    char_index = {}
    left = 0
    max_length = 0
    
    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        
        char_index[char] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Test
print(length_of_longest_substring("abcabcbb"))    # 3 ("abc")
print(length_of_longest_substring("bbbbb"))       # 1 ("b")
print(length_of_longest_substring("pwwkew"))      # 3 ("wke")
```

### Problem 2: Group Anagrams

**Problem**: Group strings that are anagrams of each other.

```python
def group_anagrams(strs):
    """
    Group anagrams using sorted string as key
    Time: O(n * k log k), Space: O(n * k)
    """
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())

def group_anagrams_frequency(strs):
    """
    Group anagrams using character frequency
    Time: O(n * k), Space: O(n * k)
    """
    from collections import defaultdict, Counter
    
    groups = defaultdict(list)
    
    for s in strs:
        # Use character count as key
        count = Counter(s)
        # Convert to sorted tuple for hashing
        key = tuple(sorted(count.items()))
        groups[key].append(s)
    
    return list(groups.values())

# Test
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_anagrams(strs))
# [["eat","tea","ate"],["tan","nat"],["bat"]]
```

### Problem 3: Set Matrix Zeroes

**Problem**: Set entire row and column to zero if an element is zero.

```python
def set_zeroes(matrix):
    """
    Set matrix zeroes using sets to track rows/cols
    Time: O(m*n), Space: O(m+n)
    """
    if not matrix or not matrix[0]:
        return
    
    rows, cols = len(matrix), len(matrix[0])
    zero_rows = set()
    zero_cols = set()
    
    # Find all zero positions
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0:
                zero_rows.add(i)
                zero_cols.add(j)
    
    # Set rows to zero
    for row in zero_rows:
        for j in range(cols):
            matrix[row][j] = 0
    
    # Set columns to zero
    for col in zero_cols:
        for i in range(rows):
            matrix[i][col] = 0

def set_zeroes_optimized(matrix):
    """
    Space-optimized version using first row/column as markers
    Time: O(m*n), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return
    
    rows, cols = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(cols))
    first_col_zero = any(matrix[i][0] == 0 for i in range(rows))
    
    # Use first row and column as markers
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][j] == 0:
                matrix[0][j] = 0
                matrix[i][0] = 0
    
    # Set internal elements to zero
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[0][j] == 0 or matrix[i][0] == 0:
                matrix[i][j] = 0
    
    # Handle first row and column
    if first_row_zero:
        for j in range(cols):
            matrix[0][j] = 0
    
    if first_col_zero:
        for i in range(rows):
            matrix[i][0] = 0

# Test
matrix = [[1,1,1],[1,0,1],[1,1,1]]
set_zeroes(matrix)
print(matrix)  # [[1,0,1],[0,0,0],[1,0,1]]
```

### Problem 4: Word Pattern

**Problem**: Check if a string follows a given pattern.

```python
def word_pattern(pattern, s):
    """
    Check if string follows pattern using bijection
    Time: O(n), Space: O(n)
    """
    words = s.split()
    
    if len(pattern) != len(words):
        return False
    
    char_to_word = {}
    word_to_char = {}
    
    for char, word in zip(pattern, words):
        # Check char -> word mapping
        if char in char_to_word:
            if char_to_word[char] != word:
                return False
        else:
            char_to_word[char] = word
        
        # Check word -> char mapping
        if word in word_to_char:
            if word_to_char[word] != char:
                return False
        else:
            word_to_char[word] = char
    
    return True

def word_pattern_set(pattern, s):
    """
    Alternative using sets
    Time: O(n), Space: O(n)
    """
    words = s.split()
    
    if len(pattern) != len(words):
        return False
    
    # Check if mapping is bijective
    return (len(set(pattern)) == len(set(words)) == 
            len(set(zip(pattern, words))))

# Test
print(word_pattern("abba", "dog cat cat dog"))    # True
print(word_pattern("abba", "dog cat cat fish"))   # False
```

### Problem 5: Find All Numbers Disappeared in Array

**Problem**: Find numbers missing from array of integers 1 to n.

```python
def find_disappeared_numbers_set(nums):
    """
    Find missing numbers using set difference
    Time: O(n), Space: O(n)
    """
    n = len(nums)
    expected = set(range(1, n + 1))
    actual = set(nums)
    
    return list(expected - actual)

def find_disappeared_numbers_inplace(nums):
    """
    Find missing numbers using array as hash set
    Time: O(n), Space: O(1)
    """
    # Mark numbers as seen by negating value at index
    for num in nums:
        index = abs(num) - 1
        if nums[index] > 0:
            nums[index] = -nums[index]
    
    # Find unmarked positions
    result = []
    for i in range(len(nums)):
        if nums[i] > 0:
            result.append(i + 1)
    
    return result

# Test
print(find_disappeared_numbers_set([4,3,2,7,8,2,3,1]))  # [5,6]
```

### Problem 6: Subsets

**Problem**: Generate all possible subsets of a set.

```python
def subsets_iterative(nums):
    """
    Generate subsets iteratively
    Time: O(2^n * n), Space: O(2^n * n)
    """
    result = [[]]
    
    for num in nums:
        # Add current number to all existing subsets
        new_subsets = []
        for subset in result:
            new_subsets.append(subset + [num])
        result.extend(new_subsets)
    
    return result

def subsets_backtrack(nums):
    """
    Generate subsets using backtracking
    Time: O(2^n * n), Space: O(2^n * n)
    """
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

def subsets_bit_manipulation(nums):
    """
    Generate subsets using bit manipulation
    Time: O(2^n * n), Space: O(2^n * n)
    """
    n = len(nums)
    result = []
    
    for mask in range(1 << n):  # 2^n possibilities
        subset = []
        for i in range(n):
            if mask & (1 << i):  # Check if bit i is set
                subset.append(nums[i])
        result.append(subset)
    
    return result

# Test
print(subsets_iterative([1,2,3]))
# [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]
```

### Problem 7: 3Sum

**Problem**: Find all unique triplets that sum to zero.

```python
def three_sum(nums):
    """
    Find unique triplets that sum to zero
    Time: O(n^2), Space: O(1) not counting output
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, n - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if current_sum == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1
    
    return result

def three_sum_set(nums):
    """
    Alternative using set for result deduplication
    Time: O(n^2), Space: O(n)
    """
    if len(nums) < 3:
        return []
    
    nums.sort()
    result_set = set()
    
    for i in range(len(nums) - 2):
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if current_sum == 0:
                result_set.add((nums[i], nums[left], nums[right]))
                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1
    
    return [list(triplet) for triplet in result_set]

# Test
print(three_sum([-1, 0, 1, 2, -1, -4]))
# [[-1, -1, 2], [-1, 0, 1]]
```

### Problem 8: Valid Sudoku

**Problem**: Check if a Sudoku board is valid.

```python
def is_valid_sudoku(board):
    """
    Validate Sudoku using sets
    Time: O(1) - fixed 9x9 board, Space: O(1)
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    
    for i in range(9):
        for j in range(9):
            val = board[i][j]
            
            if val == '.':
                continue
            
            # Calculate box index
            box_index = (i // 3) * 3 + j // 3
            
            # Check if number already exists
            if (val in rows[i] or 
                val in cols[j] or 
                val in boxes[box_index]):
                return False
            
            # Add to sets
            rows[i].add(val)
            cols[j].add(val)
            boxes[box_index].add(val)
    
    return True

def is_valid_sudoku_one_set(board):
    """
    Validate Sudoku using single set with encoded positions
    Time: O(1), Space: O(1)
    """
    seen = set()
    
    for i in range(9):
        for j in range(9):
            val = board[i][j]
            
            if val != '.':
                # Encode position information
                row_key = f"row{i}-{val}"
                col_key = f"col{j}-{val}"
                box_key = f"box{i//3}{j//3}-{val}"
                
                if (row_key in seen or 
                    col_key in seen or 
                    box_key in seen):
                    return False
                
                seen.add(row_key)
                seen.add(col_key)
                seen.add(box_key)
    
    return True

# Test
board = [
    ["5","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]
]
print(is_valid_sudoku(board))  # True
```

### Problem 9: Random Pick with Blacklist

**Problem**: Pick random number from range excluding blacklisted numbers.

```python
import random

class Solution:
    def __init__(self, n, blacklist):
        """
        Initialize with range [0, n) and blacklist
        Time: O(b), Space: O(b) where b = len(blacklist)
        """
        self.n = n
        self.blacklist_set = set(blacklist)
        
        # Map blacklisted numbers in range [0, n-len(blacklist))
        # to valid numbers in range [n-len(blacklist), n)
        self.valid_size = n - len(blacklist)
        self.mapping = {}
        
        # Find valid numbers in the upper range
        valid_upper = []
        for i in range(self.valid_size, n):
            if i not in self.blacklist_set:
                valid_upper.append(i)
        
        # Map blacklisted numbers in lower range to valid upper numbers
        valid_index = 0
        for num in blacklist:
            if num < self.valid_size:
                self.mapping[num] = valid_upper[valid_index]
                valid_index += 1
    
    def pick(self):
        """
        Pick random valid number
        Time: O(1), Space: O(1)
        """
        rand_num = random.randint(0, self.valid_size - 1)
        return self.mapping.get(rand_num, rand_num)

# Test
# obj = Solution(4, [2])
# print([obj.pick() for _ in range(10)])  # Should return 0, 1, or 3
```

### Problem 10: Design Twitter

**Problem**: Design a simplified Twitter with core functionalities.

```python
import heapq
from collections import defaultdict

class Twitter:
    def __init__(self):
        self.tweets = defaultdict(list)  # user_id -> [(timestamp, tweet_id)]
        self.follows = defaultdict(set)  # user_id -> set of followed users
        self.timestamp = 0
    
    def post_tweet(self, user_id, tweet_id):
        """Post a tweet"""
        self.tweets[user_id].append((self.timestamp, tweet_id))
        self.timestamp += 1
    
    def get_news_feed(self, user_id):
        """Get 10 most recent tweets from user and people they follow"""
        # Get all relevant users (user + following)
        relevant_users = self.follows[user_id] | {user_id}
        
        # Use heap to get most recent tweets
        tweet_heap = []
        
        for user in relevant_users:
            user_tweets = self.tweets[user]
            # Add up to 10 most recent tweets from this user
            for timestamp, tweet_id in user_tweets[-10:]:
                heapq.heappush(tweet_heap, (-timestamp, tweet_id))
        
        # Return 10 most recent tweets overall
        result = []
        for _ in range(min(10, len(tweet_heap))):
            _, tweet_id = heapq.heappop(tweet_heap)
            result.append(tweet_id)
        
        return result
    
    def follow(self, follower_id, followee_id):
        """Follow a user"""
        if follower_id != followee_id:
            self.follows[follower_id].add(followee_id)
    
    def unfollow(self, follower_id, followee_id):
        """Unfollow a user"""
        self.follows[follower_id].discard(followee_id)

# Test
twitter = Twitter()
twitter.post_tweet(1, 5)
print(twitter.get_news_feed(1))  # [5]
twitter.follow(1, 2)
twitter.post_tweet(2, 6)
print(twitter.get_news_feed(1))  # [6, 5]
```

## 🎯 Key Patterns

### Pattern 1: Sliding Window with Set
```python
# Remove duplicates in window
char_set = set()
left = 0
for right in range(len(s)):
    while s[right] in char_set:
        char_set.remove(s[left])
        left += 1
    char_set.add(s[right])
```

### Pattern 2: Set for Deduplication
```python
# Group by characteristics
groups = defaultdict(list)
for item in items:
    key = transform(item)  # e.g., sorted(item)
    groups[key].append(item)
```

### Pattern 3: Bidirectional Mapping
```python
# Ensure one-to-one mapping
map1 = {}  # key -> value
map2 = {}  # value -> key
# Check both directions for validity
```

## 📈 Time Complexities

- **Set Operations**: O(1) average for add/remove/contains
- **Set Creation**: O(n) where n = number of elements
- **Sliding Window**: O(n) with set for deduplication
- **Group Operations**: O(n * k) where k = key generation cost

## 🚀 Next Level

Master the hardest set problems:
- **[Hard Problems](hard-problems.md)** - Expert-level set algorithms
