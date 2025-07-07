# Strings - Medium Problems

## 🎯 Learning Objectives

Master intermediate string algorithms and advanced string manipulation patterns. These 15 problems introduce complex string processing techniques and algorithmic paradigms.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Longest Substring Without Repeating | Sliding Window | Medium | O(n) | O(min(m,n)) |
    | 2 | Longest Palindromic Substring | Expand Around Centers | Medium | O(n²) | O(1) |
    | 3 | String to Integer (atoi) | String Parsing | Medium | O(n) | O(1) |
    | 4 | ZigZag Conversion | String Pattern | Medium | O(n) | O(n) |
    | 5 | Reverse Words in String | Two Pointers + Parsing | Medium | O(n) | O(1) |
    | 6 | Word Break | Dynamic Programming | Medium | O(n²) | O(n) |
    | 7 | Group Anagrams | Hash Map + Sorting | Medium | O(nm log m) | O(nm) |
    | 8 | Longest Common Subsequence | 2D DP | Medium | O(mn) | O(mn) |
    | 9 | Edit Distance | 2D DP | Medium | O(mn) | O(mn) |
    | 10 | Find All Anagrams in String | Sliding Window | Medium | O(n) | O(1) |
    | 11 | Minimum Window Substring | Sliding Window | Medium | O(n) | O(n) |
    | 12 | Decode Ways | Linear DP | Medium | O(n) | O(1) |
    | 13 | Valid Palindrome II | Two Pointers | Medium | O(n) | O(1) |
    | 14 | Longest Repeating Character Replacement | Sliding Window | Medium | O(n) | O(1) |
    | 15 | Palindromic Substrings | Expand Around Centers | Medium | O(n²) | O(1) |

=== "🎯 Advanced String Patterns"

    **🪟 Sliding Window:**
    - Variable-size windows for substring problems
    - Character frequency tracking
    
    **🧮 String DP:**
    - 2D DP for string comparison problems
    - State transitions based on character matches
    
    **🔄 String Transformation:**
    - Pattern matching and string generation
    - Complex parsing with state management
    
    **🎯 Advanced Two Pointers:**
    - Multi-criteria string validation
    - Palindrome detection with modifications

=== "⚡ Advanced Interview Strategy"

    **💡 Pattern Recognition:**
    
    - **Substring problems**: Often use sliding window technique
    - **String comparison**: Usually requires 2D DP approach
    - **Pattern matching**: Consider KMP or string hashing
    - **Palindrome problems**: Expand around centers or DP
    
    **🔄 Advanced Techniques:**
    
    1. **Sliding Window**: Maintain window invariants with hash maps
    2. **String DP**: Build solutions bottom-up comparing characters
    3. **String Hashing**: Use rolling hash for efficient pattern matching
    4. **Trie Structures**: For prefix-based string problems

---

## Problem 1: Longest Substring Without Repeating Characters

**Difficulty**: 🟡 Medium  
**Pattern**: Sliding Window + Hash Map  
**Time**: O(n), **Space**: O(min(m,n))

=== "Problem Statement"

    Given a string s, find the length of the longest substring without repeating characters.

    **Example:**
    ```text
    Input: s = "abcabcbb"
    Output: 3
    Explanation: The answer is "abc", with the length of 3.
    ```

=== "Optimal Solution"

    ```python
    def length_of_longest_substring(s):
        """
        Sliding window with hash map to track character positions.
        """
        char_map = {}
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            char = s[right]
            
            # If character seen before and within current window
            if char in char_map and char_map[char] >= left:
                left = char_map[char] + 1
            
            char_map[char] = right
            max_length = max(max_length, right - left + 1)
        
        return max_length

    def length_of_longest_substring_set(s):
        """
        Alternative using set for character tracking.
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
    ```

=== "Pattern Template"

    ```python
    # Sliding window template for substring problems:
    def sliding_window_template(s, target):
        left = 0
        window_map = {}
        result = 0
        
        for right in range(len(s)):
            # Expand window
            char = s[right]
            window_map[char] = window_map.get(char, 0) + 1
            
            # Contract window if needed
            while window_invalid(window_map, target):
                left_char = s[left]
                window_map[left_char] -= 1
                if window_map[left_char] == 0:
                    del window_map[left_char]
                left += 1
            
            # Update result
            result = max(result, right - left + 1)
        
        return result
    ```

---

## Problem 2: Longest Palindromic Substring

**Difficulty**: 🟡 Medium  
**Pattern**: Expand Around Centers  
**Time**: O(n²), **Space**: O(1)

=== "Problem Statement"

    Given a string s, return the longest palindromic substring in s.

=== "Optimal Solution"

    ```python
    def longest_palindrome(s):
        """
        Expand around centers approach - check all possible centers.
        """
        if not s:
            return ""
        
        start = 0
        max_len = 1
        
        for i in range(len(s)):
            # Check for odd length palindromes (center at i)
            len1 = expand_around_center(s, i, i)
            
            # Check for even length palindromes (center between i and i+1)
            len2 = expand_around_center(s, i, i + 1)
            
            # Update maximum palindrome found
            current_max = max(len1, len2)
            if current_max > max_len:
                max_len = current_max
                start = i - (current_max - 1) // 2
        
        return s[start:start + max_len]

    def expand_around_center(s, left, right):
        """
        Expand around center and return palindrome length.
        """
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        
        return right - left - 1

    def longest_palindrome_dp(s):
        """
        Dynamic programming approach - O(n²) time, O(n²) space.
        """
        n = len(s)
        if n == 0:
            return ""
        
        dp = [[False] * n for _ in range(n)]
        start = 0
        max_len = 1
        
        # Single characters are palindromes
        for i in range(n):
            dp[i][i] = True
        
        # Check for palindromes of length 2
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                start = i
                max_len = 2
        
        # Check for palindromes of length 3 and more
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    start = i
                    max_len = length
        
        return s[start:start + max_len]
    ```

---

## Problem 3: String to Integer (atoi)

**Difficulty**: 🟡 Medium  
**Pattern**: String Parsing with State Machine  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer.

=== "Optimal Solution"

    ```python
    def my_atoi(s):
        """
        Implement atoi with proper edge case handling.
        """
        if not s:
            return 0
        
        # Constants
        INT_MAX = 2**31 - 1
        INT_MIN = -2**31
        
        # Skip leading whitespace
        i = 0
        while i < len(s) and s[i] == ' ':
            i += 1
        
        if i == len(s):
            return 0
        
        # Handle sign
        sign = 1
        if s[i] == '+':
            i += 1
        elif s[i] == '-':
            sign = -1
            i += 1
        
        # Convert digits
        result = 0
        while i < len(s) and s[i].isdigit():
            digit = int(s[i])
            
            # Check for overflow before adding digit
            if result > (INT_MAX - digit) // 10:
                return INT_MAX if sign == 1 else INT_MIN
            
            result = result * 10 + digit
            i += 1
        
        return sign * result
    ```

---

## Problem 4: Word Break

**Difficulty**: 🟡 Medium  
**Pattern**: Dynamic Programming + String Matching  
**Time**: O(n²), **Space**: O(n)

=== "Problem Statement"

    Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of dictionary words.

=== "Optimal Solution"

    ```python
    def word_break(s, word_dict):
        """
        DP approach: dp[i] = True if s[0:i] can be segmented.
        """
        word_set = set(word_dict)
        dp = [False] * (len(s) + 1)
        dp[0] = True  # Empty string can always be segmented
        
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[len(s)]

    def word_break_optimized(s, word_dict):
        """
        Optimized with early termination and max word length.
        """
        word_set = set(word_dict)
        max_len = max(len(word) for word in word_dict) if word_dict else 0
        
        dp = [False] * (len(s) + 1)
        dp[0] = True
        
        for i in range(1, len(s) + 1):
            # Only check words that can fit
            start = max(0, i - max_len)
            for j in range(start, i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[len(s)]
    ```

---

## Problem 5: Group Anagrams

**Difficulty**: 🟡 Medium  
**Pattern**: Hash Map + Sorting  
**Time**: O(nm log m), **Space**: O(nm)

=== "Problem Statement"

    Given an array of strings strs, group the anagrams together.

=== "Optimal Solution"

    ```python
    def group_anagrams(strs):
        """
        Group anagrams using sorted string as key.
        """
        from collections import defaultdict
        
        anagram_map = defaultdict(list)
        
        for s in strs:
            # Use sorted string as key
            key = ''.join(sorted(s))
            anagram_map[key].append(s)
        
        return list(anagram_map.values())

    def group_anagrams_counting(strs):
        """
        Alternative using character counting as key.
        """
        from collections import defaultdict
        
        anagram_map = defaultdict(list)
        
        for s in strs:
            # Count characters
            count = [0] * 26
            for char in s:
                count[ord(char) - ord('a')] += 1
            
            # Use count tuple as key
            key = tuple(count)
            anagram_map[key].append(s)
        
        return list(anagram_map.values())
    ```

---

## Problem 6: Longest Common Subsequence

**Difficulty**: 🟡 Medium  
**Pattern**: 2D Dynamic Programming  
**Time**: O(mn), **Space**: O(mn)

=== "Problem Statement"

    Given two strings text1 and text2, return the length of their longest common subsequence.

=== "Optimal Solution"

    ```python
    def longest_common_subsequence(text1, text2):
        """
        2D DP: dp[i][j] = LCS length of text1[0:i] and text2[0:j].
        """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]

    def longest_common_subsequence_optimized(text1, text2):
        """
        Space-optimized version using only two rows.
        """
        m, n = len(text1), len(text2)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            
            prev, curr = curr, prev
        
        return prev[n]
    ```

---

## Problem 7: Edit Distance

**Difficulty**: 🟡 Medium  
**Pattern**: 2D Dynamic Programming  
**Time**: O(mn), **Space**: O(mn)

=== "Problem Statement"

    Given two strings word1 and word2, return the minimum number of operations to convert word1 to word2.

=== "Optimal Solution"

    ```python
    def min_distance(word1, word2):
        """
        Edit distance using 2D DP.
        Operations: insert, delete, replace.
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i  # Delete all characters
        for j in range(n + 1):
            dp[0][j] = j  # Insert all characters
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No operation needed
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Delete
                        dp[i][j-1],    # Insert
                        dp[i-1][j-1]   # Replace
                    )
        
        return dp[m][n]
    ```

---

## Problem 8: Minimum Window Substring

**Difficulty**: 🟡 Medium  
**Pattern**: Sliding Window + Hash Map  
**Time**: O(n), **Space**: O(n)

=== "Problem Statement"

    Given two strings s and t, return the minimum window substring of s such that every character in t is included in the window.

=== "Optimal Solution"

    ```python
    def min_window(s, t):
        """
        Sliding window with character frequency tracking.
        """
        if not s or not t:
            return ""
        
        from collections import Counter
        
        # Count characters in t
        t_count = Counter(t)
        required = len(t_count)
        
        # Sliding window variables
        left = right = 0
        formed = 0  # Number of unique chars in window with desired frequency
        
        window_counts = {}
        
        # Result: (window length, left, right)
        ans = float('inf'), None, None
        
        while right < len(s):
            # Add character from right to window
            char = s[right]
            window_counts[char] = window_counts.get(char, 0) + 1
            
            # Check if current character's frequency matches desired count
            if char in t_count and window_counts[char] == t_count[char]:
                formed += 1
            
            # Try to contract window from left
            while left <= right and formed == required:
                char = s[left]
                
                # Update result if this window is smaller
                if right - left + 1 < ans[0]:
                    ans = (right - left + 1, left, right)
                
                # Remove character from left
                window_counts[char] -= 1
                if char in t_count and window_counts[char] < t_count[char]:
                    formed -= 1
                
                left += 1
            
            right += 1
        
        return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]
    ```

---

## Problem 9: Find All Anagrams in String

**Difficulty**: 🟡 Medium  
**Pattern**: Sliding Window + Character Counting  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Given two strings s and p, return an array of all the start indices of p's anagrams in s.

=== "Optimal Solution"

    ```python
    def find_anagrams(s, p):
        """
        Sliding window with fixed size equal to len(p).
        """
        if len(p) > len(s):
            return []
        
        from collections import Counter
        
        p_count = Counter(p)
        window_count = Counter()
        result = []
        
        # Initialize window
        for i in range(len(p)):
            window_count[s[i]] += 1
        
        # Check first window
        if window_count == p_count:
            result.append(0)
        
        # Slide window
        for i in range(len(p), len(s)):
            # Add new character
            window_count[s[i]] += 1
            
            # Remove old character
            left_char = s[i - len(p)]
            window_count[left_char] -= 1
            if window_count[left_char] == 0:
                del window_count[left_char]
            
            # Check if current window is anagram
            if window_count == p_count:
                result.append(i - len(p) + 1)
        
        return result
    ```

---

## Problem 10: Decode Ways

**Difficulty**: 🟡 Medium  
**Pattern**: Linear Dynamic Programming  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    A message containing letters from A-Z can be encoded into numbers using the mapping A=1, B=2, ..., Z=26. Given a string s containing only digits, return the number of ways to decode it.

=== "Optimal Solution"

    ```python
    def num_decodings(s):
        """
        DP approach: dp[i] = number of ways to decode s[0:i].
        """
        if not s or s[0] == '0':
            return 0
        
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty string has one way
        dp[1] = 1  # First character (non-zero)
        
        for i in range(2, n + 1):
            # Single digit decoding
            if s[i-1] != '0':
                dp[i] += dp[i-1]
            
            # Two digit decoding
            two_digit = int(s[i-2:i])
            if 10 <= two_digit <= 26:
                dp[i] += dp[i-2]
        
        return dp[n]

    def num_decodings_optimized(s):
        """
        Space-optimized version using variables.
        """
        if not s or s[0] == '0':
            return 0
        
        prev2 = 1  # dp[i-2]
        prev1 = 1  # dp[i-1]
        
        for i in range(1, len(s)):
            current = 0
            
            # Single digit
            if s[i] != '0':
                current += prev1
            
            # Two digits
            two_digit = int(s[i-1:i+1])
            if 10 <= two_digit <= 26:
                current += prev2
            
            prev2, prev1 = prev1, current
        
        return prev1
    ```

---

## Problem 11: Valid Palindrome II

**Difficulty**: 🟡 Medium  
**Pattern**: Two Pointers with One Deletion  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Given a string s, return true if the s can be palindrome after deleting at most one character from it.

=== "Optimal Solution"

    ```python
    def valid_palindrome(s):
        """
        Two pointers with at most one character deletion.
        """
        def is_palindrome_range(left, right):
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        left, right = 0, len(s) - 1
        
        while left < right:
            if s[left] != s[right]:
                # Try deleting left character or right character
                return (is_palindrome_range(left + 1, right) or 
                        is_palindrome_range(left, right - 1))
            left += 1
            right -= 1
        
        return True
    ```

---

## Problem 12: Longest Repeating Character Replacement

**Difficulty**: 🟡 Medium  
**Pattern**: Sliding Window + Character Counting  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    You can change at most k characters to make the longest possible substring containing the same character.

=== "Optimal Solution"

    ```python
    def character_replacement(s, k):
        """
        Sliding window tracking most frequent character.
        """
        from collections import defaultdict
        
        left = 0
        max_freq = 0
        max_length = 0
        char_count = defaultdict(int)
        
        for right in range(len(s)):
            char_count[s[right]] += 1
            max_freq = max(max_freq, char_count[s[right]])
            
            # If window size - max_freq > k, shrink window
            while right - left + 1 - max_freq > k:
                char_count[s[left]] -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    ```

---

## Problem 13: Palindromic Substrings

**Difficulty**: 🟡 Medium  
**Pattern**: Expand Around Centers  
**Time**: O(n²), **Space**: O(1)

=== "Problem Statement"

    Given a string s, return the number of palindromic substrings in it.

=== "Optimal Solution"

    ```python
    def count_substrings(s):
        """
        Count palindromic substrings by expanding around centers.
        """
        def expand_around_center(left, right):
            count = 0
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
            return count
        
        total_count = 0
        
        for i in range(len(s)):
            # Odd length palindromes (center at i)
            total_count += expand_around_center(i, i)
            
            # Even length palindromes (center between i and i+1)
            total_count += expand_around_center(i, i + 1)
        
        return total_count
    ```

---

## Problem 14: Reverse Words in a String

**Difficulty**: 🟡 Medium  
**Pattern**: Two Pointers + String Parsing  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Given an input string s, reverse the order of the words.

=== "Optimal Solution"

    ```python
    def reverse_words(s):
        """
        In-place word reversal with proper space handling.
        """
        # Convert to list for in-place modification
        chars = list(s)
        n = len(chars)
        
        # Step 1: Reverse entire string
        reverse_string(chars, 0, n - 1)
        
        # Step 2: Reverse each word and clean spaces
        start = 0
        for end in range(n + 1):
            if end == n or chars[end] == ' ':
                # Reverse current word
                reverse_string(chars, start, end - 1)
                start = end + 1
        
        # Step 3: Clean up extra spaces
        return clean_spaces(chars)

    def reverse_string(chars, left, right):
        """Reverse characters in place."""
        while left < right:
            chars[left], chars[right] = chars[right], chars[left]
            left += 1
            right -= 1

    def clean_spaces(chars):
        """Remove extra spaces."""
        n = len(chars)
        i = j = 0
        
        while j < n:
            # Skip spaces
            while j < n and chars[j] == ' ':
                j += 1
            
            # Copy word
            while j < n and chars[j] != ' ':
                chars[i] = chars[j]
                i += 1
                j += 1
            
            # Add single space after word (except last word)
            while j < n and chars[j] == ' ':
                j += 1
            if j < n:
                chars[i] = ' '
                i += 1
        
        return ''.join(chars[:i])
    ```

---

## Problem 15: ZigZag Conversion

**Difficulty**: 🟡 Medium  
**Pattern**: String Pattern Construction  
**Time**: O(n), **Space**: O(n)

=== "Problem Statement"

    The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows.

=== "Optimal Solution"

    ```python
    def convert(s, num_rows):
        """
        Build zigzag pattern by tracking direction and current row.
        """
        if num_rows == 1 or num_rows >= len(s):
            return s
        
        rows = [''] * num_rows
        current_row = 0
        going_down = False
        
        for char in s:
            rows[current_row] += char
            
            # Change direction at top and bottom rows
            if current_row == 0 or current_row == num_rows - 1:
                going_down = not going_down
            
            # Move to next row
            current_row += 1 if going_down else -1
        
        return ''.join(rows)
    ```

---

## 🎯 Practice Summary

### Advanced String Patterns Mastered

1. **Sliding Window Techniques**: Variable-size windows with hash maps
2. **String Dynamic Programming**: 2D DP for string comparison problems
3. **Advanced Two Pointers**: Multi-criteria validation and palindromes
4. **String Pattern Recognition**: ZigZag, parsing, and transformation
5. **Character Frequency Analysis**: Anagram detection and substring problems
6. **String Matching Algorithms**: Efficient substring search techniques

### Key Algorithmic Insights

- **Sliding Window**: Maintain window invariants with expand/contract logic
- **String DP**: Build solutions by comparing characters systematically
- **Expand Around Centers**: Efficient palindrome detection technique
- **Character Counting**: Use hash maps for frequency-based problems
- **Space Optimization**: Reduce 2D DP to 1D when possible

### Time Complexity Patterns

- **Sliding Window**: O(n) for most substring problems
- **String DP**: O(mn) for comparing two strings
- **Expand Centers**: O(n²) for palindrome problems
- **Sorting-based**: O(n log n) for grouping anagrams
- **Pattern Construction**: O(n) for string transformations

### Space Complexity Considerations

- **Hash Maps**: O(k) where k is unique characters (often O(1) for ASCII)
- **DP Tables**: O(mn) for 2D problems, can optimize to O(n)
- **In-place Operations**: O(1) extra space when modifying input
- **Result Construction**: O(n) for building output strings

### Interview Success Strategy

1. **Identify Pattern**: Substring → sliding window, comparison → DP
2. **Choose Data Structure**: Hash map for frequency, array for DP
3. **Optimize Space**: Consider space-optimized DP variants
4. **Handle Edge Cases**: Empty strings, single characters, special patterns
5. **Verify Complexity**: Ensure optimal time/space for problem constraints

### Next Steps

Ready for the ultimate challenges? Try **[Hard String Problems](hard-problems.md)** to explore:

- Advanced pattern matching (KMP, Rabin-Karp algorithms)
- Complex string DP (Regular Expression Matching, Wildcard Matching)
- Advanced parsing (Calculator implementations, Expression evaluation)
- String compression and encoding algorithms

---

*These medium string problems introduce advanced techniques essential for mastering complex string algorithms. Focus on understanding the underlying patterns and optimization techniques!*
