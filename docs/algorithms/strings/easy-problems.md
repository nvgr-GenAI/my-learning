# Strings - Easy Problems

## 🎯 Learning Objectives

Master fundamental string manipulation techniques and basic string algorithms. These 15 problems cover essential string patterns asked in technical interviews.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Valid Palindrome | Two Pointers | Easy | O(n) | O(1) |
    | 2 | Reverse String | Two Pointers | Easy | O(n) | O(1) |
    | 3 | First Unique Character | Hash Map | Easy | O(n) | O(1) |
    | 4 | Valid Anagram | Sorting/Counting | Easy | O(n log n) | O(1) |
    | 5 | Longest Common Prefix | Vertical Scanning | Easy | O(S) | O(1) |
    | 6 | Implement strStr() | String Matching | Easy | O(nm) | O(1) |
    | 7 | Count and Say | String Generation | Easy | O(n*m) | O(m) |
    | 8 | Length of Last Word | String Parsing | Easy | O(n) | O(1) |
    | 9 | Add Binary | String Arithmetic | Easy | O(max(m,n)) | O(max(m,n)) |
    | 10 | Valid Parentheses | Stack | Easy | O(n) | O(n) |
    | 11 | Remove Duplicates from String | Two Pointers | Easy | O(n) | O(1) |
    | 12 | Isomorphic Strings | Hash Map | Easy | O(n) | O(1) |
    | 13 | Rotate String | String Matching | Easy | O(n) | O(1) |
    | 14 | Reverse Words in String III | Two Pointers | Easy | O(n) | O(1) |
    | 15 | Detect Capital | String Pattern | Easy | O(n) | O(1) |

=== "🎯 Core String Patterns"

    **👥 Two Pointers:**
    - Palindrome checking, string reversal
    - In-place character manipulation
    
    **🗂️ Hash Map/Counting:**
    - Character frequency analysis
    - Anagram detection, unique characters
    
    **🔍 String Matching:**
    - Substring finding, pattern matching
    - KMP algorithm basics
    
    **📝 String Processing:**
    - Parsing, validation, transformation
    - Character classification and rules

=== "⚡ Interview Strategy"

    **💡 Problem Recognition:**
    
    - **Character manipulation**: Two pointers for in-place operations
    - **Frequency analysis**: Hash maps for counting characters
    - **Pattern matching**: String search and validation algorithms
    - **String building**: StringBuilder for efficient concatenation
    
    **🔄 Common Approaches:**
    
    1. **Two Pointers**: For palindromes, reversals, comparisons
    2. **Hash Map**: For character counting and frequency problems
    3. **Stack**: For nested structures (parentheses, brackets)
    4. **String Methods**: Leverage built-in functions when appropriate

---

## Problem 1: Valid Palindrome

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers + Character Processing  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    A phrase is a palindrome if, after converting all uppercase letters to lowercase and removing all non-alphanumeric characters, it reads the same forward and backward.

    **Example:**
    ```text
    Input: s = "A man, a plan, a canal: Panama"
    Output: true
    Explanation: "amanaplanacanalpanama" is a palindrome.
    ```

=== "Optimal Solution"

    ```python
    def is_palindrome(s):
        """
        Two pointers approach with character filtering.
        """
        left, right = 0, len(s) - 1
        
        while left < right:
            # Skip non-alphanumeric characters from left
            while left < right and not s[left].isalnum():
                left += 1
            
            # Skip non-alphanumeric characters from right
            while left < right and not s[right].isalnum():
                right -= 1
            
            # Compare characters (case-insensitive)
            if s[left].lower() != s[right].lower():
                return False
            
            left += 1
            right -= 1
        
        return True

    def is_palindrome_preprocessing(s):
        """
        Alternative: preprocess string first.
        """
        # Filter and convert to lowercase
        cleaned = ''.join(c.lower() for c in s if c.isalnum())
        
        # Check if cleaned string equals its reverse
        return cleaned == cleaned[::-1]
    ```

=== "Pattern Recognition"

    ```python
    # Two pointers palindrome template:
    def is_palindrome_template(arr, is_valid, normalize):
        left, right = 0, len(arr) - 1
        
        while left < right:
            # Skip invalid characters
            while left < right and not is_valid(arr[left]):
                left += 1
            while left < right and not is_valid(arr[right]):
                right -= 1
            
            # Compare normalized values
            if normalize(arr[left]) != normalize(arr[right]):
                return False
            
            left += 1
            right -= 1
        
        return True
    ```

---

## Problem 2: Reverse String

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers In-Place  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Reverse a string in-place. The input string is given as an array of characters.

=== "Optimal Solution"

    ```python
    def reverse_string(s):
        """
        In-place reversal using two pointers.
        """
        left, right = 0, len(s) - 1
        
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
    
    def reverse_string_recursive(s):
        """
        Recursive approach for educational purposes.
        """
        def helper(left, right):
            if left >= right:
                return
            
            s[left], s[right] = s[right], s[left]
            helper(left + 1, right - 1)
        
        helper(0, len(s) - 1)
    ```

---

## Problem 3: First Unique Character in String

**Difficulty**: 🟢 Easy  
**Pattern**: Hash Map Frequency Counting  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Find the first non-repeating character in a string and return its index. If it doesn't exist, return -1.

=== "Optimal Solution"

    ```python
    def first_uniq_char(s):
        """
        Two-pass solution with character counting.
        """
        # Count frequency of each character
        char_count = {}
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
        
        # Find first character with count 1
        for i, char in enumerate(s):
            if char_count[char] == 1:
                return i
        
        return -1

    def first_uniq_char_optimized(s):
        """
        Single pass with early termination possible.
        """
        from collections import Counter
        char_count = Counter(s)
        
        for i, char in enumerate(s):
            if char_count[char] == 1:
                return i
        
        return -1
    ```

---

## Problem 4: Valid Anagram

**Difficulty**: 🟢 Easy  
**Pattern**: Sorting or Character Counting  
**Time**: O(n log n) or O(n), **Space**: O(1)

=== "Problem Statement"

    Given two strings s and t, return true if t is an anagram of s.

=== "Optimal Solution"

    ```python
    def is_anagram_sorting(s, t):
        """
        Sorting approach - O(n log n) time.
        """
        return sorted(s) == sorted(t)

    def is_anagram_counting(s, t):
        """
        Character counting - O(n) time.
        """
        if len(s) != len(t):
            return False
        
        char_count = {}
        
        # Count characters in s
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
        
        # Subtract characters in t
        for char in t:
            if char not in char_count:
                return False
            char_count[char] -= 1
            if char_count[char] == 0:
                del char_count[char]
        
        return len(char_count) == 0

    def is_anagram_array(s, t):
        """
        Array counting for lowercase letters only.
        """
        if len(s) != len(t):
            return False
        
        count = [0] * 26
        
        for i in range(len(s)):
            count[ord(s[i]) - ord('a')] += 1
            count[ord(t[i]) - ord('a')] -= 1
        
        return all(c == 0 for c in count)
    ```

---

## Problem 5: Longest Common Prefix

**Difficulty**: 🟢 Easy  
**Pattern**: Vertical Scanning  
**Time**: O(S) where S = sum of all characters, **Space**: O(1)

=== "Problem Statement"

    Find the longest common prefix string amongst an array of strings.

=== "Optimal Solution"

    ```python
    def longest_common_prefix(strs):
        """
        Vertical scanning approach.
        """
        if not strs:
            return ""
        
        # Check each character position
        for i in range(len(strs[0])):
            char = strs[0][i]
            
            # Check if this character matches in all strings
            for j in range(1, len(strs)):
                if i >= len(strs[j]) or strs[j][i] != char:
                    return strs[0][:i]
        
        return strs[0]

    def longest_common_prefix_horizontal(strs):
        """
        Horizontal scanning approach.
        """
        if not strs:
            return ""
        
        prefix = strs[0]
        
        for i in range(1, len(strs)):
            # Reduce prefix until it matches current string
            while not strs[i].startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        
        return prefix
    ```

---

## Problem 6: Implement strStr()

**Difficulty**: 🟢 Easy  
**Pattern**: String Matching  
**Time**: O(nm) brute force, **Space**: O(1)

=== "Problem Statement"

    Return the index of the first occurrence of needle in haystack, or -1 if not found.

=== "Optimal Solution"

    ```python
    def str_str(haystack, needle):
        """
        Brute force string matching.
        """
        if not needle:
            return 0
        
        if len(needle) > len(haystack):
            return -1
        
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i+len(needle)] == needle:
                return i
        
        return -1

    def str_str_optimized(haystack, needle):
        """
        Character-by-character comparison.
        """
        if not needle:
            return 0
        
        for i in range(len(haystack) - len(needle) + 1):
            # Check if needle matches at position i
            j = 0
            while j < len(needle) and haystack[i + j] == needle[j]:
                j += 1
            
            if j == len(needle):
                return i
        
        return -1
    ```

---

## Problem 7: Count and Say

**Difficulty**: 🟢 Easy  
**Pattern**: String Generation with Counting  
**Time**: O(n*m), **Space**: O(m)

=== "Problem Statement"

    Generate the nth term of the "count-and-say" sequence.

=== "Optimal Solution"

    ```python
    def count_and_say(n):
        """
        Iterative generation of count-and-say sequence.
        """
        result = "1"
        
        for _ in range(n - 1):
            result = generate_next(result)
        
        return result

    def generate_next(s):
        """
        Generate next string in count-and-say sequence.
        """
        result = []
        i = 0
        
        while i < len(s):
            count = 1
            char = s[i]
            
            # Count consecutive identical characters
            while i + 1 < len(s) and s[i + 1] == char:
                count += 1
                i += 1
            
            # Add count and character to result
            result.append(str(count))
            result.append(char)
            i += 1
        
        return ''.join(result)
    ```

---

## Problem 8: Length of Last Word

**Difficulty**: 🟢 Easy  
**Pattern**: String Parsing  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Return the length of the last word in a string.

=== "Optimal Solution"

    ```python
    def length_of_last_word(s):
        """
        Traverse from right, skipping trailing spaces.
        """
        i = len(s) - 1
        
        # Skip trailing spaces
        while i >= 0 and s[i] == ' ':
            i -= 1
        
        # Count characters of last word
        length = 0
        while i >= 0 and s[i] != ' ':
            length += 1
            i -= 1
        
        return length

    def length_of_last_word_split(s):
        """
        Using built-in string methods.
        """
        words = s.strip().split()
        return len(words[-1]) if words else 0
    ```

---

## Problem 9: Add Binary

**Difficulty**: 🟢 Easy  
**Pattern**: String Arithmetic  
**Time**: O(max(m,n)), **Space**: O(max(m,n))

=== "Problem Statement"

    Add two binary strings and return their sum (also a binary string).

=== "Optimal Solution"

    ```python
    def add_binary(a, b):
        """
        Add binary strings digit by digit with carry.
        """
        result = []
        carry = 0
        i, j = len(a) - 1, len(b) - 1
        
        while i >= 0 or j >= 0 or carry:
            # Get current digits
            digit_a = int(a[i]) if i >= 0 else 0
            digit_b = int(b[j]) if j >= 0 else 0
            
            # Calculate sum and new carry
            total = digit_a + digit_b + carry
            result.append(str(total % 2))
            carry = total // 2
            
            i -= 1
            j -= 1
        
        return ''.join(reversed(result))

    def add_binary_builtin(a, b):
        """
        Using built-in conversion (less educational).
        """
        return bin(int(a, 2) + int(b, 2))[2:]
    ```

---

## Problem 10: Valid Parentheses

**Difficulty**: 🟢 Easy  
**Pattern**: Stack for Matching  
**Time**: O(n), **Space**: O(n)

=== "Problem Statement"

    Determine if the input string has valid parentheses (properly opened and closed).

=== "Optimal Solution"

    ```python
    def is_valid(s):
        """
        Stack-based parentheses matching.
        """
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in mapping:
                # Closing bracket
                top = stack.pop() if stack else '#'
                if mapping[char] != top:
                    return False
            else:
                # Opening bracket
                stack.append(char)
        
        return not stack
    ```

---

## Problem 11: Remove Duplicates from Sorted String

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Remove duplicates from a sorted string in-place.

=== "Optimal Solution"

    ```python
    def remove_duplicates(s):
        """
        Two pointers for in-place duplicate removal.
        """
        if not s:
            return ""
        
        s = list(s)  # Convert to list for in-place modification
        write_index = 1
        
        for read_index in range(1, len(s)):
            if s[read_index] != s[read_index - 1]:
                s[write_index] = s[read_index]
                write_index += 1
        
        return ''.join(s[:write_index])
    ```

---

## Problem 12: Isomorphic Strings

**Difficulty**: 🟢 Easy  
**Pattern**: Hash Map Bidirectional Mapping  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Determine if two strings are isomorphic (characters can be replaced to get one from the other).

=== "Optimal Solution"

    ```python
    def is_isomorphic(s, t):
        """
        Two hash maps for bidirectional character mapping.
        """
        if len(s) != len(t):
            return False
        
        s_to_t = {}
        t_to_s = {}
        
        for char_s, char_t in zip(s, t):
            # Check s -> t mapping
            if char_s in s_to_t:
                if s_to_t[char_s] != char_t:
                    return False
            else:
                s_to_t[char_s] = char_t
            
            # Check t -> s mapping
            if char_t in t_to_s:
                if t_to_s[char_t] != char_s:
                    return False
            else:
                t_to_s[char_t] = char_s
        
        return True
    ```

---

## Problem 13: Rotate String

**Difficulty**: 🟢 Easy  
**Pattern**: String Matching  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Check if string A can become string B after rotating A.

=== "Optimal Solution"

    ```python
    def rotate_string(A, B):
        """
        Check if B is substring of A+A.
        """
        return len(A) == len(B) and B in A + A
    ```

---

## Problem 14: Reverse Words in String III

**Difficulty**: 🟢 Easy  
**Pattern**: Two Pointers  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Reverse each word in a string while keeping the word order.

=== "Optimal Solution"

    ```python
    def reverse_words(s):
        """
        Reverse each word separately.
        """
        words = s.split()
        return ' '.join(word[::-1] for word in words)

    def reverse_words_in_place(s):
        """
        In-place reversal for each word.
        """
        s = list(s)
        start = 0
        
        for i in range(len(s) + 1):
            if i == len(s) or s[i] == ' ':
                # Reverse word from start to i-1
                reverse_substring(s, start, i - 1)
                start = i + 1
        
        return ''.join(s)

    def reverse_substring(s, left, right):
        """
        Reverse substring in-place.
        """
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
    ```

---

## Problem 15: Detect Capital Use

**Difficulty**: 🟢 Easy  
**Pattern**: String Pattern Validation  
**Time**: O(n), **Space**: O(1)

=== "Problem Statement"

    Check if the capital usage in a word is correct (all capitals, all lowercase, or only first capital).

=== "Optimal Solution"

    ```python
    def detect_capital_use(word):
        """
        Check three valid capital patterns.
        """
        # All uppercase
        if word.isupper():
            return True
        
        # All lowercase
        if word.islower():
            return True
        
        # First uppercase, rest lowercase
        if word[0].isupper() and word[1:].islower():
            return True
        
        return False

    def detect_capital_use_counting(word):
        """
        Count uppercase letters approach.
        """
        uppercase_count = sum(1 for c in word if c.isupper())
        
        # All uppercase or all lowercase
        if uppercase_count == len(word) or uppercase_count == 0:
            return True
        
        # Only first letter uppercase
        if uppercase_count == 1 and word[0].isupper():
            return True
        
        return False
    ```

---

## 🎯 Practice Summary

### Key String Patterns Mastered

1. **Two Pointers**: Palindromes, reversals, in-place operations
2. **Hash Map/Counting**: Character frequency, anagrams, mappings
3. **String Matching**: Substring search, pattern validation
4. **Stack**: Nested structures, parentheses matching
5. **String Building**: Efficient concatenation and generation
6. **Character Processing**: Classification, transformation, validation

### Common String Techniques

- **Character Validation**: `.isalnum()`, `.isalpha()`, `.isdigit()`
- **Case Handling**: `.lower()`, `.upper()`, `.isupper()`, `.islower()`
- **String Methods**: `.split()`, `.join()`, `.strip()`, `.startswith()`
- **Slicing**: `s[::-1]` for reversal, `s[i:j]` for substrings

### Time Complexity Patterns

- **Linear Scan**: O(n) for most string processing
- **Sorting**: O(n log n) for anagram detection
- **Nested Loops**: O(nm) for substring matching
- **Hash Operations**: O(1) average for character lookups

### Space Complexity Considerations

- **In-place**: O(1) extra space when modifying input
- **Hash Maps**: O(k) where k is unique characters (O(1) for ASCII)
- **Stack**: O(n) worst case for nested structures
- **String Building**: O(n) for result construction

### Interview Success Tips

1. **Ask about constraints**: ASCII vs Unicode, case sensitivity
2. **Consider edge cases**: Empty strings, single characters
3. **Choose right approach**: In-place vs new string creation
4. **Optimize when needed**: Use built-in methods wisely
5. **Handle character encoding**: ASCII assumptions vs general Unicode

### Next Steps

Ready for more challenges? Try **[Medium String Problems](medium-problems.md)** to explore:

- Advanced string algorithms (KMP, Rabin-Karp)
- Dynamic programming on strings (Edit Distance, LCS)
- Complex string manipulations (Word Break, Regular Expressions)
- String sliding window techniques

---

*These fundamental string problems build essential text processing skills. Master these patterns before advancing to more complex string algorithms!*
