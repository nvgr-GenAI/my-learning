# String Algorithms 🔤

## 🎯 Overview

String algorithms are essential for text processing, pattern matching, and solving complex string manipulation problems. This comprehensive guide covers fundamental string operations, advanced algorithms, and common interview patterns.

## 📋 Core String Operations

### Basic String Manipulations

```python
def reverse_string(s):
    """Reverse a string in-place (if using list)"""
    if isinstance(s, str):
        return s[::-1]
    
    # In-place for list of characters
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    return s

def is_palindrome(s):
    """Check if string is palindrome"""
    # Clean string: remove non-alphanumeric and convert to lowercase
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

def is_palindrome_two_pointers(s):
    """Check palindrome using two pointers"""
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True

def longest_common_prefix(strs):
    """Find longest common prefix among strings"""
    if not strs:
        return ""
    
    # Find minimum length
    min_len = min(len(s) for s in strs)
    
    for i in range(min_len):
        char = strs[0][i]
        for string in strs[1:]:
            if string[i] != char:
                return strs[0][:i]
    
    return strs[0][:min_len]
```

## 🔍 Pattern Matching Algorithms

### Naive Pattern Matching

```python
def naive_search(text, pattern):
    """Naive pattern matching - O(nm) time"""
    n, m = len(text), len(pattern)
    matches = []
    
    for i in range(n - m + 1):
        j = 0
        while j < m and text[i + j] == pattern[j]:
            j += 1
        
        if j == m:  # Pattern found
            matches.append(i)
    
    return matches
```

### KMP (Knuth-Morris-Pratt) Algorithm

```python
def compute_lps(pattern):
    """Compute Longest Proper Prefix which is also Suffix"""
    m = len(pattern)
    lps = [0] * m
    length = 0  # Length of previous longest prefix suffix
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    return lps

def kmp_search(text, pattern):
    """KMP pattern matching - O(n+m) time"""
    n, m = len(text), len(pattern)
    if m == 0:
        return []
    
    lps = compute_lps(pattern)
    matches = []
    
    i = j = 0  # i for text, j for pattern
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches
```

### Rabin-Karp Algorithm (Rolling Hash)

```python
def rabin_karp_search(text, pattern, prime=101):
    """Rabin-Karp with rolling hash - O(n+m) average time"""
    n, m = len(text), len(pattern)
    if m > n:
        return []
    
    base = 256  # Number of characters in alphabet
    matches = []
    
    # Calculate hash values
    pattern_hash = 0
    text_hash = 0
    h = 1
    
    # h = base^(m-1) % prime
    for i in range(m - 1):
        h = (h * base) % prime
    
    # Calculate initial hash values
    for i in range(m):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % prime
        text_hash = (base * text_hash + ord(text[i])) % prime
    
    # Slide pattern over text
    for i in range(n - m + 1):
        # Check if hash values match
        if pattern_hash == text_hash:
            # Check characters one by one
            if text[i:i+m] == pattern:
                matches.append(i)
        
        # Calculate hash for next window
        if i < n - m:
            text_hash = (base * (text_hash - ord(text[i]) * h) + 
                        ord(text[i + m])) % prime
            
            # Handle negative hash
            if text_hash < 0:
                text_hash += prime
    
    return matches
```

### Z Algorithm

```python
def z_algorithm(s):
    """Compute Z array for string s"""
    n = len(s)
    z = [0] * n
    l = r = 0
    
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    
    return z

def z_search(text, pattern):
    """Pattern matching using Z algorithm"""
    combined = pattern + "$" + text
    z = z_algorithm(combined)
    
    matches = []
    pattern_len = len(pattern)
    
    for i in range(pattern_len + 1, len(combined)):
        if z[i] == pattern_len:
            matches.append(i - pattern_len - 1)
    
    return matches
```

## 🧬 Advanced String Algorithms

### Suffix Array

```python
def build_suffix_array(s):
    """Build suffix array using simple sorting"""
    n = len(s)
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()
    return [suffix[1] for suffix in suffixes]

def build_suffix_array_optimized(s):
    """Build suffix array in O(n log²n) time"""
    n = len(s)
    
    # Initial ranking based on first character
    rank = [ord(c) for c in s]
    temp_rank = [0] * n
    suffix_array = list(range(n))
    
    k = 1
    while k < n:
        # Sort suffixes based on current rankings
        suffix_array.sort(key=lambda x: (rank[x], rank[(x + k) % n]))
        
        # Update rankings
        temp_rank[suffix_array[0]] = 0
        for i in range(1, n):
            curr = suffix_array[i]
            prev = suffix_array[i - 1]
            
            if (rank[curr], rank[(curr + k) % n]) == \
               (rank[prev], rank[(prev + k) % n]):
                temp_rank[curr] = temp_rank[prev]
            else:
                temp_rank[curr] = temp_rank[prev] + 1
        
        rank = temp_rank[:]
        k *= 2
    
    return suffix_array

def lcp_array(s, suffix_array):
    """Build LCP (Longest Common Prefix) array"""
    n = len(s)
    rank = [0] * n
    
    # Build rank array
    for i in range(n):
        rank[suffix_array[i]] = i
    
    lcp = [0] * (n - 1)
    h = 0
    
    for i in range(n):
        if rank[i] > 0:
            j = suffix_array[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i] - 1] = h
            if h > 0:
                h -= 1
    
    return lcp
```

### Manacher's Algorithm (Palindromes)

```python
def manacher_algorithm(s):
    """Find all palindromes using Manacher's algorithm"""
    # Preprocess string
    processed = '#'.join('^{}$'.format(s))
    n = len(processed)
    P = [0] * n  # Array to store palindrome lengths
    center = right = 0
    
    for i in range(1, n - 1):
        # Mirror of current position
        mirror = 2 * center - i
        
        if i < right:
            P[i] = min(right - i, P[mirror])
        
        # Try to expand palindrome centered at i
        try:
            while processed[i + (1 + P[i])] == processed[i - (1 + P[i])]:
                P[i] += 1
        except IndexError:
            pass
        
        # If palindrome centered at i extends past right, adjust center and right
        if i + P[i] > right:
            center, right = i, i + P[i]
    
    return P

def longest_palindromic_substring(s):
    """Find longest palindromic substring"""
    if not s:
        return ""
    
    P = manacher_algorithm(s)
    max_len = max(P)
    center_index = P.index(max_len)
    
    # Convert back to original string coordinates
    start = (center_index - max_len) // 2
    return s[start:start + max_len]

def count_palindromic_substrings(s):
    """Count all palindromic substrings"""
    P = manacher_algorithm(s)
    return sum((length + 1) // 2 for length in P)
```

### Trie (Prefix Tree)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # Count of words ending here

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert word into trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.count += 1
    
    def search(self, word):
        """Search if word exists in trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        """Check if any word starts with prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def find_words_with_prefix(self, prefix):
        """Find all words with given prefix"""
        def dfs(node, current_word, results):
            if node.is_end:
                results.append(current_word)
            
            for char, child in node.children.items():
                dfs(child, current_word + char, results)
        
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        results = []
        dfs(node, prefix, results)
        return results
    
    def delete(self, word):
        """Delete word from trie"""
        def delete_helper(node, word, index):
            if index == len(word):
                if not node.is_end:
                    return False
                node.is_end = False
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False
            
            should_delete = delete_helper(node.children[char], word, index + 1)
            
            if should_delete:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end
            
            return False
        
        delete_helper(self.root, word, 0)
```

## 🎯 String Problem Patterns

### Pattern 1: Sliding Window

```python
def longest_substring_without_repeating(s):
    """Find length of longest substring without repeating characters"""
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

def min_window_substring(s, t):
    """Find minimum window substring containing all characters of t"""
    from collections import Counter, defaultdict
    
    if len(s) < len(t):
        return ""
    
    # Count characters in t
    t_count = Counter(t)
    required = len(t_count)
    formed = 0
    
    # Sliding window
    window_counts = defaultdict(int)
    left = right = 0
    min_len = float('inf')
    min_left = 0
    
    while right < len(s):
        # Expand window
        char = s[right]
        window_counts[char] += 1
        
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1
        
        # Contract window
        while left <= right and formed == required:
            char = s[left]
            
            # Update minimum window
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left
            
            window_counts[char] -= 1
            if char in t_count and window_counts[char] < t_count[char]:
                formed -= 1
            
            left += 1
        
        right += 1
    
    return "" if min_len == float('inf') else s[min_left:min_left + min_len]
```

### Pattern 2: Two Pointers

```python
def is_subsequence(s, t):
    """Check if s is subsequence of t"""
    i = j = 0
    
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    
    return i == len(s)

def merge_strings_alternately(word1, word2):
    """Merge two strings alternately"""
    result = []
    i = j = 0
    
    while i < len(word1) and j < len(word2):
        result.append(word1[i])
        result.append(word2[j])
        i += 1
        j += 1
    
    # Add remaining characters
    result.extend(word1[i:])
    result.extend(word2[j:])
    
    return ''.join(result)
```

### Pattern 3: String Transformation

```python
def edit_distance(word1, word2):
    """Calculate minimum edit distance (Levenshtein distance)"""
    m, n = len(word1), len(word2)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    return dp[m][n]

def longest_common_subsequence(text1, text2):
    """Find length of longest common subsequence"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

### Pattern 4: String Parsing and Validation

```python
def valid_parentheses(s):
    """Check if parentheses are valid"""
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return not stack

def decode_string(s):
    """Decode string like "3[a2[c]]" to "accaccacc" """
    stack = []
    current_num = 0
    current_str = ""
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            stack.append((current_str, current_num))
            current_str = ""
            current_num = 0
        elif char == ']':
            prev_str, num = stack.pop()
            current_str = prev_str + current_str * num
        else:
            current_str += char
    
    return current_str
```

## 🚀 Advanced Problems

### String Matching with Wildcards

```python
def wildcard_matching(s, p):
    """Match string s with pattern p containing * and ?"""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Empty pattern matches empty string
    dp[0][0] = True
    
    # Handle patterns with * at the beginning
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
            elif p[j-1] == '?' or s[i-1] == p[j-1]:
                dp[i][j] = dp[i-1][j-1]
    
    return dp[m][n]

def regular_expression_matching(s, p):
    """Match string s with regex pattern p"""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, etc.
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-2]  # Zero occurrences
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j]  # One or more
            elif p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
    
    return dp[m][n]
```

### Anagram Problems

```python
def group_anagrams(strs):
    """Group strings that are anagrams"""
    from collections import defaultdict
    
    anagram_groups = defaultdict(list)
    
    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        anagram_groups[key].append(s)
    
    return list(anagram_groups.values())

def find_anagrams(s, p):
    """Find all start indices of p's anagrams in s"""
    from collections import Counter
    
    if len(p) > len(s):
        return []
    
    p_count = Counter(p)
    window_count = Counter()
    result = []
    
    for i in range(len(s)):
        # Add character to window
        window_count[s[i]] += 1
        
        # Remove character if window size exceeds p length
        if i >= len(p):
            if window_count[s[i - len(p)]] == 1:
                del window_count[s[i - len(p)]]
            else:
                window_count[s[i - len(p)]] -= 1
        
        # Check if window is anagram of p
        if window_count == p_count:
            result.append(i - len(p) + 1)
    
    return result
```

## 🔍 Time Complexities

| Algorithm | Time Complexity | Space Complexity | Use Case |
|-----------|----------------|------------------|----------|
| Naive Search | O(nm) | O(1) | Simple pattern matching |
| KMP | O(n+m) | O(m) | Efficient pattern matching |
| Rabin-Karp | O(n+m) avg, O(nm) worst | O(1) | Multiple pattern search |
| Z Algorithm | O(n+m) | O(n+m) | Pattern matching, preprocessing |
| Suffix Array | O(n log²n) | O(n) | Multiple queries on string |
| Manacher | O(n) | O(n) | All palindromes |
| Trie Operations | O(m) | O(ALPHABET_SIZE × N × M) | Prefix queries |

## 🎯 Key Takeaways

1. **Choose the right algorithm**: KMP for single pattern, Rabin-Karp for multiple patterns
2. **Use appropriate data structures**: Trie for prefix operations, suffix array for multiple queries
3. **Sliding window**: Efficient for substring problems with constraints
4. **Two pointers**: Great for palindromes and subsequence problems
5. **Dynamic programming**: Essential for edit distance and LCS problems
6. **Hash maps**: Useful for anagram and character frequency problems
7. **String preprocessing**: Can simplify many complex problems

## 📚 Practice Problems

1. **Easy**: Valid Palindrome, Implement strStr(), Longest Common Prefix, Valid Anagram
2. **Medium**: Longest Palindromic Substring, Group Anagrams, Decode String, Find All Anagrams
3. **Hard**: Minimum Window Substring, Edit Distance, Regular Expression Matching, Wildcard Matching

Master these patterns and algorithms to excel at string manipulation problems in interviews and competitive programming!
