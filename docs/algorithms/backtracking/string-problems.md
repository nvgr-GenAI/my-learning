# String Problems

This section covers backtracking algorithms for string-related problems including palindromes, patterns, and IP addresses.

## Core Concepts

### String Backtracking

- Build strings character by character
- Validate constraints at each step
- Backtrack when constraints are violated

### Pattern Matching

- Match strings against patterns with wildcards
- Handle special characters and constraints
- Use recursive validation

---

## Problems

### 1. Generate Parentheses

**Problem**: Generate all valid combinations of n pairs of parentheses.

```python
def generate_parentheses(n):
    """Generate all valid parentheses combinations."""
    result = []
    
    def backtrack(current, open_count, close_count):
        # Base case: used all parentheses
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # Add opening parenthesis if we can
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        
        # Add closing parenthesis if valid
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result

def is_valid_parentheses(s):
    """Check if parentheses string is valid."""
    count = 0
    for char in s:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
            if count < 0:
                return False
    return count == 0
```

### 2. Letter Combinations of Phone Number

**Problem**: Generate all possible letter combinations for a phone number.

```python
def letter_combinations(digits):
    """Generate letter combinations for phone number digits."""
    if not digits:
        return []
    
    digit_to_letters = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    result = []
    
    def backtrack(index, current_combination):
        # Base case: processed all digits
        if index == len(digits):
            result.append(current_combination)
            return
        
        digit = digits[index]
        letters = digit_to_letters.get(digit, '')
        
        for letter in letters:
            backtrack(index + 1, current_combination + letter)
    
    backtrack(0, '')
    return result

# Iterative approach
def letter_combinations_iterative(digits):
    """Iterative solution using queue."""
    if not digits:
        return []
    
    digit_to_letters = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    result = ['']
    for digit in digits:
        letters = digit_to_letters[digit]
        result = [combo + letter for combo in result for letter in letters]
    
    return result
```

### 3. Restore IP Addresses

**Problem**: Generate all valid IP addresses from a string of digits.

```python
def restore_ip_addresses(s):
    """Generate all valid IP addresses from digit string."""
    result = []
    
    def is_valid_part(part):
        # Check if part is valid IP address segment
        if not part or len(part) > 3:
            return False
        if len(part) > 1 and part[0] == '0':
            return False
        return 0 <= int(part) <= 255
    
    def backtrack(start, parts):
        # Base case: found 4 parts
        if len(parts) == 4:
            if start == len(s):
                result.append('.'.join(parts))
            return
        
        # Try different lengths for current part
        for end in range(start + 1, min(start + 4, len(s) + 1)):
            part = s[start:end]
            if is_valid_part(part):
                parts.append(part)
                backtrack(end, parts)
                parts.pop()
    
    backtrack(0, [])
    return result

def count_ip_addresses(s):
    """Count number of valid IP addresses."""
    count = 0
    
    def is_valid_part(part):
        if not part or len(part) > 3:
            return False
        if len(part) > 1 and part[0] == '0':
            return False
        return 0 <= int(part) <= 255
    
    def backtrack(start, parts_count):
        nonlocal count
        if parts_count == 4:
            if start == len(s):
                count += 1
            return
        
        for end in range(start + 1, min(start + 4, len(s) + 1)):
            part = s[start:end]
            if is_valid_part(part):
                backtrack(end, parts_count + 1)
    
    backtrack(0, 0)
    return count
```

### 4. Palindrome Partitioning

**Problem**: Partition string into all possible palindromic substrings.

```python
def partition_palindromes(s):
    """Partition string into palindromic substrings."""
    result = []
    
    def is_palindrome(string):
        return string == string[::-1]
    
    def backtrack(start, current_partition):
        if start == len(s):
            result.append(current_partition[:])
            return
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                current_partition.append(substring)
                backtrack(end, current_partition)
                current_partition.pop()
    
    backtrack(0, [])
    return result

# Optimized with memoization
def partition_palindromes_optimized(s):
    """Optimized palindrome partitioning with precomputed palindromes."""
    n = len(s)
    
    # Precompute palindrome checks
    is_palindrome = [[False] * n for _ in range(n)]
    
    # Every single character is a palindrome
    for i in range(n):
        is_palindrome[i][i] = True
    
    # Check for palindromes of length 2
    for i in range(n - 1):
        is_palindrome[i][i + 1] = (s[i] == s[i + 1])
    
    # Check for longer palindromes
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_palindrome[i][j] = (s[i] == s[j] and is_palindrome[i + 1][j - 1])
    
    result = []
    
    def backtrack(start, current_partition):
        if start == len(s):
            result.append(current_partition[:])
            return
        
        for end in range(start, len(s)):
            if is_palindrome[start][end]:
                current_partition.append(s[start:end + 1])
                backtrack(end + 1, current_partition)
                current_partition.pop()
    
    backtrack(0, [])
    return result
```

### 5. Word Pattern Matching

**Problem**: Check if string follows a given pattern with wildcards.

```python
def word_pattern_match(pattern, s):
    """Check if string matches pattern with bidirectional mapping."""
    
    def backtrack(p_idx, s_idx, char_to_word, word_to_char):
        # Base case: processed entire pattern
        if p_idx == len(pattern):
            return s_idx == len(s)
        
        char = pattern[p_idx]
        
        if char in char_to_word:
            # Character already mapped to a word
            word = char_to_word[char]
            if not s[s_idx:].startswith(word):
                return False
            return backtrack(p_idx + 1, s_idx + len(word), 
                           char_to_word, word_to_char)
        else:
            # Try all possible words for this character
            for end in range(s_idx + 1, len(s) + 1):
                word = s[s_idx:end]
                
                # Check if word is already mapped to another character
                if word in word_to_char:
                    continue
                
                # Create mapping and continue
                char_to_word[char] = word
                word_to_char[word] = char
                
                if backtrack(p_idx + 1, end, char_to_word, word_to_char):
                    return True
                
                # Backtrack
                del char_to_word[char]
                del word_to_char[word]
            
            return False
    
    return backtrack(0, 0, {}, {})

def wildcard_matching(s, p):
    """Match string against pattern with * and ? wildcards."""
    
    def backtrack(s_idx, p_idx):
        # Base cases
        if p_idx == len(p):
            return s_idx == len(s)
        
        if s_idx == len(s):
            # Check if remaining pattern is all '*'
            return all(char == '*' for char in p[p_idx:])
        
        # Current characters
        s_char = s[s_idx]
        p_char = p[p_idx]
        
        if p_char == '*':
            # Try all possibilities for *
            # Match 0 characters
            if backtrack(s_idx, p_idx + 1):
                return True
            # Match 1 or more characters
            if backtrack(s_idx + 1, p_idx):
                return True
        elif p_char == '?' or p_char == s_char:
            # Character match
            return backtrack(s_idx + 1, p_idx + 1)
        
        return False
    
    return backtrack(0, 0)
```

### 6. Remove Invalid Parentheses

**Problem**: Remove minimum number of parentheses to make string valid.

```python
def remove_invalid_parentheses(s):
    """Remove minimum parentheses to make string valid."""
    
    def get_min_removals(s):
        """Calculate minimum removals needed."""
        left_rem = right_rem = 0
        
        for char in s:
            if char == '(':
                left_rem += 1
            elif char == ')':
                if left_rem > 0:
                    left_rem -= 1
                else:
                    right_rem += 1
        
        return left_rem, right_rem
    
    def is_valid(s):
        """Check if string has valid parentheses."""
        count = 0
        for char in s:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
                if count < 0:
                    return False
        return count == 0
    
    left_rem, right_rem = get_min_removals(s)
    result = set()
    
    def backtrack(index, left_count, right_count, 
                  left_rem, right_rem, current):
        if index == len(s):
            if left_rem == 0 and right_rem == 0:
                if is_valid(current):
                    result.add(current)
            return
        
        char = s[index]
        
        # Option 1: Remove current character (if it's a parenthesis)
        if char == '(' and left_rem > 0:
            backtrack(index + 1, left_count, right_count,
                     left_rem - 1, right_rem, current)
        elif char == ')' and right_rem > 0:
            backtrack(index + 1, left_count, right_count,
                     left_rem, right_rem - 1, current)
        
        # Option 2: Keep current character
        backtrack(index + 1, 
                 left_count + (1 if char == '(' else 0),
                 right_count + (1 if char == ')' else 0),
                 left_rem, right_rem, current + char)
    
    backtrack(0, 0, 0, left_rem, right_rem, '')
    return list(result) if result else ['']
```

## Advanced Techniques

### String Optimization

1. **Memoization**: Cache results for repeated subproblems
2. **Trie Structures**: Efficient prefix matching
3. **KMP Algorithm**: Advanced pattern matching

### Constraint Validation

- Validate constraints incrementally
- Use early termination when possible
- Implement efficient validity checks

### Memory Management

- Use string builders for large strings
- Implement in-place modifications when possible
- Use iterative approaches for deep recursion

## Applications

- Text processing and validation
- Pattern matching and regex
- Code generation and parsing
- DNA sequence analysis
- Network protocol validation
