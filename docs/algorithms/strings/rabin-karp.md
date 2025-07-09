# Rabin-Karp Algorithm 🔍

Efficient string matching algorithm using hashing technique.

## 🎯 Problem Statement

Find all occurrences of a pattern in a text using rolling hash technique.

**Input**: Text string T and pattern string P
**Output**: All starting positions where P occurs in T

## 🧠 Algorithm Approach

### Key Idea
Use rolling hash to efficiently compute hash values for all substrings of length |P| in text T.

### Rolling Hash Concept
- Compute hash incrementally by removing leftmost character and adding rightmost character
- Avoid recomputing entire hash for each substring

## 📝 Implementation

```python
def rabin_karp(text: str, pattern: str, prime: int = 101) -> list:
    """
    Rabin-Karp string matching algorithm
    
    Args:
        text: Input text string
        pattern: Pattern to search for
        prime: Prime number for hashing
        
    Returns:
        List of starting indices where pattern occurs in text
    """
    n = len(text)
    m = len(pattern)
    
    if m > n:
        return []
    
    # Base for rolling hash
    base = 256
    
    # Hash values
    pattern_hash = 0
    text_hash = 0
    h = 1
    
    # Calculate h = base^(m-1) % prime
    for i in range(m - 1):
        h = (h * base) % prime
    
    # Calculate hash for pattern and first window of text
    for i in range(m):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % prime
        text_hash = (base * text_hash + ord(text[i])) % prime
    
    matches = []
    
    # Slide pattern over text
    for i in range(n - m + 1):
        # Check if hash values match
        if pattern_hash == text_hash:
            # Verify character by character (handle hash collisions)
            if text[i:i + m] == pattern:
                matches.append(i)
        
        # Calculate hash for next window
        if i < n - m:
            text_hash = (base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            
            # Handle negative hash values
            if text_hash < 0:
                text_hash += prime
    
    return matches

# Optimized version with better collision handling
def rabin_karp_optimized(text: str, pattern: str, prime: int = 1000000007) -> list:
    """
    Optimized Rabin-Karp with better collision handling
    
    Args:
        text: Input text string
        pattern: Pattern to search for
        prime: Large prime number for hashing
        
    Returns:
        List of starting indices where pattern occurs in text
    """
    n = len(text)
    m = len(pattern)
    
    if m > n:
        return []
    
    base = 256
    
    # Precompute pattern hash
    pattern_hash = 0
    for char in pattern:
        pattern_hash = (pattern_hash * base + ord(char)) % prime
    
    # Precompute base^(m-1) % prime for rolling hash
    base_power = 1
    for i in range(m - 1):
        base_power = (base_power * base) % prime
    
    # Rolling hash for text
    text_hash = 0
    matches = []
    
    # Process each character in text
    for i in range(n):
        # Add current character to hash
        text_hash = (text_hash * base + ord(text[i])) % prime
        
        # If we have processed at least m characters
        if i >= m - 1:
            # Check if hashes match
            if text_hash == pattern_hash:
                # Verify actual substring
                start_pos = i - m + 1
                if text[start_pos:start_pos + m] == pattern:
                    matches.append(start_pos)
            
            # Remove leftmost character for next iteration
            if i < n - 1:
                text_hash = (text_hash - ord(text[i - m + 1]) * base_power) % prime
                if text_hash < 0:
                    text_hash += prime
    
    return matches

# Multiple pattern search
def rabin_karp_multiple(text: str, patterns: list, prime: int = 1000000007) -> dict:
    """
    Search for multiple patterns simultaneously
    
    Args:
        text: Input text string
        patterns: List of patterns to search for
        prime: Prime number for hashing
        
    Returns:
        Dictionary mapping patterns to their occurrence positions
    """
    n = len(text)
    base = 256
    
    # Group patterns by length for efficiency
    pattern_groups = {}
    for pattern in patterns:
        length = len(pattern)
        if length not in pattern_groups:
            pattern_groups[length] = []
        pattern_groups[length].append(pattern)
    
    all_matches = {pattern: [] for pattern in patterns}
    
    # Process each length group
    for m in pattern_groups:
        if m > n:
            continue
            
        # Compute hashes for all patterns of length m
        pattern_hashes = {}
        for pattern in pattern_groups[m]:
            pattern_hash = 0
            for char in pattern:
                pattern_hash = (pattern_hash * base + ord(char)) % prime
            pattern_hashes[pattern_hash] = pattern
        
        # Rolling hash for text
        base_power = 1
        for i in range(m - 1):
            base_power = (base_power * base) % prime
        
        text_hash = 0
        
        # Process text
        for i in range(n):
            text_hash = (text_hash * base + ord(text[i])) % prime
            
            if i >= m - 1:
                # Check if hash matches any pattern
                if text_hash in pattern_hashes:
                    start_pos = i - m + 1
                    matched_pattern = pattern_hashes[text_hash]
                    if text[start_pos:start_pos + m] == matched_pattern:
                        all_matches[matched_pattern].append(start_pos)
                
                # Remove leftmost character
                if i < n - 1:
                    text_hash = (text_hash - ord(text[i - m + 1]) * base_power) % prime
                    if text_hash < 0:
                        text_hash += prime
    
    return all_matches

# Advanced version with preprocessing
class RabinKarpMatcher:
    def __init__(self, prime: int = 1000000007):
        self.prime = prime
        self.base = 256
        
    def preprocess_pattern(self, pattern: str):
        """Preprocess pattern for multiple searches"""
        self.pattern = pattern
        self.pattern_length = len(pattern)
        
        # Compute pattern hash
        self.pattern_hash = 0
        for char in pattern:
            self.pattern_hash = (self.pattern_hash * self.base + ord(char)) % self.prime
        
        # Precompute base^(m-1) % prime
        self.base_power = 1
        for i in range(self.pattern_length - 1):
            self.base_power = (self.base_power * self.base) % self.prime
    
    def search(self, text: str) -> list:
        """Search for preprocessed pattern in text"""
        n = len(text)
        m = self.pattern_length
        
        if m > n:
            return []
        
        matches = []
        text_hash = 0
        
        # Process each character in text
        for i in range(n):
            text_hash = (text_hash * self.base + ord(text[i])) % self.prime
            
            if i >= m - 1:
                # Check if hashes match
                if text_hash == self.pattern_hash:
                    start_pos = i - m + 1
                    if text[start_pos:start_pos + m] == self.pattern:
                        matches.append(start_pos)
                
                # Remove leftmost character
                if i < n - 1:
                    text_hash = (text_hash - ord(text[i - m + 1]) * self.base_power) % self.prime
                    if text_hash < 0:
                        text_hash += self.prime
        
        return matches

# Example usage and benchmarking
if __name__ == "__main__":
    # Test basic functionality
    text = "AABAACAADAABAAABAA"
    pattern = "AABA"
    
    matches = rabin_karp(text, pattern)
    print(f"Pattern '{pattern}' found at positions: {matches}")
    
    # Test optimized version
    matches_opt = rabin_karp_optimized(text, pattern)
    print(f"Optimized result: {matches_opt}")
    
    # Test multiple patterns
    patterns = ["AABA", "AAB", "CAA"]
    multi_matches = rabin_karp_multiple(text, patterns)
    print(f"Multiple patterns: {multi_matches}")
    
    # Test with RabinKarpMatcher class
    matcher = RabinKarpMatcher()
    matcher.preprocess_pattern("AABA")
    class_matches = matcher.search(text)
    print(f"Class-based result: {class_matches}")
    
    # Performance comparison
    import time
    
    def benchmark_search(text, pattern, iterations=1000):
        """Benchmark different search methods"""
        
        # Rabin-Karp
        start = time.time()
        for _ in range(iterations):
            rabin_karp(text, pattern)
        rk_time = time.time() - start
        
        # Built-in find
        start = time.time()
        for _ in range(iterations):
            pos = 0
            matches = []
            while True:
                pos = text.find(pattern, pos)
                if pos == -1:
                    break
                matches.append(pos)
                pos += 1
        builtin_time = time.time() - start
        
        print(f"Rabin-Karp: {rk_time:.4f}s")
        print(f"Built-in find: {builtin_time:.4f}s")
        print(f"Speedup: {builtin_time/rk_time:.2f}x")
    
    # Large text benchmark
    large_text = "ABCDEFGHIJKLMNOP" * 1000
    benchmark_search(large_text, "FGHIJK")
```

## ⚡ Time Complexity Analysis

### Average Case
- **Time Complexity**: O(n + m)
- **Space Complexity**: O(1)

### Worst Case
- **Time Complexity**: O(n × m) when many hash collisions occur
- **Space Complexity**: O(1)

### Comparison
| Algorithm | Average Case | Worst Case | Space |
|-----------|-------------|------------|-------|
| Rabin-Karp | O(n + m) | O(n × m) | O(1) |
| KMP | O(n + m) | O(n + m) | O(m) |
| Boyer-Moore | O(n/m) | O(n × m) | O(σ) |

## 🔄 Step-by-Step Example

```text
Text: "ABAAABAB"
Pattern: "AABA"
Prime: 101
Base: 256

Step 1: Compute pattern hash
Pattern "AABA" → hash = (((65*256 + 65)*256 + 66)*256 + 65) % 101 = 69

Step 2: Compute first window hash
Text "ABAA" → hash = (((65*256 + 66)*256 + 65)*256 + 65) % 101 = 78

Step 3: Slide window and update hash
Position 0: hash = 78 ≠ 69
Position 1: Remove 'A', add 'A' → hash = 69 = 69 ✓
           Verify: "BAAA" ≠ "AABA" ✗
Position 2: Remove 'B', add 'B' → hash = 96 ≠ 69
Position 3: Remove 'A', add 'A' → hash = 78 ≠ 69
Position 4: Remove 'A', add 'B' → hash = 69 = 69 ✓
           Verify: "ABAB" ≠ "AABA" ✗ (Hash collision!)

Result: Pattern not found (correct)
```

## 🎯 Key Insights

1. **Rolling Hash**: Efficiently update hash by removing and adding characters
2. **Hash Collisions**: Always verify match with actual string comparison
3. **Prime Choice**: Large prime reduces collision probability
4. **Spurious Hits**: Hash match doesn't guarantee string match

## 📊 Hash Function Properties

```python
def analyze_hash_distribution(patterns: list, prime: int = 101):
    """Analyze hash distribution to detect collisions"""
    base = 256
    hash_counts = {}
    
    for pattern in patterns:
        pattern_hash = 0
        for char in pattern:
            pattern_hash = (pattern_hash * base + ord(char)) % prime
        
        hash_counts[pattern_hash] = hash_counts.get(pattern_hash, 0) + 1
    
    collisions = sum(1 for count in hash_counts.values() if count > 1)
    print(f"Total patterns: {len(patterns)}")
    print(f"Unique hashes: {len(hash_counts)}")
    print(f"Collisions: {collisions}")
    
    return hash_counts
```

## 🔧 Optimizations

1. **Prime Selection**: Use large primes to reduce collisions
2. **Multiple Hashing**: Use multiple hash functions for verification
3. **Preprocessing**: Cache pattern hash for multiple searches
4. **Base Optimization**: Choose base that minimizes collisions

## 💡 Applications

- **Text Editors**: Find and replace operations
- **Web Search**: Searching in large documents
- **Bioinformatics**: DNA sequence matching
- **Plagiarism Detection**: Finding copied text segments
- **Network Security**: Pattern matching in network traffic

## 🚀 Advanced Variants

1. **Multi-pattern Rabin-Karp**: Search for multiple patterns simultaneously
2. **2D Pattern Matching**: Extend to 2D grid pattern matching
3. **Approximate Matching**: Allow mismatches within threshold
4. **Parallel Rabin-Karp**: Distribute search across multiple processors

---

*Rabin-Karp algorithm demonstrates how hashing can achieve average-case linear time for string matching.*
