# String Pattern Matching

## Overview

Pattern matching is the process of finding occurrences of a pattern (substring) within a larger text string. This is one of the most fundamental operations in string processing with applications in text editors, search engines, bioinformatics, and more.

## Pattern Matching Algorithms

### 1. Naive Pattern Matching

The simplest approach is to check every possible position in the text where the pattern could match.

#### Algorithm

1. Slide the pattern over the text one by one
2. For each position, check if the pattern matches the current position in the text

```python
def naive_search(text, pattern):
    m, n = len(pattern), len(text)
    positions = []
    
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            positions.append(i)
    
    return positions
```

#### Time Complexity

- Worst-case: O(m × n), where m is pattern length and n is text length
- Best-case: O(n)

#### When to Use

- Short patterns or small texts
- When simplicity is preferred over performance
- When the pattern rarely matches in the text

### 2. Knuth-Morris-Pratt (KMP) Algorithm

KMP improves on the naive approach by avoiding redundant comparisons using a precomputed prefix function.

#### Algorithm

1. Precompute a prefix function (partial match table) for the pattern
2. Use this information to skip comparisons when a mismatch occurs

[Learn more about KMP Algorithm](kmp.md)

#### Time Complexity

- Preprocessing: O(m)
- Searching: O(n)
- Overall: O(m + n)

#### When to Use

- When efficient searching is needed
- For large texts and patterns
- When the pattern may appear multiple times

### 3. Rabin-Karp Algorithm

This algorithm uses hashing to find patterns, making it particularly efficient for multiple pattern searches.

#### Algorithm

1. Compute hash values for the pattern and all possible substrings of the text
2. Compare hash values first, then verify matches by comparing the actual strings

[Learn more about Rabin-Karp Algorithm](rabin-karp.md)

#### Time Complexity

- Average case: O(m + n)
- Worst case: O(m × n)

#### When to Use

- Multiple pattern searches
- When patterns have similar structures
- In applications like plagiarism detection

### 4. Boyer-Moore Algorithm

Often the most efficient string-searching algorithm in practice, Boyer-Moore uses information gathered from preprocessing the pattern to skip sections of the text.

#### Algorithm

1. Preprocess the pattern to build two heuristic tables:
   - Bad Character Heuristic
   - Good Suffix Heuristic
2. Align the pattern with the text and compare characters from right to left
3. Use the heuristics to skip comparisons when mismatches occur

[Learn more about Boyer-Moore Algorithm](boyer-moore.md)

#### Time Complexity

- Best case: O(n/m)
- Worst case: O(m × n)
- Average case: O(n)

#### When to Use

- For long patterns and texts
- When efficiency is critical
- In text editors and search tools

### 5. Z Algorithm

The Z algorithm finds all occurrences of a pattern in a text in linear time by computing the Z array.

#### Algorithm

1. Concatenate pattern + special_character + text
2. Compute the Z array (where Z[i] is the length of the longest substring starting from i that is also a prefix)
3. Positions where Z[i] equals pattern length are matches

[Learn more about Z Algorithm](z-algorithm.md)

#### Time Complexity

- O(m + n)

#### When to Use

- For efficient pattern matching with linear time guarantee
- When preprocessing the pattern and text together is acceptable

### 6. Aho-Corasick Algorithm

This algorithm efficiently searches for multiple patterns simultaneously.

#### Algorithm

1. Build a finite automaton from the set of patterns
2. Process the text using this automaton to find all occurrences of all patterns

[Learn more about Aho-Corasick Algorithm](aho-corasick.md)

#### Time Complexity

- Preprocessing: O(sum of pattern lengths)
- Searching: O(n + number of matches)

#### When to Use

- When searching for multiple patterns simultaneously
- In applications like virus scanning, intrusion detection
- For dictionary matching problems

## Comparison of Pattern Matching Algorithms

| Algorithm | Preprocessing | Searching | Space | Best for |
|-----------|--------------|-----------|-------|----------|
| Naive | O(1) | O(m × n) | O(1) | Short patterns, simplicity |
| KMP | O(m) | O(n) | O(m) | Single pattern, guaranteed linear time |
| Rabin-Karp | O(m) | O(n) average, O(m × n) worst | O(1) | Multiple patterns |
| Boyer-Moore | O(m + alphabet) | O(n/m) best, O(m × n) worst | O(m + alphabet) | Long patterns, practical efficiency |
| Z Algorithm | O(m + n) | part of preprocessing | O(m + n) | Linear time guarantee |
| Aho-Corasick | O(sum of patterns) | O(n + matches) | O(sum of patterns) | Multiple pattern search |

## Applications

1. **Text Editors**: Find/Replace functionality
2. **Web Search Engines**: Locating keywords in documents
3. **Bioinformatics**: DNA sequence matching
4. **Spam Filters**: Detecting patterns in emails
5. **Intrusion Detection Systems**: Identifying attack signatures
6. **Data Mining**: Extracting patterns from large datasets

## Practice Problems

1. Implement the KMP algorithm and use it to find all occurrences of a pattern in a text
2. Create a plagiarism detector using the Rabin-Karp algorithm
3. Build a simple spell checker using pattern matching techniques
4. Implement the Aho-Corasick algorithm for a simple search engine
5. Solve the "Longest Repeated Substring" problem using pattern matching

## Related Topics

- [String Fundamentals](fundamentals.md)
- [Suffix Arrays and Trees](suffix-arrays.md)
- [Tries Data Structure](tries.md)
- [Edit Distance](edit-distance.md)
