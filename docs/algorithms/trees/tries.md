# Tries (Prefix Trees)

## 🎯 Overview

A Trie, also known as a prefix tree, is a tree-like data structure used to store strings efficiently. Each node represents a character, and paths from root to leaves represent complete strings.

## 🔑 Key Concepts

### Structure
- **Root Node**: Empty node at the start
- **Character Nodes**: Each node stores one character
- **End Markers**: Flag to indicate end of valid word
- **Paths**: Root-to-node paths form prefixes

### Properties
- **Prefix Sharing**: Common prefixes share the same path
- **Efficient Search**: O(m) where m is string length
- **Memory Trade-off**: Space for time complexity

---

## 📚 Core Operations

### 1. Trie Implementation

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # Dictionary of character -> TrieNode
        self.is_end_word = False  # Flag for complete word
        self.word_count = 0  # Count of words ending here

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert a word into the trie - O(m) time"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_word = True
        node.word_count += 1
    
    def search(self, word):
        """Search for exact word - O(m) time"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_word
    
    def starts_with(self, prefix):
        """Check if any word starts with prefix - O(m) time"""
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return True
    
    def delete(self, word):
        """Delete a word from trie"""
        def _delete_helper(node, word, index):
            if index == len(word):
                if not node.is_end_word:
                    return False  # Word doesn't exist
                
                node.is_end_word = False
                node.word_count = 0
                
                # Return True if node has no children (can be deleted)
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False
            
            should_delete_child = _delete_helper(
                node.children[char], word, index + 1
            )
            
            if should_delete_child:
                del node.children[char]
                # Return True if current node can be deleted
                return (not node.is_end_word and 
                       len(node.children) == 0)
            
            return False
        
        _delete_helper(self.root, word, 0)

# Example usage
trie = Trie()
words = ["cat", "car", "card", "care", "careful"]
for word in words:
    trie.insert(word)

print(trie.search("car"))        # True
print(trie.search("care"))       # True
print(trie.starts_with("ca"))    # True
print(trie.starts_with("bat"))   # False
```

### 2. Advanced Trie Operations

```python
class AdvancedTrie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_word = True
    
    def get_all_words_with_prefix(self, prefix):
        """Get all words that start with given prefix"""
        def dfs(node, current_word, results):
            if node.is_end_word:
                results.append(current_word)
            
            for char, child_node in node.children.items():
                dfs(child_node, current_word + char, results)
        
        node = self.root
        # Navigate to prefix end
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        results = []
        dfs(node, prefix, results)
        return results
    
    def longest_common_prefix(self):
        """Find longest common prefix of all words"""
        node = self.root
        prefix = ""
        
        while (len(node.children) == 1 and 
               not node.is_end_word and 
               node != self.root):
            char = next(iter(node.children))
            prefix += char
            node = node.children[char]
        
        return prefix
    
    def word_break(self, s):
        """Check if string can be segmented into valid words"""
        def can_break(start):
            if start == len(s):
                return True
            
            node = self.root
            for i in range(start, len(s)):
                char = s[i]
                if char not in node.children:
                    break
                
                node = node.children[char]
                if node.is_end_word and can_break(i + 1):
                    return True
            
            return False
        
        return can_break(0)
    
    def replace_words(self, sentence):
        """Replace words with their shortest root form"""
        def find_root(word):
            node = self.root
            root = ""
            
            for char in word:
                if char not in node.children:
                    return word
                
                node = node.children[char]
                root += char
                
                if node.is_end_word:
                    return root
            
            return word
        
        words = sentence.split()
        return ' '.join(find_root(word) for word in words)

# Example usage
advanced_trie = AdvancedTrie()
words = ["cat", "cats", "caterpillar", "car", "card"]
for word in words:
    advanced_trie.insert(word)

print(advanced_trie.get_all_words_with_prefix("ca"))
# Output: ['cat', 'cats', 'caterpillar', 'car', 'card']
```

---

## 🚀 Advanced Applications

### 1. Autocomplete System

```python
class AutocompleteSystem:
    def __init__(self, sentences, times):
        self.root = TrieNode()
        self.current_input = ""
        
        # Insert all sentences with their frequencies
        for i, sentence in enumerate(sentences):
            self.insert_sentence(sentence, times[i])
    
    def insert_sentence(self, sentence, frequency):
        node = self.root
        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_word = True
        node.word_count = frequency
    
    def input(self, c):
        if c == '#':
            # End of input - add current sentence to trie
            self.insert_sentence(self.current_input, 1)
            self.current_input = ""
            return []
        
        self.current_input += c
        
        # Find all sentences with current prefix
        node = self.root
        for char in self.current_input:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Get top 3 suggestions
        suggestions = []
        self._dfs_suggestions(node, self.current_input, suggestions)
        
        # Sort by frequency (desc) then lexicographically
        suggestions.sort(key=lambda x: (-x[1], x[0]))
        
        return [s[0] for s in suggestions[:3]]
    
    def _dfs_suggestions(self, node, prefix, suggestions):
        if node.is_end_word:
            suggestions.append((prefix, node.word_count))
        
        for char, child in node.children.items():
            self._dfs_suggestions(child, prefix + char, suggestions)

# Example usage
system = AutocompleteSystem(
    ["i love you", "island", "ironman", "i love leetcode"], 
    [5, 3, 2, 2]
)

print(system.input('i'))    # ["i love you", "island", "i love leetcode"]
print(system.input(' '))   # ["i love you", "i love leetcode"]
print(system.input('#'))   # Add "i " to trie, return []
```

### 2. Word Search II (Backtracking + Trie)

```python
def find_words(board, words):
    """Find all words in board using trie optimization"""
    
    # Build trie from words
    trie = Trie()
    for word in words:
        trie.insert(word)
    
    rows, cols = len(board), len(board[0])
    result = set()
    
    def backtrack(r, c, node, path):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        
        char = board[r][c]
        if char not in node.children or char == '#':
            return
        
        node = node.children[char]
        path += char
        
        if node.is_end_word:
            result.add(path)
        
        # Mark as visited
        board[r][c] = '#'
        
        # Explore 4 directions
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            backtrack(r + dr, c + dc, node, path)
        
        # Restore
        board[r][c] = char
    
    # Try starting from each cell
    for i in range(rows):
        for j in range(cols):
            backtrack(i, j, trie.root, "")
    
    return list(result)

# Example usage
board = [
    ['o','a','a','n'],
    ['e','t','a','e'],
    ['i','h','k','r'],
    ['i','f','l','v']
]
words = ["oath", "pea", "eat", "rain"]
print(find_words(board, words))  # ["oath", "eat"]
```

---

## 🎛️ Trie Variations

### 1. Compressed Trie (Radix Tree)

```python
class RadixTrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.edge_label = ""

class RadixTrie:
    """Space-optimized trie that compresses single-child paths"""
    
    def __init__(self):
        self.root = RadixTrieNode()
    
    def insert(self, word):
        """Insert word with path compression"""
        node = self.root
        i = 0
        
        while i < len(word):
            char = word[i]
            
            if char not in node.children:
                # Create new node with remaining suffix
                new_node = RadixTrieNode()
                new_node.edge_label = word[i:]
                new_node.is_end = True
                node.children[char] = new_node
                return
            
            child = node.children[char]
            edge_label = child.edge_label
            
            # Find common prefix
            j = 0
            while (j < len(edge_label) and 
                   i + j < len(word) and 
                   edge_label[j] == word[i + j]):
                j += 1
            
            if j == len(edge_label):
                # Complete edge match, continue
                node = child
                i += j
            else:
                # Partial match - split edge
                self._split_edge(node, char, child, j, word[i:])
                return
        
        node.is_end = True
    
    def _split_edge(self, parent, char, child, split_pos, remaining_word):
        """Split an edge when there's a partial match"""
        # Create intermediate node
        intermediate = RadixTrieNode()
        intermediate.edge_label = child.edge_label[:split_pos]
        
        # Update child's edge label
        child.edge_label = child.edge_label[split_pos:]
        
        # Connect nodes
        parent.children[char] = intermediate
        if child.edge_label:
            intermediate.children[child.edge_label[0]] = child
        
        # Add remaining word
        if len(remaining_word) > split_pos:
            new_child = RadixTrieNode()
            new_child.edge_label = remaining_word[split_pos:]
            new_child.is_end = True
            intermediate.children[remaining_word[split_pos]] = new_child
        else:
            intermediate.is_end = True
```

### 2. Suffix Trie

```python
class SuffixTrie:
    """Trie containing all suffixes of a string"""
    
    def __init__(self, text):
        self.root = TrieNode()
        self.text = text
        self._build_suffix_trie()
    
    def _build_suffix_trie(self):
        """Build trie with all suffixes"""
        for i in range(len(self.text)):
            suffix = self.text[i:] + '$'  # Add terminator
            self._insert_suffix(suffix, i)
    
    def _insert_suffix(self, suffix, start_index):
        node = self.root
        for char in suffix:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_word = True
        node.start_index = start_index
    
    def search_pattern(self, pattern):
        """Find all occurrences of pattern"""
        node = self.root
        
        # Navigate to pattern end
        for char in pattern:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all start indices in subtrie
        indices = []
        self._collect_indices(node, indices)
        return sorted(indices)
    
    def _collect_indices(self, node, indices):
        if hasattr(node, 'start_index'):
            indices.append(node.start_index)
        
        for child in node.children.values():
            self._collect_indices(child, indices)

# Example usage
suffix_trie = SuffixTrie("banana")
print(suffix_trie.search_pattern("ana"))  # [1, 3] - positions of "ana"
```

---

## 📊 Complexity Analysis

| **Operation** | **Time Complexity** | **Space Complexity** | **Notes** |
|---------------|-------------------|---------------------|-----------|
| **Insert** | O(m) | O(m) | m = string length |
| **Search** | O(m) | O(1) | Exact word search |
| **Prefix Search** | O(m) | O(1) | Check if prefix exists |
| **Delete** | O(m) | O(1) | May need cleanup |
| **Autocomplete** | O(m + n) | O(n) | n = results count |

## 🎯 Applications

### Real-World Use Cases
- **Search Engines**: Query autocompletion
- **IDEs**: Code autocompletion
- **Spell Checkers**: Dictionary lookups
- **IP Routing**: Longest prefix matching
- **Bioinformatics**: DNA sequence analysis

### Problem Types
- **String Matching**: Multiple pattern search
- **Dictionary Problems**: Word validation
- **Prefix Operations**: Common prefix, autocomplete
- **Word Games**: Boggle, Scrabble validation

## 🔥 Interview Problems

1. **Implement Trie**: Basic insert, search, startsWith
2. **Word Search II**: Find words in 2D grid
3. **Replace Words**: Replace with shortest root
4. **Stream of Characters**: Check if suffix forms valid word
5. **Maximum XOR**: Trie for binary representations

---

*Tries are powerful for string processing - master them for efficient text algorithms!*
