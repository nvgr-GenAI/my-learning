# Tries (Prefix Trees)

## Overview

A trie (pronounced "try" or "tree") is a tree-like data structure used to store a dynamic set of strings, where the keys are usually strings. Unlike a binary search tree, nodes in the trie do not store their associated key. Instead, the position of a node in the trie defines the key with which it is associated.

Tries are particularly efficient for operations like dictionary lookup, prefix matching, and string sorting. They provide O(m) lookup time, where m is the length of the key being searched, which can be significantly faster than hash tables for certain operations.

## Basic Structure

A trie node typically contains:

1. A boolean flag to indicate if the node represents the end of a word
2. An array or map of child nodes (one for each possible character)

For English words (lowercase), each node might have up to 26 children.

## Operations

### 1. Insertion

To insert a string into a trie:

1. Start at the root
2. For each character in the string:
   - If a child node for that character exists, move to that node
   - If not, create a new node and move to it
3. Mark the final node as the end of a word

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # Maps characters to TrieNode
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True
```

**Time Complexity**: O(m) where m is the length of the word
**Space Complexity**: O(m) for the new nodes in the worst case

### 2. Search

To search for a string in a trie:

1. Start at the root
2. For each character in the string:
   - If a child node for that character exists, move to that node
   - If not, the string is not in the trie
3. If the final node is marked as the end of a word, the string is in the trie

```python
def search(self, word):
    current = self.root
    for char in word:
        if char not in current.children:
            return False
        current = current.children[char]
    return current.is_end_of_word
```

**Time Complexity**: O(m) where m is the length of the word
**Space Complexity**: O(1)

### 3. Prefix Search

To check if any word in the trie starts with a given prefix:

```python
def starts_with(self, prefix):
    current = self.root
    for char in prefix:
        if char not in current.children:
            return False
        current = current.children[char]
    return True
```

**Time Complexity**: O(m) where m is the length of the prefix
**Space Complexity**: O(1)

### 4. Deletion

Removing a word from a trie is more complex as we need to ensure we don't remove nodes that are part of other words:

```python
def delete(self, word):
    def delete_helper(node, word, index):
        if index == len(word):
            # We've reached the end of the word
            if not node.is_end_of_word:
                return False  # Word not found
            node.is_end_of_word = False
            return len(node.children) == 0  # True if this node can be deleted
        
        char = word[index]
        if char not in node.children:
            return False  # Word not found
        
        should_delete_child = delete_helper(node.children[char], word, index + 1)
        
        if should_delete_child:
            del node.children[char]
            return len(node.children) == 0 and not node.is_end_of_word
        
        return False
    
    delete_helper(self.root, word, 0)
```

**Time Complexity**: O(m) where m is the length of the word
**Space Complexity**: O(m) for recursion stack

## Advanced Operations

### 1. Retrieving All Words

To get all words stored in the trie:

```python
def get_all_words(self):
    result = []
    
    def dfs(node, prefix):
        if node.is_end_of_word:
            result.append(prefix)
        
        for char, child in node.children.items():
            dfs(child, prefix + char)
    
    dfs(self.root, "")
    return result
```

**Time Complexity**: O(n) where n is the total number of characters in all words
**Space Complexity**: O(n)

### 2. Autocomplete

To find all words with a given prefix:

```python
def autocomplete(self, prefix):
    current = self.root
    for char in prefix:
        if char not in current.children:
            return []
        current = current.children[char]
    
    result = []
    
    def dfs(node, current_prefix):
        if node.is_end_of_word:
            result.append(current_prefix)
        
        for char, child in node.children.items():
            dfs(child, current_prefix + char)
    
    dfs(current, prefix)
    return result
```

**Time Complexity**: O(n) where n is the total number of characters in all matching words
**Space Complexity**: O(n)

## Space Optimization Techniques

### 1. Compressed Trie (Radix Tree)

A compressed trie merges nodes that have only one child, resulting in strings instead of single characters on edges.

```python
class CompressedTrieNode:
    def __init__(self):
        self.children = {}  # Maps strings to CompressedTrieNode
        self.is_end_of_word = False
```

### 2. Ternary Search Tree

A hybrid between a binary search tree and a trie, where each node has three children: less than, equal to, and greater than.

```python
class TSTNode:
    def __init__(self, char):
        self.char = char
        self.left = None    # For characters < node.char
        self.middle = None  # For traversing down when character matches
        self.right = None   # For characters > node.char
        self.is_end_of_word = False
```

## Applications

1. **Autocomplete Systems**: Suggestions based on user input prefix
2. **Spell Checkers**: Efficiently verify if words are valid
3. **IP Routing Tables**: Store network addresses using CIDR notation
4. **T9 Predictive Text**: Mobile phone keyboards
5. **Word Games**: Efficiently find valid words (e.g., Boggle, Scrabble)
6. **Genomic Data**: Store and query DNA/RNA sequences
7. **Search Engines**: Index terms for fast retrieval

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Insert | O(m) | m is length of key |
| Search | O(m) | m is length of key |
| Delete | O(m) | May require traversing back up |
| Prefix Search | O(m + k) | m is length of prefix, k is number of results |
| Space Complexity | O(ALPHABET_SIZE × n × m) | n is number of keys, m is average length |

## Advantages and Disadvantages

### Advantages

1. **Fast Lookups**: O(m) time regardless of dictionary size
2. **Prefix Operations**: Efficient for prefix-based queries
3. **Lexicographical Ordering**: Words are sorted naturally
4. **Space Efficiency**: (With compression) Can be more space-efficient than hash tables

### Disadvantages

1. **Memory Overhead**: Basic implementation can consume significant memory
2. **Complex Implementation**: Advanced optimizations add complexity
3. **Cache Performance**: Pointer chasing can lead to cache misses
4. **Not Suitable for All Applications**: For simple lookups, hash tables might be preferred

## Implementation in Different Languages

### Java Implementation

```java
class TrieNode {
    private Map<Character, TrieNode> children = new HashMap<>();
    private boolean isEndOfWord;
    
    public Map<Character, TrieNode> getChildren() {
        return children;
    }
    
    public boolean isEndOfWord() {
        return isEndOfWord;
    }
    
    public void setEndOfWord(boolean endOfWord) {
        isEndOfWord = endOfWord;
    }
}

class Trie {
    private TrieNode root;
    
    public Trie() {
        root = new TrieNode();
    }
    
    public void insert(String word) {
        TrieNode current = root;
        
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            TrieNode node = current.getChildren().get(ch);
            
            if (node == null) {
                node = new TrieNode();
                current.getChildren().put(ch, node);
            }
            
            current = node;
        }
        
        current.setEndOfWord(true);
    }
    
    public boolean search(String word) {
        TrieNode current = root;
        
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            TrieNode node = current.getChildren().get(ch);
            
            if (node == null) {
                return false;
            }
            
            current = node;
        }
        
        return current.isEndOfWord();
    }
    
    public boolean startsWith(String prefix) {
        TrieNode current = root;
        
        for (int i = 0; i < prefix.length(); i++) {
            char ch = prefix.charAt(i);
            TrieNode node = current.getChildren().get(ch);
            
            if (node == null) {
                return false;
            }
            
            current = node;
        }
        
        return true;
    }
}
```

## Common Practice Problems

1. **Word Boggle**: Find all possible words in a grid of letters
2. **Longest Common Prefix**: Find the longest common prefix among a set of strings
3. **Word Break Problem**: Determine if a string can be segmented into dictionary words
4. **Auto-complete System**: Design an efficient autocomplete system using tries
5. **Phone Number to Words**: Convert phone numbers to possible word combinations

## Related Data Structures

- [Suffix Trees](suffix-trees.md)
- [Suffix Arrays](suffix-arrays.md)
- [Aho-Corasick Automaton](aho-corasick.md)
- [Directed Acyclic Word Graph (DAWG)](../graphs/dags.md)

## References

1. Fredkin, E. (1960). Trie memory. Communications of the ACM, 3(9), 490-499.
2. Knuth, D. E. (1998). The Art of Computer Programming, Volume 3: Sorting and Searching. Addison-Wesley.
3. Skiena, S. S. (2008). The Algorithm Design Manual. Springer Science & Business Media.
