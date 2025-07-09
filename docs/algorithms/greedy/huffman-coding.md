# Huffman Coding

## Overview

Huffman Coding is a lossless data compression algorithm that uses variable-length encoding to represent common characters with shorter codes and less common characters with longer codes. It was developed by David A. Huffman in 1952 and is widely used in file compression utilities.

## Algorithm

1. Calculate the frequency of each character in the input text
2. Build a priority queue (min-heap) of nodes, each containing a character and its frequency
3. While there is more than one node in the priority queue:
   - Extract the two nodes with the lowest frequencies
   - Create a new internal node with these two nodes as children, with frequency equal to the sum of their frequencies
   - Add the new node back to the priority queue
4. The remaining node is the root of the Huffman tree
5. Traverse the tree to assign codes (0 for left edges, 1 for right edges)

## Implementation

### Python Implementation

```python
import heapq
from collections import Counter

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    """
    Build a Huffman tree from the input text.
    
    Args:
        text: Input string
        
    Returns:
        Root of the Huffman tree
    """
    # Count the frequency of each character
    frequency = Counter(text)
    
    # Create a priority queue with nodes for each character
    priority_queue = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(priority_queue)
    
    # Build the Huffman tree
    while len(priority_queue) > 1:
        # Extract the two nodes with lowest frequency
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        
        # Create a new internal node with these two nodes as children
        internal_node = Node(None, left.freq + right.freq)
        internal_node.left = left
        internal_node.right = right
        
        # Add the internal node back to the priority queue
        heapq.heappush(priority_queue, internal_node)
    
    # The remaining node is the root of the Huffman tree
    return priority_queue[0]

def build_huffman_codes(node, code="", codes=None):
    """
    Build Huffman codes from the Huffman tree.
    
    Args:
        node: Current node in the Huffman tree
        code: Current code (path to the node)
        codes: Dictionary to store character-to-code mappings
        
    Returns:
        Dictionary mapping characters to their Huffman codes
    """
    if codes is None:
        codes = {}
    
    # If this is a leaf node (has a character), assign the code
    if node.char is not None:
        codes[node.char] = code
    else:
        # Traverse left (add '0' to the code)
        if node.left:
            build_huffman_codes(node.left, code + "0", codes)
        
        # Traverse right (add '1' to the code)
        if node.right:
            build_huffman_codes(node.right, code + "1", codes)
    
    return codes

def huffman_encode(text):
    """
    Encode the input text using Huffman coding.
    
    Args:
        text: Input string
        
    Returns:
        Tuple of (encoded text, Huffman codes)
    """
    # Handle empty text
    if not text:
        return "", {}
    
    # Handle text with only one unique character
    if len(set(text)) == 1:
        return "0" * len(text), {text[0]: "0"}
    
    # Build the Huffman tree
    root = build_huffman_tree(text)
    
    # Build the Huffman codes
    codes = build_huffman_codes(root)
    
    # Encode the text
    encoded_text = "".join(codes[char] for char in text)
    
    return encoded_text, codes

def huffman_decode(encoded_text, codes):
    """
    Decode the Huffman encoded text.
    
    Args:
        encoded_text: Binary string of the encoded text
        codes: Dictionary mapping characters to their Huffman codes
        
    Returns:
        Decoded text
    """
    # Invert the codes for decoding
    reverse_codes = {code: char for char, code in codes.items()}
    
    decoded_text = ""
    current_code = ""
    
    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_codes:
            decoded_text += reverse_codes[current_code]
            current_code = ""
    
    return decoded_text

# Example usage
text = "this is an example for huffman encoding"
encoded, codes = huffman_encode(text)

print(f"Original text: {text}")
print(f"Encoded text: {encoded}")
print(f"Huffman codes: {codes}")

# Calculate compression ratio
original_size = len(text) * 8  # assuming 8 bits per character
compressed_size = len(encoded)
compression_ratio = original_size / compressed_size

print(f"Original size (bits): {original_size}")
print(f"Compressed size (bits): {compressed_size}")
print(f"Compression ratio: {compression_ratio:.2f}x")

# Verify decoding
decoded = huffman_decode(encoded, codes)
print(f"Decoded text: {decoded}")
assert text == decoded, "Decoding failed!"
```

### Java Implementation

```java
import java.util.*;

public class HuffmanCoding {
    
    static class Node implements Comparable<Node> {
        Character character;
        int frequency;
        Node left;
        Node right;
        
        public Node(Character character, int frequency) {
            this.character = character;
            this.frequency = frequency;
            this.left = null;
            this.right = null;
        }
        
        @Override
        public int compareTo(Node other) {
            return this.frequency - other.frequency;
        }
    }
    
    public static Node buildHuffmanTree(String text) {
        // Count the frequency of each character
        Map<Character, Integer> frequencyMap = new HashMap<>();
        for (char c : text.toCharArray()) {
            frequencyMap.put(c, frequencyMap.getOrDefault(c, 0) + 1);
        }
        
        // Create a priority queue (min-heap)
        PriorityQueue<Node> priorityQueue = new PriorityQueue<>();
        
        // Add nodes for each character
        for (Map.Entry<Character, Integer> entry : frequencyMap.entrySet()) {
            priorityQueue.add(new Node(entry.getKey(), entry.getValue()));
        }
        
        // Build the Huffman tree
        while (priorityQueue.size() > 1) {
            // Extract the two nodes with lowest frequency
            Node left = priorityQueue.poll();
            Node right = priorityQueue.poll();
            
            // Create a new internal node with these two nodes as children
            Node internalNode = new Node(null, left.frequency + right.frequency);
            internalNode.left = left;
            internalNode.right = right;
            
            // Add the internal node back to the priority queue
            priorityQueue.add(internalNode);
        }
        
        // The remaining node is the root of the Huffman tree
        return priorityQueue.poll();
    }
    
    public static Map<Character, String> buildHuffmanCodes(Node root) {
        Map<Character, String> codes = new HashMap<>();
        buildHuffmanCodesRecursive(root, "", codes);
        return codes;
    }
    
    private static void buildHuffmanCodesRecursive(Node node, String code, Map<Character, String> codes) {
        // If this is a leaf node (has a character), assign the code
        if (node.character != null) {
            codes.put(node.character, code);
            return;
        }
        
        // Traverse left (add '0' to the code)
        if (node.left != null) {
            buildHuffmanCodesRecursive(node.left, code + "0", codes);
        }
        
        // Traverse right (add '1' to the code)
        if (node.right != null) {
            buildHuffmanCodesRecursive(node.right, code + "1", codes);
        }
    }
    
    public static String encode(String text, Map<Character, String> codes) {
        StringBuilder encodedText = new StringBuilder();
        
        for (char c : text.toCharArray()) {
            encodedText.append(codes.get(c));
        }
        
        return encodedText.toString();
    }
    
    public static String decode(String encodedText, Node root) {
        StringBuilder decodedText = new StringBuilder();
        Node current = root;
        
        for (char bit : encodedText.toCharArray()) {
            // Navigate the tree
            if (bit == '0') {
                current = current.left;
            } else {
                current = current.right;
            }
            
            // If we reach a leaf node
            if (current.character != null) {
                decodedText.append(current.character);
                current = root; // Reset to root for next character
            }
        }
        
        return decodedText.toString();
    }
    
    public static void main(String[] args) {
        String text = "this is an example for huffman encoding";
        
        // Build the Huffman tree
        Node root = buildHuffmanTree(text);
        
        // Build the Huffman codes
        Map<Character, String> codes = buildHuffmanCodes(root);
        
        // Encode the text
        String encodedText = encode(text, codes);
        
        System.out.println("Original text: " + text);
        System.out.println("Encoded text: " + encodedText);
        System.out.println("Huffman codes: " + codes);
        
        // Calculate compression ratio
        int originalSize = text.length() * 8; // assuming 8 bits per character
        int compressedSize = encodedText.length();
        double compressionRatio = (double) originalSize / compressedSize;
        
        System.out.println("Original size (bits): " + originalSize);
        System.out.println("Compressed size (bits): " + compressedSize);
        System.out.println("Compression ratio: " + String.format("%.2f", compressionRatio) + "x");
        
        // Verify decoding
        String decodedText = decode(encodedText, root);
        System.out.println("Decoded text: " + decodedText);
        System.out.println("Decoding successful: " + text.equals(decodedText));
    }
}
```

## Complexity Analysis

- **Time Complexity**: O(n log n), where n is the number of unique characters in the input text
  - Building the frequency table: O(n)
  - Building the Huffman tree: O(k log k), where k is the number of unique characters
  - Encoding/decoding: O(n)
- **Space Complexity**: O(n), for storing the Huffman tree and the codes

## Properties of Huffman Coding

1. **Prefix Property**: No code is a prefix of another code, which allows unambiguous decoding
2. **Optimality**: Produces the optimal variable-length codes for a given character distribution
3. **Lossless**: The original data can be perfectly reconstructed from the compressed data

## Applications

1. **File Compression**: Used in compression algorithms like GZIP and PKZIP
2. **Image Compression**: Component of JPEG compression
3. **Data Transmission**: Reduces the amount of data that needs to be transmitted
4. **JPEG Encoding**: Used in JPEG file format for image compression
5. **MP3 Compression**: Part of the MP3 audio compression algorithm

## Variations

1. **Adaptive Huffman Coding**: Updates the Huffman tree as it reads the input
2. **Canonical Huffman Coding**: A particular assignment of codewords that enables faster decoding
3. **Modified Huffman Coding for JPEG**: Used in JPEG image compression
4. **Huffman Coding with Unequal Letter Costs**: When different symbols have different transmission costs

## Advantages and Disadvantages

### Advantages
- Lossless compression
- Variable-length codes that adapt to the frequency of characters
- Simple to implement and efficient for many applications

### Disadvantages
- Requires knowledge of character frequencies in advance for static Huffman coding
- Two passes over the data are needed (one to calculate frequencies, one to encode)
- Not as efficient for adaptive compression as some more complex algorithms

## Practice Problems

1. [Huffman Encoding](https://practice.geeksforgeeks.org/problems/huffman-encoding3654/1) - Implement Huffman encoding
2. [Minimum Cost Tree From Leaf Values](https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/) - Similar tree construction problem
3. [Optimal File Merge Pattern](https://www.geeksforgeeks.org/optimal-file-merge-patterns/) - Related greedy algorithm problem

## References

1. Huffman, D. A. (1952). "A Method for the Construction of Minimum-Redundancy Codes". Proceedings of the IRE, 40(9), 1098-1101.
2. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
3. Sedgewick, R., & Wayne, K. (2011). Algorithms (4th ed.). Addison-Wesley Professional.
