# Linked Lists 🔗

## 📚 Master Linked List Data Structures

Linked Lists are fundamental dynamic data structures where elements (nodes) are stored in sequence, with each node containing data and a reference (pointer) to the next node. Master different types of linked lists and their applications.

---

## 🎯 What You'll Learn

### Core Concepts

- **Node Structure**: Data and pointer components
- **Memory Management**: Dynamic allocation and deallocation
- **Pointer Manipulation**: Traversal, insertion, and deletion
- **List Variants**: Singly, doubly, and circular linked lists

### Skills Development

- **Algorithm Design**: Implement efficient linked list operations
- **Problem Solving**: Handle edge cases and special scenarios
- **Memory Optimization**: Understand space vs time trade-offs
- **Real-world Applications**: When to use different list types

---

## 📖 Learning Path

### 1. 🟢 Fundamentals & Theory

Start with the theoretical foundation and core concepts.

[📚 Fundamentals](fundamentals.md){ .md-button .md-button--primary }

Core theoretical concepts:

- Memory layout and pointers
- Time and space complexity analysis
- Comparison with arrays and other data structures
- When to use different linked list types

### 2. 🔧 Core Implementations

Master fundamental linked list types and their implementations.

#### [📘 Singly Linked List](singly-linked-list.md){ .md-button }

**Foundation**: Learn the basics

- Node structure and operations
- Insertion, deletion, and traversal
- Simple and efficient implementation

#### [📗 Doubly Linked List](doubly-linked-list.md){ .md-button }

**Bidirectional**: Enhanced navigation

- Forward and backward traversal
- More complex but more flexible
- Use cases: browsers, music players

#### [📙 Circular Linked List](circular-linked-list.md){ .md-button }

**Continuous**: Loop-based structure

- Circular connections
- Round-robin applications
- Josephus problem solutions

### 3. 🟢 Easy Problems

Build confidence with fundamental linked list problems.

[🎯 Easy Problems](easy-problems.md){ .md-button }

Perfect for beginners:

- Reverse a linked list
- Remove duplicates
- Find middle element
- Detect cycles

### 4. 🟡 Medium Problems

Tackle more complex linked list algorithms.

[⚡ Medium Problems](medium-problems.md){ .md-button }

Intermediate challenges:

- Merge sorted lists
- Add two numbers
- Intersection of lists
- Reorder lists

### 5. 🔴 Hard Problems

Master advanced linked list techniques.

[🚀 Hard Problems](hard-problems.md){ .md-button }

Advanced challenges:

- Reverse nodes in k-groups
- Copy list with random pointers
- LRU cache implementation
- Advanced cycle detection

---

## 🔍 Quick Reference

### Types Comparison

| **Type** | **Memory/Node** | **Traversal** | **Insertion** | **Use Cases** |
|----------|----------------|---------------|---------------|---------------|
| **Singly** | 1 pointer | Forward only | O(1) at head | General purpose |
| **Doubly** | 2 pointers | Bidirectional | O(1) both ends | Navigation, undo/redo |
| **Circular** | 1 pointer + loop | Continuous | O(1) at head | Round-robin, buffers |

### Time Complexities

| **Operation** | **Singly** | **Doubly** | **Circular** |
|---------------|------------|------------|--------------|
| **Access** | O(n) | O(n) | O(n) |
| **Search** | O(n) | O(n) | O(n) |
| **Insert Head** | O(1) | O(1) | O(n) |
| **Insert Tail** | O(n) | O(1) | O(n) |
| **Delete Head** | O(1) | O(1) | O(n) |
| **Delete Tail** | O(n) | O(1) | O(n) |

---

## 💡 Key Insights

### ✅ When to Use Linked Lists

- **Dynamic size**: Size unknown at compile time
- **Frequent insertions/deletions**: Especially at beginning
- **Memory efficiency**: No wasted space (unlike arrays)
- **Sequential access**: Don't need random access

### ❌ When to Avoid Linked Lists

- **Random access needed**: Use arrays instead
- **Memory overhead matters**: Pointers add overhead
- **Cache performance critical**: Arrays have better locality
- **Simple data structures sufficient**: Don't over-engineer

### 🎯 Problem-Solving Patterns

1. **Two Pointers**: Fast and slow pointer technique
2. **Dummy Head**: Simplify edge cases in insertions/deletions
3. **Recursion**: Natural fit for linked list problems
4. **Stack**: For problems requiring reverse order processing

---

## 🚀 Next Steps

After mastering linked lists, explore related topics:

- **[Stacks](../stacks/index.md)**: LIFO data structure often implemented with linked lists
- **[Queues](../queues/index.md)**: FIFO data structure with linked list implementation
- **[Trees](../../trees/index.md)**: Tree structures use similar node concepts
- **[Graphs](../../graphs/index.md)**: Adjacency lists use linked list concepts

---

Start with **[Fundamentals](fundamentals.md)** to understand the theory and concepts, then explore **[Singly Linked List](singly-linked-list.md)** to build your foundation, and progress through the different types and problem difficulties. Each section builds upon the previous one, ensuring comprehensive understanding! 🎓
