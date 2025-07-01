# Data Structures

## Overview

Data structures are fundamental building blocks in computer science that organize and store data efficiently. Understanding different data structures and their trade-offs is crucial for writing efficient algorithms.

## Core Data Structures

### Linear Data Structures

**[Arrays](arrays.md)**
- Fixed-size sequential collection
- O(1) access time
- Foundation for many other structures

**[Linked Lists](linked-lists.md)**
- Dynamic size collection with nodes
- Efficient insertion/deletion
- Singly, doubly, and circular variants

**[Stacks & Queues](stacks-queues.md)**
- Stack: LIFO (Last In, First Out)
- Queue: FIFO (First In, First Out)
- Essential for algorithms and system design

### Non-Linear Data Structures

**[Trees](trees.md)**
- Hierarchical structure with root and children
- Binary trees, BSTs, AVL trees, etc.
- Efficient searching and sorting

**[Hash Tables](hash-tables.md)**
- Key-value mapping with hash functions
- Average O(1) operations
- Critical for fast lookups

**[Heaps](heaps.md)**
- Complete binary tree with heap property
- Priority queues implementation
- Heap sort algorithm

## Data Structure Comparison

| Data Structure | Access | Search | Insertion | Deletion | Space |
|----------------|--------|--------|-----------|----------|-------|
| Array          | O(1)   | O(n)   | O(n)      | O(n)     | O(n)  |
| Linked List    | O(n)   | O(n)   | O(1)      | O(1)     | O(n)  |
| Stack          | O(n)   | O(n)   | O(1)      | O(1)     | O(n)  |
| Queue          | O(n)   | O(n)   | O(1)      | O(1)     | O(n)  |
| Hash Table     | O(1)   | O(1)   | O(1)      | O(1)     | O(n)  |
| Binary Tree    | O(n)   | O(n)   | O(n)      | O(n)     | O(n)  |
| BST (balanced) | O(log n) | O(log n) | O(log n) | O(log n) | O(n) |
| Heap           | O(1)   | O(n)   | O(log n)  | O(log n) | O(n)  |

## Choosing the Right Data Structure

### When to Use Arrays
- Need random access to elements
- Memory is a constraint
- Size is known and relatively fixed

### When to Use Linked Lists
- Frequent insertions/deletions
- Size varies significantly
- Don't need random access

### When to Use Hash Tables
- Need fast lookups
- Key-value relationships
- Caching scenarios

### When to Use Trees
- Hierarchical data
- Need sorted order
- Range queries

### When to Use Heaps
- Priority-based processing
- Finding min/max efficiently
- Implementing priority queues

## Study Plan

1. **Week 1**: Arrays and Linked Lists
2. **Week 2**: Stacks, Queues, and Hash Tables
3. **Week 3**: Trees (Binary Trees, BST)
4. **Week 4**: Heaps and Advanced Trees

## Common Interview Topics

- Implement basic data structures from scratch
- Choose appropriate data structure for given problem
- Optimize time/space complexity using different structures
- Understand trade-offs between different approaches

## Practice Resources

- LeetCode Data Structure problems
- Implement each data structure in your preferred language
- Solve problems that require combining multiple data structures
