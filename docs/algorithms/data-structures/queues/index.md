# Queues

## 📚 Overview

A **Queue** is a linear data structure that follows the **First In, First Out (FIFO)** principle. Think of it like a line at a store - the first person in line is the first person to be served. Elements are added at one end (rear/back) and removed from the other end (front).

## 🎯 What You'll Learn

This section covers everything you need to master queues:

### 📖 [Fundamentals & Theory](fundamentals.md)

- Queue concept and FIFO principle
- Basic operations (enqueue, dequeue, front, rear)
- Types of queues and their characteristics
- Time and space complexity analysis

### 🏗️ Implementation Types

Master different queue implementations for various scenarios:

#### [📘 Array-Based Queue](array-queue.md){ .md-button }

**Fixed/Dynamic**: Simple array implementation

- Contiguous memory layout
- O(1) operations with proper indexing
- Best for predictable sizes

#### [📗 Linked List Queue](linked-list-queue.md){ .md-button }

**Dynamic**: Node-based implementation

- True dynamic sizing
- O(1) enqueue and dequeue
- Best for unpredictable sizes

#### [📙 Circular Queue](circular-queue.md){ .md-button }

**Space Efficient**: Ring buffer approach

- Reuses array positions
- Optimal for fixed buffers
- Common in systems programming

#### [📕 Priority Queue](priority-queue.md){ .md-button }

**Priority-Based**: Order by importance

- Elements served by priority
- Essential for algorithms
- Multiple implementation options

### 🟢 [Easy Problems](easy-problems.md)

Perfect for beginners to understand queue applications:

- Implement Queue using Stacks
- Number of Recent Calls
- Design Circular Queue
- Moving Average from Data Stream
- Number of Students Unable to Eat Lunch

### 🟡 [Medium Problems](medium-problems.md)

Intermediate challenges using advanced queue techniques:

- Sliding Window Maximum
- Design Hit Counter
- Shortest Path in Binary Matrix
- Rotting Oranges
- Task Scheduler

### 🔴 [Hard Problems](hard-problems.md)

Advanced problems for queue mastery:

- Sliding Window Median
- Design Phone Directory
- Race Car
- Shortest Path to Get All Keys
- Minimum Window Substring

## 🚀 Quick Start

If you're new to queues, start with **[Fundamentals & Theory](fundamentals.md)** to understand the core FIFO concept, then explore the **[Implementation Types](array-queue.md)** to see different approaches, and finally progress through problems based on your comfort level.

## 📊 At a Glance

| **Operation** | **Time Complexity** | **Space Complexity** |
|---------------|-------------------|---------------------|
| **Enqueue** | O(1) | O(1) |
| **Dequeue** | O(1) | O(1) |
| **Front** | O(1) | O(1) |
| **Rear** | O(1) | O(1) |
| **isEmpty** | O(1) | O(1) |
| **Search** | O(n) | O(1) |

## 🎓 Learning Path

```mermaid
graph TD
    A[Start: Queues] --> B[Learn FIFO Principle]
    B --> C[Understand Basic Operations]
    C --> D[Choose Implementation Type]
    D --> E[Array-Based Queue]
    D --> F[Linked List Queue]
    D --> G[Circular Queue]
    D --> H[Priority Queue]
    E --> I[Practice Easy Problems]
    F --> I
    G --> I
    H --> I
    I --> J[Tackle Medium Problems] 
    J --> K[Master Hard Problems]
    K --> L[Apply to Real Projects]
    
    C --> M[Enqueue/Dequeue Mechanics]
    C --> N[Edge Case Handling]
    
    I --> O[Queue with Stacks]
    J --> P[Sliding Window Patterns]
    K --> Q[Advanced Data Structures]
```

## 🔄 Queue Types Overview

### 1. **Simple Queue (Linear Queue)**

Basic FIFO implementation with front and rear pointers.

### 2. **Circular Queue (Ring Buffer)**

Efficient space utilization by wrapping around when reaching array end.

### 3. **Priority Queue**

Elements are served based on priority rather than insertion order.

### 4. **Double-Ended Queue (Deque)**

Allows insertion and deletion at both ends.

## 🏆 Success Metrics

Track your progress:

- [ ] Understand FIFO principle thoroughly
- [ ] Choose appropriate implementation for your use case
- [ ] Implement basic queues using arrays and linked lists
- [ ] Master circular queues for space efficiency
- [ ] Understand priority queues and their applications
- [ ] Solve 5+ easy problems independently
- [ ] Master sliding window pattern with queues
- [ ] Solve 3+ medium problems with optimization

## 💡 Pro Tips

!!! tip "When to Use Queues"
    - **Task scheduling**: Operating systems, print queues
    - **BFS traversal**: Level-order tree/graph traversal
    - **Buffering**: IO operations, streaming data
    - **Request handling**: Web servers, API rate limiting
    - **Sliding window**: Moving averages, range queries

!!! warning "Common Pitfalls"
    - Array-based queues can waste space (use circular queues)
    - Not checking for empty queue before dequeue
    - Confusion between queue and stack operations
    - Inefficient implementation leading to O(n) operations

!!! success "Best Practices"
    - Always check if queue is empty before dequeue/front
    - Use circular queues for fixed-size scenarios
    - Consider deque when you need both-end operations
    - Think about queues for level-by-level processing

## 🔄 Real-World Applications

### System Programming

- **Process scheduling**: CPU task management
- **Buffer management**: IO operations, printer queues
- **Breadth-First Search**: Graph and tree algorithms

### Web Development

- **Request queues**: HTTP request handling
- **Message queues**: Microservices communication
- **Rate limiting**: API request throttling

### Data Processing

- **Stream processing**: Real-time data analysis
- **Cache replacement**: LRU, FIFO cache policies
- **Load balancing**: Distributing requests across servers

---

Ready to dive in? Start with **[Fundamentals & Theory](fundamentals.md)** to build your foundation!
