# Greedy Algorithms

## 🎯 Overview

Greedy algorithms make locally optimal choices at each step, hoping to find a global optimum. They're characterized by their simple, intuitive approach and often provide efficient solutions to optimization problems, though they don't always guarantee the optimal solution.

## 📋 What You'll Learn

### 🎯 **Core Concepts**
- [Fundamentals](fundamentals.md) - Greedy choice property, optimal substructure
- [Analysis Techniques](analysis.md) - Proving correctness and optimality

### 📚 **Problem Categories**

#### **By Difficulty Level**
- [Easy Problems](easy-problems.md) - Basic greedy strategies
- [Medium Problems](medium-problems.md) - Complex optimization scenarios
- [Hard Problems](hard-problems.md) - Advanced greedy techniques

#### **By Problem Type**
- [Scheduling Problems](scheduling.md) - Activity selection, job scheduling
- [Graph Problems](graph-problems.md) - Minimum spanning tree, shortest paths
- [Array Problems](array-problems.md) - Jump games, candy distribution
- [String Problems](string-problems.md) - Pattern matching, compression

## 🔥 Why Greedy Algorithms Matter

- ✅ **Simplicity** - Easy to understand and implement
- ✅ **Efficiency** - Often linear or O(n log n) time complexity
- ✅ **Intuitive** - Mirror human decision-making process
- ✅ **Practical** - Many real-world applications
- ✅ **Foundation** - Building blocks for complex algorithms

## 🎨 The Greedy Template

```python
def greedy_algorithm(problem_input):
    # Initialize solution
    solution = []
    
    # Sort or organize input based on greedy criteria
    candidates = sort_by_greedy_criteria(problem_input)
    
    # Make greedy choices
    for candidate in candidates:
        if is_feasible(candidate, solution):
            solution.append(candidate)
    
    return solution
```

## 🧠 Key Properties

### 1. **Greedy Choice Property**
- A globally optimal solution can be arrived at by making locally optimal choices
- Once a choice is made, it's never reconsidered

### 2. **Optimal Substructure**
- An optimal solution contains optimal solutions to subproblems
- Similar to dynamic programming, but no overlapping subproblems

### 3. **No Backtracking**
- Decisions are final and irreversible
- Much simpler than backtracking approaches

## 🏆 Classic Greedy Problems

### **Activity Selection Problem**
```python
def activity_selection(activities):
    """
    Select maximum number of non-overlapping activities
    Greedy choice: Always pick activity that finishes earliest
    """
    # Sort by finish time
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_finish = activities[0][1]
    
    for start, finish in activities[1:]:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish
    
    return selected
```

### **Fractional Knapsack**
```python
def fractional_knapsack(capacity, items):
    """
    Maximize value with fractional items allowed
    Greedy choice: Pick items with highest value-to-weight ratio
    """
    # Sort by value-to-weight ratio (descending)
    items.sort(key=lambda x: x[1]/x[0], reverse=True)
    
    total_value = 0
    for weight, value in items:
        if capacity >= weight:
            # Take entire item
            total_value += value
            capacity -= weight
        else:
            # Take fraction of item
            total_value += value * (capacity / weight)
            break
    
    return total_value
```

### **Huffman Coding**
```python
import heapq
from collections import defaultdict, Counter

def huffman_coding(text):
    """
    Build optimal prefix-free code
    Greedy choice: Always merge two least frequent nodes
    """
    # Count frequencies
    freq = Counter(text)
    
    # Create priority queue
    heap = [[freq[char], char] for char in freq]
    heapq.heapify(heap)
    
    # Build Huffman tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # Create internal node
        merged = [left[0] + right[0], left, right]
        heapq.heappush(heap, merged)
    
    # Build code table
    codes = {}
    def build_codes(node, code=""):
        if isinstance(node[1], str):  # Leaf node
            codes[node[1]] = code or "0"
        else:  # Internal node
            build_codes(node[1], code + "0")
            build_codes(node[2], code + "1")
    
    build_codes(heap[0])
    return codes
```

## 📊 Common Greedy Strategies

### **1. Sort-Based Greedy**
```python
def job_scheduling(jobs):
    """Sort jobs by deadline, then by profit"""
    jobs.sort(key=lambda x: (x.deadline, -x.profit))
    # Apply greedy selection...
```

### **2. Priority-Based Greedy**
```python
def dijkstra_shortest_path(graph, start):
    """Always select unvisited node with minimum distance"""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        # Greedy choice: process nearest unvisited node
```

### **3. Exchange-Based Greedy**
```python
def coin_change_greedy(amount, coins):
    """Use largest denomination first (only works for certain coin systems)"""
    coins.sort(reverse=True)
    result = []
    
    for coin in coins:
        while amount >= coin:
            result.append(coin)
            amount -= coin
    
    return result if amount == 0 else None
```

## ⚠️ When Greedy Fails

### **Coin Change Problem**
```python
# Greedy fails with coins = [1, 3, 4] and amount = 6
# Greedy: 4 + 1 + 1 = 3 coins
# Optimal: 3 + 3 = 2 coins

def coin_change_dp(amount, coins):
    """Dynamic programming solution for optimal coin change"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

### **0/1 Knapsack Problem**
```python
# Greedy by value-to-weight ratio doesn't work
# Need dynamic programming for optimal solution

def knapsack_01_dp(capacity, weights, values):
    """Dynamic programming solution for 0/1 knapsack"""
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    values[i-1] + dp[i-1][w - weights[i-1]],
                    dp[i-1][w]
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]
```

## 🔍 Proving Greedy Correctness

### **Method 1: Exchange Argument**
1. Assume optimal solution differs from greedy
2. Show we can exchange choices to match greedy
3. Prove exchange doesn't worsen the solution

### **Method 2: Greedy Stays Ahead**
1. Show greedy solution is always "ahead" of any other solution
2. Prove this lead is maintained at each step

### **Method 3: Contradiction**
1. Assume greedy doesn't produce optimal solution
2. Derive contradiction from this assumption

## 📈 Complexity Analysis

### **Time Complexity Patterns**
- **Sorting-based**: Usually O(n log n) due to sorting
- **Priority queue**: O(n log n) for heap operations
- **Linear scan**: O(n) for simple greedy choices

### **Space Complexity**
- Often O(1) extra space beyond input
- Sometimes O(n) for auxiliary data structures

## 🎯 Problem Recognition

### **Use Greedy When:**
- Local optimal choices lead to global optimum
- Problem has optimal substructure
- No need to reconsider previous decisions
- Simple, intuitive solution exists

### **Don't Use Greedy When:**
- Multiple factors need simultaneous optimization
- Future decisions depend on current choices
- Optimal substructure doesn't hold

## 📚 Learning Path

### **Step 1: Understand Theory**
- Master greedy choice property
- Learn proof techniques
- Study classic problems

### **Step 2: Practice Recognition**
- Identify when greedy works
- Understand when it fails
- Compare with DP solutions

### **Step 3: Implementation Skills**
- Efficient sorting strategies
- Priority queue usage
- Clean code structure

### **Step 4: Advanced Applications**
- Graph algorithms (MST, shortest paths)
- Approximation algorithms
- Online algorithms

## 🚀 Quick Start

Ready to master greedy algorithms?

1. **📚 Learn [Fundamentals](fundamentals.md)** - Build theoretical foundation
2. **🎯 Practice [Easy Problems](easy-problems.md)** - Apply basic strategies
3. **🧠 Challenge [Medium Problems](medium-problems.md)** - Develop intuition
4. **🏆 Master [Hard Problems](hard-problems.md)** - Handle complex scenarios

---

!!! warning "Remember"
    Greedy algorithms are seductive in their simplicity, but they don't always work! Always verify that the greedy choice property holds for your specific problem.

!!! success "Pro Tip"
    When greedy works, it often provides the most elegant and efficient solution. When it doesn't, understanding why helps you choose the right alternative approach.

*Begin your greedy journey with [fundamentals](fundamentals.md) to build a solid foundation!*
