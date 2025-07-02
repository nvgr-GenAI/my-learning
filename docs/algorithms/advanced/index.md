# Advanced Algorithmic Techniques

## 📋 Overview

Advanced algorithmic techniques represent sophisticated problem-solving paradigms that go beyond basic data structures and simple algorithms. These techniques are essential for tackling complex computational problems efficiently.

## 🔍 What You'll Learn

- **Divide & Conquer**: Break complex problems into manageable subproblems
- **Greedy Algorithms**: Make locally optimal choices for global optimization
- **Backtracking**: Systematic exploration of solution spaces
- **Advanced Patterns**: Sliding window, two pointers, bit manipulation

## 📚 Section Contents

### 🔀 Core Paradigms

- **[Divide & Conquer](divide-conquer.md)** - Recursive problem decomposition
- **[Greedy Algorithms](greedy.md)** - Local optimization strategies
- **[Backtracking](backtracking.md)** - Systematic search with pruning
- **[Branch & Bound](branch-bound.md)** - Optimization with bounds

### 🎯 Problem-Solving Techniques

- **[Two Pointers](two-pointers.md)** - Efficient array/string processing
- **[Sliding Window](sliding-window.md)** - Subarray/substring optimization
- **[Bit Manipulation](bit-manipulation.md)** - Efficient operations using bits
- **[String Algorithms](string-algorithms.md)** - Advanced string processing

### 🔍 Search & Optimization

- **[Binary Search Variants](binary-search.md)** - Advanced search techniques
- **[Ternary Search](ternary-search.md)** - Finding maximum/minimum in unimodal functions
- **[Meet in the Middle](meet-middle.md)** - Reducing exponential complexity

### 💪 Practice Problems

#### 🟢 Easy Problems
- **[Easy Advanced Problems](easy-problems.md)**
  - Basic divide & conquer, simple greedy
  - Two pointers, sliding window basics

#### 🟡 Medium Problems  
- **[Medium Advanced Problems](medium-problems.md)**
  - Complex backtracking, advanced greedy
  - Bit manipulation, string algorithms

#### 🔴 Hard Problems
- **[Hard Advanced Problems](hard-problems.md)**
  - Optimization problems, complex search spaces
  - Advanced mathematical techniques

## 🧠 Core Paradigm Comparison

| Technique | When to Use | Time Complexity | Examples |
|-----------|-------------|----------------|----------|
| **Divide & Conquer** | Recursive subproblems | Often O(n log n) | Merge Sort, Binary Search |
| **Greedy** | Local optimal → Global optimal | Usually O(n) to O(n log n) | Activity Selection, Huffman |
| **Backtracking** | Constraint satisfaction | Exponential (with pruning) | N-Queens, Sudoku |
| **Dynamic Programming** | Overlapping subproblems | Polynomial | Knapsack, LCS |

## 🔧 Divide & Conquer Framework

### Template Structure
```python
def divide_conquer(problem):
    # Base case
    if is_base_case(problem):
        return solve_base_case(problem)
    
    # Divide
    subproblems = divide(problem)
    
    # Conquer
    solutions = [divide_conquer(sub) for sub in subproblems]
    
    # Combine
    return combine(solutions)
```

### Classic Example: Merge Sort
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Combine
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

## 🎯 Greedy Algorithm Framework

### Key Properties
1. **Greedy Choice Property**: Local optimal choice leads to global optimal
2. **Optimal Substructure**: Optimal solution contains optimal subsolutions

### Template Structure
```python
def greedy_algorithm(problem):
    solution = []
    
    while not is_complete(solution):
        # Make greedy choice
        choice = select_best_option(problem, solution)
        solution.append(choice)
        
        # Update problem state
        problem = update_problem(problem, choice)
    
    return solution
```

### Classic Example: Activity Selection
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

## 🔍 Advanced Search Techniques

### Binary Search on Answer
```python
def binary_search_answer(left, right, check_function):
    """
    Find the optimal answer in a monotonic search space
    """
    while left < right:
        mid = (left + right) // 2
        
        if check_function(mid):
            right = mid  # Can do better
        else:
            left = mid + 1  # Need more
    
    return left
```

### Two Pointers Technique
```python
def two_sum_sorted(arr, target):
    """
    Find pair that sums to target in sorted array
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []
```

### Sliding Window Technique
```python
def max_sum_subarray_size_k(arr, k):
    """
    Find maximum sum of subarray of size k
    """
    if len(arr) < k:
        return 0
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

## 🎨 Advanced Problem Patterns

### 1. **Optimization Problems**
- Use greedy when local optimal guarantees global optimal
- Use DP when subproblems overlap
- Use divide & conquer when subproblems are independent

### 2. **Search Space Reduction**
- Binary search on answer for monotonic functions
- Two pointers for sorted arrays
- Meet in the middle for exponential problems

### 3. **Constraint Satisfaction**
- Backtracking with intelligent pruning
- Branch and bound for optimization
- Heuristic search for large spaces

## 💡 Problem-Solving Strategy

### Step 1: Identify the Paradigm
- **Optimization**: Greedy or DP?
- **Search**: Binary search variants?
- **Recursive structure**: Divide & conquer?
- **Constraints**: Backtracking needed?

### Step 2: Verify Prerequisites
- **Greedy**: Does greedy choice property hold?
- **Divide & Conquer**: Can problem be split independently?
- **Binary Search**: Is search space monotonic?

### Step 3: Design & Implement
- Follow paradigm template
- Handle edge cases
- Optimize if needed

### Step 4: Analyze Complexity
- Time: Often depends on problem structure
- Space: Consider recursion stack, auxiliary space

## 🚀 Applications

### System Design
- **Load Balancing**: Greedy algorithms for optimal distribution
- **Caching**: Optimal replacement strategies
- **Resource Allocation**: Optimization techniques

### Competitive Programming
- **Contest Problems**: Advanced techniques for complex problems
- **Optimization**: Finding best solutions under constraints
- **Mathematical**: Number theory combined with algorithms

### Machine Learning
- **Feature Selection**: Greedy approaches
- **Optimization**: Gradient descent and variants
- **Search**: Hyperparameter optimization

---

*Advanced techniques are the tools that separate good programmers from great ones. Master these paradigms to tackle any computational challenge!*
4. **Analyze complexity** - Time and space requirements
5. **Prove correctness** - Especially important for greedy algorithms
