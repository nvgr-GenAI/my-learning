# Backtracking Fundamentals

## What is Backtracking?

Backtracking is an algorithmic technique for solving problems by trying to build a solution incrementally and abandoning candidates that fail to satisfy the problem constraints.

## Key Concepts

### Core Principles

1. **Incremental Construction**: Build solution step by step
2. **Constraint Checking**: Validate partial solutions
3. **Backtrack on Failure**: Undo invalid choices
4. **Exhaustive Search**: Explore all possibilities

### When to Use Backtracking

- **Constraint Satisfaction Problems**: N-Queens, Sudoku
- **Combinatorial Problems**: Permutations, combinations
- **Path Finding**: Maze solving, Hamilton paths
- **Optimization**: Finding all optimal solutions

## Generic Template

```python
def backtrack(solution, choices, constraints):
    # Base case: solution is complete
    if is_complete(solution):
        return process_solution(solution)
    
    # Try all possible choices
    for choice in get_choices(choices, solution):
        # Make choice
        solution.append(choice)
        
        # Check if choice is valid
        if is_valid(solution, constraints):
            # Recurse with new choice
            if backtrack(solution, choices, constraints):
                return True
        
        # Backtrack: undo choice
        solution.pop()
    
    return False
```

## Time and Space Complexity

### Time Complexity
- **Worst Case**: O(b^d) where b is branching factor, d is depth
- **Best Case**: O(d) if solution found quickly
- **Average**: Depends on constraint effectiveness

### Space Complexity
- **Stack Space**: O(d) for recursion depth
- **Solution Space**: O(d) for current path

## Common Patterns

### 1. Subset Generation
```python
def generate_subsets(nums):
    result = []
    
    def backtrack(start, path):
        result.append(path[:])  # Add current subset
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
```

### 2. Permutation Generation
```python
def generate_permutations(nums):
    result = []
    
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()
    
    backtrack([], nums)
    return result
```

### 3. Combination Generation
```python
def generate_combinations(n, k):
    result = []
    
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(1, [])
    return result
```

## Optimization Techniques

### 1. Pruning
```python
def backtrack_with_pruning(solution, target):
    if sum(solution) > target:  # Prune early
        return False
    
    if is_complete(solution):
        return sum(solution) == target
    
    # Continue search...
```

### 2. Constraint Propagation
```python
def backtrack_with_constraints(solution, constraints):
    # Check constraints early
    if not satisfies_constraints(solution, constraints):
        return False
    
    # Continue if constraints satisfied
```

### 3. Heuristic Ordering
```python
def backtrack_with_heuristics(choices):
    # Order choices by likelihood of success
    choices.sort(key=lambda x: estimate_success_probability(x))
    
    for choice in choices:
        # Try most promising choices first
        pass
```

## Common Applications

### 1. N-Queens Problem
- Place N queens on N×N chessboard
- No two queens attack each other
- Backtrack when placement violates constraints

### 2. Sudoku Solver
- Fill 9×9 grid with digits 1-9
- Each row, column, and 3×3 box contains all digits
- Backtrack when number placement is invalid

### 3. Maze Solving
- Find path from start to end
- Backtrack when hitting dead ends
- Mark visited cells to avoid cycles

### 4. Subset Sum
- Find subset that sums to target
- Backtrack when sum exceeds target
- Try including/excluding each element

## Best Practices

### 1. Early Termination
```python
def backtrack_early_termination(solution):
    # Check constraints as early as possible
    if not is_promising(solution):
        return False
```

### 2. State Management
```python
def backtrack_state_management(state):
    # Save state before making choice
    saved_state = save_state(state)
    
    # Make choice
    make_choice(state)
    
    # Recurse
    if backtrack(state):
        return True
    
    # Restore state
    restore_state(saved_state)
    return False
```

### 3. Memory Optimization
```python
def backtrack_memory_optimized(solution):
    # Use generators for large solution spaces
    def backtrack_generator():
        if is_complete(solution):
            yield solution[:]
        
        for choice in choices:
            solution.append(choice)
            yield from backtrack_generator()
            solution.pop()
```

## Debugging Tips

1. **Trace Execution**: Print decision points
2. **Visualize Search Tree**: Draw exploration path
3. **Check Base Cases**: Verify termination conditions
4. **Validate Constraints**: Ensure proper pruning
5. **Test Small Inputs**: Debug with simple cases

## Practice Problems

### Beginner
- Generate all subsets
- Generate all permutations
- Combination sum
- Letter combinations of phone number

### Intermediate
- N-Queens problem
- Sudoku solver
- Word search
- Palindrome partitioning

### Advanced
- Expression add operators
- Remove invalid parentheses
- Wildcard matching
- Regular expression matching
