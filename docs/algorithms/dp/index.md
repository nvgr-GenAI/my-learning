# Dynamic Programming

## 📋 Overview

Dynamic Programming (DP) is a powerful algorithmic technique for solving optimization problems by breaking them down into simpler subproblems. It's particularly effective when subproblems overlap and the optimal solution can be constructed from optimal solutions to subproblems.

## 🔍 What You'll Learn

- **Core Concepts**: Overlapping subproblems, optimal substructure, memoization vs tabulation
- **Classical Problems**: Fibonacci, knapsack, longest common subsequence, edit distance
- **Advanced Patterns**: 2D DP, state machines, probability DP, interval DP

## 📚 Section Contents

### 🎯 Fundamentals

- **[DP Fundamentals](fundamentals.md)** - Core concepts, memoization vs tabulation
- **[Common Patterns](patterns.md)** - Recognition techniques and solution templates

### 📖 Classical Problems

- **[Fibonacci & Linear DP](fibonacci.md)** - Simple recurrence relations
- **[0/1 Knapsack](knapsack.md)** - Classic optimization problem  
- **[Longest Common Subsequence](lcs.md)** - String/sequence problems
- **[Edit Distance](edit-distance.md)** - String transformation problems

### 💪 Practice by Difficulty

#### 🟢 Easy Problems
- **[Easy DP Problems](easy-problems.md)**
  - Climbing Stairs, House Robber, Maximum Subarray
  - Basic recurrence relations

#### 🟡 Medium Problems  
- **[Medium DP Problems](medium-problems.md)**
  - Coin Change, Unique Paths, Word Break
  - 2D DP, optimization problems

#### 🔴 Hard Problems
- **[Hard DP Problems](hard-problems.md)**
  - Regular Expression Matching, Burst Balloons
  - Complex state spaces, interval DP

### 🎨 Advanced Topics

- **[2D Dynamic Programming](2d-dp.md)** - Grid problems, path counting
- **[Interval DP](interval-dp.md)** - Matrix chain multiplication, palindrome partitioning
- **[State Machine DP](state-machine.md)** - Buy/sell stock, game theory
- **[Probability DP](probability-dp.md)** - Expected value problems

## 🧠 Key Concepts

### 1. **Optimal Substructure**
```python
# Problem can be broken into optimal subproblems
def fibonacci(n):
    # fib(n) = fib(n-1) + fib(n-2)
    # Optimal solution uses optimal solutions to subproblems
    pass
```

### 2. **Overlapping Subproblems**
```python
# Same subproblems solved multiple times
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)  # Recalculates same values
```

### 3. **Memoization (Top-Down)**
```python
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
```

### 4. **Tabulation (Bottom-Up)**
```python
def fib_dp(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

## 📊 DP Problem Categories

| Category | Example Problems |
|----------|------------------|
| **Linear DP** | Fibonacci, Climbing Stairs, House Robber |
| **Grid DP** | Unique Paths, Minimum Path Sum |
| **Interval DP** | Matrix Chain Multiplication, Burst Balloons |
| **Subset DP** | 0/1 Knapsack, Subset Sum |
| **String DP** | Edit Distance, LCS, Word Break |
| **Tree DP** | House Robber III, Diameter of Binary Tree |
| **State Machine** | Best Time to Buy/Sell Stock |

## 🔧 Problem-Solving Framework

### Step 1: Identify DP Problem
- ✅ Optimization problem (min/max/count)
- ✅ Overlapping subproblems
- ✅ Optimal substructure
- ✅ Recursive nature

### Step 2: Define State
```python
# What does dp[i] represent?
# Examples:
# dp[i] = maximum profit using first i items
# dp[i][j] = minimum cost to reach cell (i,j)
# dp[i][j] = LCS of first i chars of s1 and first j chars of s2
```

### Step 3: Find Recurrence Relation
```python
# How does dp[i] relate to previous states?
# Examples:
# dp[i] = max(dp[i-1], dp[i-2] + nums[i])  # House Robber
# dp[i][j] = dp[i-1][j] + dp[i][j-1]      # Unique Paths
```

### Step 4: Handle Base Cases
```python
# What are the simplest cases?
# dp[0] = ?
# dp[1] = ?
```

### Step 5: Determine Fill Order
- Bottom-up: Fill from base cases to final answer
- Top-down: Use recursion with memoization

## 🚀 Getting Started

**New to DP?** Start with:
1. **[DP Fundamentals](fundamentals.md)** - Understand core concepts
2. **[Fibonacci](fibonacci.md)** - Simple linear DP
3. **[Easy Problems](easy-problems.md)** - Practice basic patterns

**Have some experience?** Jump to:
- **[Medium Problems](medium-problems.md)** for optimization challenges
- **[2D DP](2d-dp.md)** for grid-based problems

**Advanced practitioner?** Challenge yourself with:
- **[Hard Problems](hard-problems.md)** for complex state spaces
- **[Interval DP](interval-dp.md)** for advanced techniques

## 💡 Pro Tips

1. **Start Simple**: Always try recursive solution first
2. **Draw it Out**: Visualize state transitions
3. **Space Optimization**: Often can reduce space complexity
4. **Pattern Recognition**: Many problems follow similar patterns
5. **Practice Regularly**: DP intuition comes with experience

---

*Dynamic Programming is often considered one of the most challenging topics in algorithms, but once mastered, it becomes an incredibly powerful problem-solving tool!*
