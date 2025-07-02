# Backtracking - Medium Problems

## Problem Categories

### 1. Constraint Satisfaction
- N-Queens
- Sudoku solver
- Word search

### 2. Path Finding
- Unique paths with obstacles
- Maze solving
- Knight's tour

### 3. Combinatorial Optimization
- Partition to k equal sum subsets
- Palindrome partitioning
- Restore IP addresses

---

## 1. N-Queens Problem

**Problem**: Place `n` queens on an `n×n` chessboard such that no two queens attack each other.

**Example**:
```
Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
```

**Solution**:
```python
def solveNQueens(n):
    """
    Solve N-Queens problem using backtracking.
    
    Time Complexity: O(n!) - factorial solutions to try
    Space Complexity: O(n) - recursion depth and board storage
    """
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonal (top-left to bottom-right)
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        
        # Check diagonal (top-right to bottom-left)
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        
        return True
    
    def backtrack(row):
        # Base case: all queens placed
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        # Try placing queen in each column of current row
        for col in range(n):
            if is_safe(row, col):
                # Place queen
                board[row][col] = 'Q'
                # Recurse to next row
                backtrack(row + 1)
                # Backtrack
                board[row][col] = '.'
    
    backtrack(0)
    return result

# Optimized version using sets for O(1) conflict checking
def solveNQueens_optimized(n):
    result = []
    
    def backtrack(row, cols, diag1, diag2, board):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            # Check conflicts using sets
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            
            # Place queen
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            
            # Recurse
            backtrack(row + 1, cols, diag1, diag2, board)
            
            # Backtrack
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
    
    board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(0, set(), set(), set(), board)
    return result
```

**Key Points**:
- Check column and both diagonals for conflicts
- Use sets for O(1) conflict detection optimization
- Diagonal formulas: `row - col` and `row + col`

---

## 2. Word Search

**Problem**: Given a 2D board and a word, find if the word exists in the grid. Words can be constructed from letters of adjacent cells.

**Example**:
```
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
```

**Solution**:
```python
def exist(board, word):
    """
    Search for word in 2D board using backtracking.
    
    Time Complexity: O(m*n*4^k) where m,n are board dimensions, k is word length
    Space Complexity: O(k) - recursion depth up to word length
    """
    if not board or not board[0] or not word:
        return False
    
    rows, cols = len(board), len(board[0])
    
    def backtrack(row, col, index):
        # Base case: found complete word
        if index == len(word):
            return True
        
        # Check boundaries and character match
        if (row < 0 or row >= rows or col < 0 or col >= cols or 
            board[row][col] != word[index]):
            return False
        
        # Mark cell as visited
        temp = board[row][col]
        board[row][col] = '#'
        
        # Explore all 4 directions
        found = (backtrack(row + 1, col, index + 1) or
                backtrack(row - 1, col, index + 1) or
                backtrack(row, col + 1, index + 1) or
                backtrack(row, col - 1, index + 1))
        
        # Backtrack: restore cell value
        board[row][col] = temp
        
        return found
    
    # Try starting from each cell
    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True
    
    return False

# Alternative using visited set instead of modifying board
def exist_with_visited(board, word):
    rows, cols = len(board), len(board[0])
    
    def backtrack(row, col, index, visited):
        if index == len(word):
            return True
        
        if (row < 0 or row >= rows or col < 0 or col >= cols or
            (row, col) in visited or board[row][col] != word[index]):
            return False
        
        visited.add((row, col))
        
        found = (backtrack(row + 1, col, index + 1, visited) or
                backtrack(row - 1, col, index + 1, visited) or
                backtrack(row, col + 1, index + 1, visited) or
                backtrack(row, col - 1, index + 1, visited))
        
        visited.remove((row, col))
        return found
    
    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0, set()):
                return True
    
    return False
```

**Key Points**:
- Mark cells as visited to avoid cycles
- Explore all 4 directions from each valid position
- Backtrack by restoring original cell value
- Alternative: use visited set instead of modifying board

---

## 3. Palindrome Partitioning

**Problem**: Given a string `s`, partition `s` such that every substring of the partition is a palindrome. Return all possible palindrome partitioning.

**Example**:
```
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
```

**Solution**:
```python
def partition(s):
    """
    Find all palindrome partitions using backtracking.
    
    Time Complexity: O(2^n * n) - 2^n partitions, O(n) to check palindrome
    Space Complexity: O(n) - recursion depth
    """
    result = []
    
    def is_palindrome(string):
        return string == string[::-1]
    
    def backtrack(start, path):
        # Base case: processed entire string
        if start == len(s):
            result.append(path[:])
            return
        
        # Try all possible ending positions
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                path.append(substring)
                backtrack(end, path)
                path.pop()
    
    backtrack(0, [])
    return result

# Optimized version with palindrome memoization
def partition_optimized(s):
    n = len(s)
    # Precompute palindrome matrix
    is_pal = [[False] * n for _ in range(n)]
    
    # Single characters are palindromes
    for i in range(n):
        is_pal[i][i] = True
    
    # Check for palindromes of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            is_pal[i][i + 1] = True
    
    # Check for palindromes of length 3 and more
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and is_pal[i + 1][j - 1]:
                is_pal[i][j] = True
    
    result = []
    
    def backtrack(start, path):
        if start == n:
            result.append(path[:])
            return
        
        for end in range(start, n):
            if is_pal[start][end]:
                path.append(s[start:end + 1])
                backtrack(end + 1, path)
                path.pop()
    
    backtrack(0, [])
    return result
```

**Key Points**:
- Try all possible substring endings from current position
- Check if substring is palindrome before recursing
- Optimization: precompute palindrome matrix

---

## 4. Sudoku Solver

**Problem**: Write a program to solve a Sudoku puzzle by filling the empty cells.

**Example**:
```
Input: board = [["5","3",".",".","7",".",".",".","."],
                ["6",".",".","1","9","5",".",".","."],
                [".","9","8",".",".",".",".","6","."],
                ["8",".",".",".","6",".",".",".","3"],
                ["4",".",".","8",".","3",".",".","1"],
                ["7",".",".",".","2",".",".",".","6"],
                [".","6",".",".",".",".","2","8","."],
                [".",".",".","4","1","9",".",".","5"],
                [".",".",".",".","8",".",".","7","9"]]
```

**Solution**:
```python
def solveSudoku(board):
    """
    Solve Sudoku puzzle using backtracking.
    
    Time Complexity: O(9^(empty_cells)) - worst case
    Space Complexity: O(1) - modify board in place
    """
    def is_valid(row, col, num):
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    # Try digits 1-9
                    for num in '123456789':
                        if is_valid(i, j, num):
                            board[i][j] = num
                            
                            if backtrack():
                                return True
                            
                            # Backtrack
                            board[i][j] = '.'
                    
                    return False  # No valid number found
        
        return True  # All cells filled
    
    backtrack()

# Optimized version with constraint propagation
def solveSudoku_optimized(board):
    def get_candidates(row, col):
        if board[row][col] != '.':
            return []
        
        used = set()
        
        # Check row
        for j in range(9):
            if board[row][j] != '.':
                used.add(board[row][j])
        
        # Check column
        for i in range(9):
            if board[i][col] != '.':
                used.add(board[i][col])
        
        # Check box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] != '.':
                    used.add(board[i][j])
        
        return [num for num in '123456789' if num not in used]
    
    def find_best_cell():
        min_candidates = 10
        best_cell = None
        
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    candidates = get_candidates(i, j)
                    if len(candidates) < min_candidates:
                        min_candidates = len(candidates)
                        best_cell = (i, j, candidates)
        
        return best_cell
    
    def backtrack():
        cell_info = find_best_cell()
        if not cell_info:
            return True  # All cells filled
        
        row, col, candidates = cell_info
        
        for num in candidates:
            board[row][col] = num
            
            if backtrack():
                return True
            
            board[row][col] = '.'
        
        return False
    
    backtrack()
```

**Key Points**:
- Validate placement against row, column, and 3×3 box constraints
- Optimization: choose cell with fewest candidates (MRV heuristic)
- Constraint propagation to reduce search space

---

## 5. Restore IP Addresses

**Problem**: Given a string `s` containing only digits, return all possible valid IP addresses that can be obtained from `s`.

**Example**:
```
Input: s = "25525511135"
Output: ["255.255.11.135","255.255.111.35"]
```

**Solution**:
```python
def restoreIpAddresses(s):
    """
    Generate all valid IP addresses using backtracking.
    
    Time Complexity: O(3^4) = O(81) - at most 3 choices for each of 4 parts
    Space Complexity: O(1) - constant recursion depth
    """
    result = []
    
    def is_valid_part(part):
        # Check length
        if len(part) > 3 or len(part) == 0:
            return False
        
        # Check leading zeros
        if len(part) > 1 and part[0] == '0':
            return False
        
        # Check range
        return 0 <= int(part) <= 255
    
    def backtrack(start, path):
        # Base case: 4 parts found
        if len(path) == 4:
            if start == len(s):  # Used all characters
                result.append('.'.join(path))
            return
        
        # Pruning: too few characters left for remaining parts
        remaining_parts = 4 - len(path)
        remaining_chars = len(s) - start
        if remaining_chars < remaining_parts or remaining_chars > remaining_parts * 3:
            return
        
        # Try different lengths for current part
        for length in range(1, 4):
            if start + length <= len(s):
                part = s[start:start + length]
                if is_valid_part(part):
                    path.append(part)
                    backtrack(start + length, path)
                    path.pop()
    
    backtrack(0, [])
    return result

# Alternative iterative approach
def restoreIpAddresses_iterative(s):
    if len(s) < 4 or len(s) > 12:
        return []
    
    result = []
    
    # Try all possible positions for dots
    for i in range(1, min(4, len(s))):
        for j in range(i + 1, min(i + 4, len(s))):
            for k in range(j + 1, min(j + 4, len(s))):
                part1 = s[:i]
                part2 = s[i:j]
                part3 = s[j:k]
                part4 = s[k:]
                
                if (is_valid_part(part1) and is_valid_part(part2) and
                    is_valid_part(part3) and is_valid_part(part4)):
                    result.append(f"{part1}.{part2}.{part3}.{part4}")
    
    return result
```

**Key Points**:
- Validate each part: length ≤ 3, no leading zeros (except "0"), value ≤ 255
- Pruning: check if remaining characters can form valid parts
- Alternative: try all possible dot positions iteratively

---

## Common Patterns for Medium Problems

### 1. Constraint Validation
```python
def is_valid_move(state, move):
    # Check all relevant constraints
    return all_constraints_satisfied(state, move)

def backtrack(state):
    if is_complete(state):
        return True
    
    for move in get_possible_moves(state):
        if is_valid_move(state, move):
            make_move(state, move)
            if backtrack(state):
                return True
            undo_move(state, move)
    
    return False
```

### 2. Optimization Heuristics
```python
def backtrack_with_heuristics(state):
    if is_complete(state):
        return True
    
    # Choose most constrained variable (MRV)
    best_choice = min(get_choices(state), key=lambda x: len(get_options(x)))
    
    # Order values by least constraining value (LCV)
    options = sorted(get_options(best_choice), key=lambda x: count_conflicts(x))
    
    for option in options:
        if is_valid(best_choice, option):
            apply_choice(best_choice, option)
            if backtrack_with_heuristics(state):
                return True
            undo_choice(best_choice, option)
    
    return False
```

### 3. Early Pruning
```python
def backtrack_with_pruning(state, target):
    # Prune impossible branches early
    if not can_reach_target(state, target):
        return False
    
    if satisfies_target(state, target):
        return True
    
    for choice in get_choices(state):
        if is_promising(state, choice, target):
            make_choice(state, choice)
            if backtrack_with_pruning(state, target):
                return True
            undo_choice(state, choice)
    
    return False
```

These medium-level problems introduce more complex constraint handling and optimization techniques that are essential for tackling advanced backtracking challenges.
