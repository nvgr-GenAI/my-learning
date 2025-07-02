# Grid Problems

This section covers backtracking algorithms for solving grid-based problems like N-Queens, Sudoku, and path finding.

## Core Concepts

### Grid Traversal

- Navigate through 2D grids with constraints
- Track visited cells and valid moves
- Backtrack when hitting dead ends

### Constraint Satisfaction

- Multiple constraints must be satisfied simultaneously
- Use pruning to eliminate invalid states early
- Check constraints before making moves

---

## Problems

### 1. N-Queens Problem

**Problem**: Place N queens on an N×N chessboard so no queens attack each other.

```python
def solve_n_queens(n):
    """Solve N-Queens problem and return all solutions."""
    result = []
    
    def is_safe(board, row, col):
        # Check column
        for i in range(row):
            if board[i] == col:
                return False
        
        # Check diagonal (top-left to bottom-right)
        for i in range(row):
            if board[i] - col == i - row:
                return False
        
        # Check diagonal (top-right to bottom-left)
        for i in range(row):
            if board[i] - col == row - i:
                return False
        
        return True
    
    def backtrack(board, row):
        if row == n:
            # Convert board to string representation
            solution = []
            for r in range(n):
                row_str = '.' * n
                row_str = row_str[:board[r]] + 'Q' + row_str[board[r] + 1:]
                solution.append(row_str)
            result.append(solution)
            return
        
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(board, row + 1)
    
    backtrack([-1] * n, 0)
    return result

def count_n_queens(n):
    """Count total number of N-Queens solutions."""
    count = 0
    
    def is_safe(positions, row, col):
        for i in range(row):
            if (positions[i] == col or 
                positions[i] - col == i - row or 
                positions[i] - col == row - i):
                return False
        return True
    
    def backtrack(positions, row):
        nonlocal count
        if row == n:
            count += 1
            return
        
        for col in range(n):
            if is_safe(positions, row, col):
                positions[row] = col
                backtrack(positions, row + 1)
    
    backtrack([-1] * n, 0)
    return count
```

### 2. Sudoku Solver

**Problem**: Solve a 9×9 Sudoku puzzle using backtracking.

```python
def solve_sudoku(board):
    """Solve Sudoku puzzle in-place."""
    
    def is_valid(board, row, col, num):
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == num:
                    return False
        
        return True
    
    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = '.'
                    return False
        return True
    
    backtrack()
    return board

# Optimized version with better cell selection
def solve_sudoku_optimized(board):
    """Optimized Sudoku solver."""
    
    def get_candidates(board, row, col):
        """Get possible values for a cell."""
        used = set()
        
        # Check row
        for j in range(9):
            if board[row][j] != '.':
                used.add(board[row][j])
        
        # Check column
        for i in range(9):
            if board[i][col] != '.':
                used.add(board[i][col])
        
        # Check 3x3 box
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] != '.':
                    used.add(board[i][j])
        
        return [str(i) for i in range(1, 10) if str(i) not in used]
    
    def find_best_cell(board):
        """Find empty cell with fewest candidates."""
        min_candidates = 10
        best_cell = None
        
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    candidates = get_candidates(board, i, j)
                    if len(candidates) < min_candidates:
                        min_candidates = len(candidates)
                        best_cell = (i, j, candidates)
        
        return best_cell
    
    def backtrack():
        cell_info = find_best_cell(board)
        if not cell_info:
            return True  # Solved
        
        row, col, candidates = cell_info
        
        for num in candidates:
            board[row][col] = num
            if backtrack():
                return True
            board[row][col] = '.'
        
        return False
    
    backtrack()
    return board
```

### 3. Word Search in Grid

**Problem**: Find if a word exists in a 2D grid of characters.

```python
def exist(board, word):
    """Check if word exists in board."""
    if not board or not board[0]:
        return False
    
    rows, cols = len(board), len(board[0])
    
    def backtrack(row, col, index):
        # Base case: found the word
        if index == len(word):
            return True
        
        # Check bounds and character match
        if (row < 0 or row >= rows or col < 0 or col >= cols or
            board[row][col] != word[index]):
            return False
        
        # Mark current cell as visited
        temp = board[row][col]
        board[row][col] = '#'
        
        # Explore all 4 directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        found = False
        for dr, dc in directions:
            if backtrack(row + dr, col + dc, index + 1):
                found = True
                break
        
        # Restore cell
        board[row][col] = temp
        return found
    
    # Try starting from each cell
    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True
    
    return False

def find_words(board, words):
    """Find all words that exist in board (Word Search II)."""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word = None
    
    # Build trie
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word
    
    result = []
    rows, cols = len(board), len(board[0])
    
    def backtrack(row, col, node):
        if node.word:
            result.append(node.word)
            node.word = None  # Avoid duplicates
        
        if (row < 0 or row >= rows or col < 0 or col >= cols or
            board[row][col] not in node.children):
            return
        
        char = board[row][col]
        board[row][col] = '#'
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr, dc in directions:
            backtrack(row + dr, col + dc, node.children[char])
        
        board[row][col] = char
    
    for i in range(rows):
        for j in range(cols):
            backtrack(i, j, root)
    
    return result
```

### 4. Rat in a Maze

**Problem**: Find path for rat to reach destination in a maze.

```python
def solve_maze(maze):
    """Find path in maze from (0,0) to (n-1,n-1)."""
    n = len(maze)
    solution = [[0] * n for _ in range(n)]
    
    def is_safe(x, y):
        return (0 <= x < n and 0 <= y < n and 
                maze[x][y] == 1 and solution[x][y] == 0)
    
    def backtrack(x, y):
        if x == n - 1 and y == n - 1:
            solution[x][y] = 1
            return True
        
        if is_safe(x, y):
            solution[x][y] = 1
            
            # Move right
            if backtrack(x, y + 1):
                return True
            
            # Move down
            if backtrack(x + 1, y):
                return True
            
            # Move left
            if backtrack(x, y - 1):
                return True
            
            # Move up
            if backtrack(x - 1, y):
                return True
            
            # Backtrack
            solution[x][y] = 0
        
        return False
    
    if backtrack(0, 0):
        return solution
    return None

def find_all_paths(maze):
    """Find all paths in maze."""
    n = len(maze)
    paths = []
    path = []
    visited = [[False] * n for _ in range(n)]
    
    def backtrack(x, y):
        if x == n - 1 and y == n - 1:
            path.append((x, y))
            paths.append(path[:])
            path.pop()
            return
        
        if (0 <= x < n and 0 <= y < n and 
            maze[x][y] == 1 and not visited[x][y]):
            
            visited[x][y] = True
            path.append((x, y))
            
            # Try all 4 directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                backtrack(x + dx, y + dy)
            
            path.pop()
            visited[x][y] = False
    
    backtrack(0, 0)
    return paths
```

## Advanced Techniques

### Constraint Propagation

- Reduce search space by enforcing constraints
- Use constraint satisfaction techniques
- Implement forward checking

### Optimization Strategies

1. **Most Constrained Variable**: Choose cell with fewest options
2. **Least Constraining Value**: Choose value that eliminates fewest options
3. **Arc Consistency**: Maintain consistency between constraints

### Pruning Techniques

- **Bound Checking**: Verify boundaries before recursion
- **Constraint Validation**: Check all constraints early
- **Symmetry Breaking**: Avoid equivalent solutions

## Applications

- Game solving (Chess, Sudoku, etc.)
- Pathfinding and navigation
- Layout and placement problems
- Circuit design and optimization
- Puzzle solving and game AI
