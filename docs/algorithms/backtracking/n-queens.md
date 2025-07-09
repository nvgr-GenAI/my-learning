# N-Queens Problem

## Overview

The N-Queens problem is a classic backtracking problem that involves placing N chess queens on an N×N chessboard so that no two queens threaten each other. This means that no two queens can share the same row, column, or diagonal.

## Problem Statement

Place N queens on an N×N chessboard such that no two queens attack each other (share the same row, column, or diagonal).

## Algorithm

The N-Queens problem can be efficiently solved using backtracking with the following approach:

1. Start with an empty N×N chessboard
2. Try placing queens one by one in different columns of the first row
3. When a queen is placed, check if it conflicts with any previously placed queens
4. If there's no conflict, recursively place queens in the subsequent rows
5. If all queens can't be placed without conflicts, backtrack by removing the queen from the current position and try the next position
6. If all rows are filled (N queens are placed successfully), a valid solution is found

## Implementation

### Python Implementation

```python
def solve_n_queens(n):
    """
    Solve the N-Queens problem.
    
    Args:
        n: Size of the board (N×N)
        
    Returns:
        List of all possible solutions
    """
    solutions = []
    
    # Initialize the board (empty)
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    def is_safe(row, col):
        """
        Check if it's safe to place a queen at board[row][col]
        """
        # Check the column (up)
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check upper-left diagonal
        for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
            if board[i][j] == 'Q':
                return False
        
        # Check upper-right diagonal
        for i, j in zip(range(row - 1, -1, -1), range(col + 1, n)):
            if board[i][j] == 'Q':
                return False
        
        return True
    
    def backtrack(row):
        """
        Backtracking function to place queens row by row
        """
        # If we've placed queens in all rows, we have a solution
        if row == n:
            # Convert board to solution format
            solution = [''.join(row) for row in board]
            solutions.append(solution)
            return
        
        # Try placing a queen in each column of the current row
        for col in range(n):
            if is_safe(row, col):
                # Place the queen
                board[row][col] = 'Q'
                
                # Recursively place queens in the next row
                backtrack(row + 1)
                
                # Backtrack: remove the queen to try other positions
                board[row][col] = '.'
    
    # Start the backtracking process from the first row
    backtrack(0)
    
    return solutions

# Example usage
n = 4
solutions = solve_n_queens(n)
print(f"Found {len(solutions)} solutions for {n}-Queens problem:")

for i, solution in enumerate(solutions, 1):
    print(f"\nSolution {i}:")
    for row in solution:
        print(row)

# More efficient implementation using sets to track conflicts
def solve_n_queens_optimized(n):
    """
    Optimized solution for the N-Queens problem.
    
    Args:
        n: Size of the board (N×N)
        
    Returns:
        List of all possible solutions
    """
    solutions = []
    
    # Use sets to keep track of occupied columns and diagonals
    cols = set()
    pos_diag = set()  # (row + col) is constant for a positive diagonal
    neg_diag = set()  # (row - col) is constant for a negative diagonal
    
    # Board representation (positions of queens in each row)
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    def backtrack(row):
        if row == n:
            # Convert board to solution format
            solution = [''.join(row) for row in board]
            solutions.append(solution)
            return
        
        for col in range(n):
            # Check if the position is under attack
            if col in cols or (row + col) in pos_diag or (row - col) in neg_diag:
                continue
            
            # Place the queen
            board[row][col] = 'Q'
            cols.add(col)
            pos_diag.add(row + col)
            neg_diag.add(row - col)
            
            # Proceed to the next row
            backtrack(row + 1)
            
            # Backtrack
            board[row][col] = '.'
            cols.remove(col)
            pos_diag.remove(row + col)
            neg_diag.remove(row - col)
    
    backtrack(0)
    return solutions

# Example usage of optimized solution
solutions_opt = solve_n_queens_optimized(4)
print(f"\nOptimized solution found {len(solutions_opt)} solutions")
```

### Java Implementation

```java
import java.util.*;

public class NQueens {
    private List<List<String>> solutions;
    private char[][] board;
    private int n;
    
    public List<List<String>> solveNQueens(int n) {
        this.solutions = new ArrayList<>();
        this.n = n;
        this.board = new char[n][n];
        
        // Initialize the board with empty cells
        for (int i = 0; i < n; i++) {
            Arrays.fill(board[i], '.');
        }
        
        backtrack(0);
        return solutions;
    }
    
    private void backtrack(int row) {
        // If we've placed queens in all rows, we have a solution
        if (row == n) {
            // Convert board to solution format
            List<String> solution = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                solution.add(new String(board[i]));
            }
            solutions.add(solution);
            return;
        }
        
        // Try placing a queen in each column of the current row
        for (int col = 0; col < n; col++) {
            if (isSafe(row, col)) {
                // Place the queen
                board[row][col] = 'Q';
                
                // Recursively place queens in the next row
                backtrack(row + 1);
                
                // Backtrack: remove the queen to try other positions
                board[row][col] = '.';
            }
        }
    }
    
    private boolean isSafe(int row, int col) {
        // Check column (up)
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }
        
        // Check upper-left diagonal
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        
        // Check upper-right diagonal
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        
        return true;
    }
    
    // Optimized solution using sets
    public List<List<String>> solveNQueensOptimized(int n) {
        List<List<String>> result = new ArrayList<>();
        char[][] board = new char[n][n];
        
        for (int i = 0; i < n; i++) {
            Arrays.fill(board[i], '.');
        }
        
        Set<Integer> cols = new HashSet<>();
        Set<Integer> posDiag = new HashSet<>(); // (r + c)
        Set<Integer> negDiag = new HashSet<>(); // (r - c)
        
        backtrackOptimized(0, cols, posDiag, negDiag, board, result, n);
        return result;
    }
    
    private void backtrackOptimized(int row, Set<Integer> cols, Set<Integer> posDiag, 
                                   Set<Integer> negDiag, char[][] board, 
                                   List<List<String>> result, int n) {
        if (row == n) {
            List<String> solution = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                solution.add(new String(board[i]));
            }
            result.add(solution);
            return;
        }
        
        for (int col = 0; col < n; col++) {
            int posD = row + col;
            int negD = row - col;
            
            // Check if the position is under attack
            if (cols.contains(col) || posDiag.contains(posD) || negDiag.contains(negD)) {
                continue;
            }
            
            // Place the queen
            board[row][col] = 'Q';
            cols.add(col);
            posDiag.add(posD);
            negDiag.add(negD);
            
            // Move to next row
            backtrackOptimized(row + 1, cols, posDiag, negDiag, board, result, n);
            
            // Backtrack
            board[row][col] = '.';
            cols.remove(col);
            posDiag.remove(posD);
            negDiag.remove(negD);
        }
    }
    
    public static void main(String[] args) {
        NQueens solution = new NQueens();
        int n = 4;
        
        List<List<String>> solutions = solution.solveNQueens(n);
        System.out.println("Found " + solutions.size() + " solutions for " + n + "-Queens problem:");
        
        for (int i = 0; i < solutions.size(); i++) {
            System.out.println("\nSolution " + (i + 1) + ":");
            List<String> sol = solutions.get(i);
            for (String row : sol) {
                System.out.println(row);
            }
        }
        
        // Test optimized solution
        List<List<String>> solutionsOpt = solution.solveNQueensOptimized(n);
        System.out.println("\nOptimized solution found " + solutionsOpt.size() + " solutions");
    }
}
```

## Complexity Analysis

- **Time Complexity**: O(N!), where N is the size of the board
  - In the worst case, we have to explore N positions for the first queen, (N-1) for the second, and so on
- **Space Complexity**: O(N²) for storing the board, plus O(N) for the recursion stack
  - Optimized version uses O(N) additional space for the sets that track conflicts

## Optimization Techniques

1. **Using Sets for Conflict Detection**: Instead of checking all previous rows for conflicts each time, we can use sets to keep track of occupied columns and diagonals
2. **Bit Manipulation**: For even more efficient conflict checking, we can use bitsets instead of arrays or sets
3. **Symmetry Reduction**: We can reduce the search space by recognizing symmetrical board positions

## Variations

1. **N-Queens Counting Problem**: Count the number of solutions without generating them all
2. **N-Queens Completion Problem**: Place additional queens on a board that already has some queens
3. **N-Queens on Non-standard Boards**: Solve the problem on boards of different shapes
4. **N-Rooks Problem**: Similar problem with rooks instead of queens (much easier as rooks only attack along rows and columns)
5. **N-Bishops Problem**: Place N bishops so none attack each other

## Applications

1. **Circuit Design**: Placement of components to avoid interference
2. **Task Scheduling**: Allocating resources without conflicts
3. **VLSI Design**: Placement of modules on a chip
4. **Parallel Memory Systems**: Avoiding memory bank conflicts
5. **Network Communications**: Avoiding signal interference

## Practice Problems

1. [N-Queens](https://leetcode.com/problems/n-queens/) - Generate all solutions for the N-Queens problem
2. [N-Queens II](https://leetcode.com/problems/n-queens-ii/) - Count the number of solutions
3. [Valid Sudoku](https://leetcode.com/problems/valid-sudoku/) - Similar constraint satisfaction problem
4. [Sudoku Solver](https://leetcode.com/problems/sudoku-solver/) - Another classic backtracking problem

## References

1. Golomb, S. W., & Baumert, L. D. (1965). "Backtrack programming". Journal of the ACM, 12(4), 516-524.
2. Bell, J., & Stevens, B. (2009). "A survey of known results and research areas for n-queens". Discrete Mathematics, 309(1), 1-31.
3. Gent, I. P., Jefferson, C., & Nightingale, P. (2017). "Complexity of n-Queens Completion". Journal of Artificial Intelligence Research, 59, 815-848.
