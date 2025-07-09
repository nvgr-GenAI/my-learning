# Sudoku Solver

## Overview

The Sudoku Solver is a classic application of backtracking that solves the Sudoku puzzle. A standard Sudoku puzzle is a 9×9 grid partially filled with digits from 1 to 9. The goal is to fill the remaining cells such that each row, each column, and each of the nine 3×3 subgrids contains all digits from 1 to 9 without repetition.

## Problem Statement

Given a partially filled 9×9 Sudoku grid, complete the grid such that each row, each column, and each 3×3 subgrid contains the digits from 1 to 9 exactly once.

## Algorithm

The backtracking approach to solve Sudoku involves:

1. Find an empty cell in the grid (if none, the puzzle is solved)
2. Try placing digits 1 through 9 in the empty cell
3. For each digit, check if it can be validly placed in that cell (no conflicts with row, column, or subgrid)
4. If a digit can be validly placed, recursively attempt to solve the rest of the grid
5. If the recursive call is successful, the puzzle is solved
6. If no digit can be validly placed or if the recursive call fails, backtrack by removing the digit and try another option

## Implementation

### Python Implementation

```python
def solve_sudoku(board):
    """
    Solves a Sudoku puzzle using backtracking.
    
    Args:
        board: 9x9 Sudoku grid (list of lists) with empty cells represented as '.'
        
    Returns:
        True if the Sudoku is solvable, False otherwise.
        The board is modified in-place with the solution.
    """
    # Find an empty cell
    empty_cell = find_empty(board)
    
    # If no empty cell is found, the puzzle is solved
    if not empty_cell:
        return True
    
    row, col = empty_cell
    
    # Try digits 1-9
    for digit in range(1, 10):
        digit_str = str(digit)
        
        # Check if the digit can be placed
        if is_valid(board, row, col, digit_str):
            # Place the digit
            board[row][col] = digit_str
            
            # Recursively try to solve the rest
            if solve_sudoku(board):
                return True
            
            # If the recursive call failed, backtrack
            board[row][col] = '.'
    
    # No solution found with any digit
    return False

def find_empty(board):
    """
    Finds an empty cell in the Sudoku grid.
    
    Args:
        board: 9x9 Sudoku grid
        
    Returns:
        Tuple (row, col) of an empty cell, or None if no empty cell exists
    """
    for row in range(9):
        for col in range(9):
            if board[row][col] == '.':
                return (row, col)
    return None

def is_valid(board, row, col, digit):
    """
    Checks if placing a digit at the specified position is valid.
    
    Args:
        board: 9x9 Sudoku grid
        row: Row index
        col: Column index
        digit: Digit to check
        
    Returns:
        True if the placement is valid, False otherwise
    """
    # Check row
    for x in range(9):
        if board[row][x] == digit:
            return False
    
    # Check column
    for x in range(9):
        if board[x][col] == digit:
            return False
    
    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == digit:
                return False
    
    return True

# Example usage
example_board = [
    ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"]
]

print("Sudoku Puzzle:")
for row in example_board:
    print(" ".join(row))

if solve_sudoku(example_board):
    print("\nSudoku Solution:")
    for row in example_board:
        print(" ".join(row))
else:
    print("\nNo solution exists.")
```

### Java Implementation

```java
public class SudokuSolver {
    
    public static boolean solveSudoku(char[][] board) {
        // Find an empty cell
        int[] emptyCell = findEmpty(board);
        
        // If no empty cell is found, the puzzle is solved
        if (emptyCell == null) {
            return true;
        }
        
        int row = emptyCell[0];
        int col = emptyCell[1];
        
        // Try digits 1-9
        for (char digit = '1'; digit <= '9'; digit++) {
            // Check if the digit can be placed
            if (isValid(board, row, col, digit)) {
                // Place the digit
                board[row][col] = digit;
                
                // Recursively try to solve the rest
                if (solveSudoku(board)) {
                    return true;
                }
                
                // If the recursive call failed, backtrack
                board[row][col] = '.';
            }
        }
        
        // No solution found with any digit
        return false;
    }
    
    private static int[] findEmpty(char[][] board) {
        for (int row = 0; row < 9; row++) {
            for (int col = 0; col < 9; col++) {
                if (board[row][col] == '.') {
                    return new int[] {row, col};
                }
            }
        }
        return null;
    }
    
    private static boolean isValid(char[][] board, int row, int col, char digit) {
        // Check row
        for (int x = 0; x < 9; x++) {
            if (board[row][x] == digit) {
                return false;
            }
        }
        
        // Check column
        for (int x = 0; x < 9; x++) {
            if (board[x][col] == digit) {
                return false;
            }
        }
        
        // Check 3x3 box
        int boxRow = 3 * (row / 3);
        int boxCol = 3 * (col / 3);
        for (int r = boxRow; r < boxRow + 3; r++) {
            for (int c = boxCol; c < boxCol + 3; c++) {
                if (board[r][c] == digit) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    // Print the Sudoku board
    private static void printBoard(char[][] board) {
        for (int i = 0; i < 9; i++) {
            if (i % 3 == 0 && i != 0) {
                System.out.println("------+-------+------");
            }
            for (int j = 0; j < 9; j++) {
                if (j % 3 == 0 && j != 0) {
                    System.out.print("| ");
                }
                System.out.print(board[i][j] + " ");
            }
            System.out.println();
        }
    }
    
    public static void main(String[] args) {
        char[][] board = {
            {'5', '3', '.', '.', '7', '.', '.', '.', '.'},
            {'6', '.', '.', '1', '9', '5', '.', '.', '.'},
            {'.', '9', '8', '.', '.', '.', '.', '6', '.'},
            {'8', '.', '.', '.', '6', '.', '.', '.', '3'},
            {'4', '.', '.', '8', '.', '3', '.', '.', '1'},
            {'7', '.', '.', '.', '2', '.', '.', '.', '6'},
            {'.', '6', '.', '.', '.', '.', '2', '8', '.'},
            {'.', '.', '.', '4', '1', '9', '.', '.', '5'},
            {'.', '.', '.', '.', '8', '.', '.', '7', '9'}
        };
        
        System.out.println("Sudoku Puzzle:");
        printBoard(board);
        
        if (solveSudoku(board)) {
            System.out.println("\nSudoku Solution:");
            printBoard(board);
        } else {
            System.out.println("\nNo solution exists.");
        }
    }
}
```

## Complexity Analysis

- **Time Complexity**: O(9^(N*N)), where N is the board size (9 for standard Sudoku)
  - In the worst case, we need to try all possible digits for each empty cell
  - The actual time is much less in practice due to constraints eliminating many possibilities
- **Space Complexity**: O(N²) for the board, plus O(N²) for the recursion stack in the worst case

## Optimization Techniques

1. **Constraint Propagation**: Use techniques like "naked singles" or "hidden singles" to reduce the search space
2. **Minimum Remaining Values (MRV) Heuristic**: Choose cells with fewer valid options first
3. **Degree Heuristic**: Choose cells that constrain the most other cells
4. **Bitmasks**: Use bitwise operations for more efficient checking of valid digits
5. **Dancing Links**: A specialized technique for exact cover problems like Sudoku

## Variations

1. **Different Board Sizes**: 4×4, 16×16, or other sizes
2. **Irregular Sudoku**: The 3×3 subgrids are replaced with irregularly shaped regions
3. **Hyper Sudoku**: Additional constraints on specified regions
4. **Samurai Sudoku**: Five overlapping Sudoku puzzles
5. **3D Sudoku**: Three-dimensional version with additional constraints

## Applications

1. **Puzzle Games**: Sudoku and similar constraint satisfaction puzzles
2. **Timetabling**: Scheduling classes, exams, or meetings
3. **Graph Coloring**: Assigning colors to graph nodes with constraints
4. **VLSI Design**: Component placement with constraints
5. **Automated Planning**: Planning with resource constraints

## Practice Problems

1. [Sudoku Solver](https://leetcode.com/problems/sudoku-solver/) - Solve a given Sudoku board
2. [Valid Sudoku](https://leetcode.com/problems/valid-sudoku/) - Determine if a partially filled Sudoku board is valid
3. [Word Search II](https://leetcode.com/problems/word-search-ii/) - Another backtracking problem with a grid
4. [N-Queens](https://leetcode.com/problems/n-queens/) - Classic backtracking problem with constraints

## References

1. Knuth, D. E. (2000). "Dancing Links". Millenial Perspectives in Computer Science, 187-214.
2. Norvig, P. (2006). "Solving Every Sudoku Puzzle". [Online](http://norvig.com/sudoku.html).
3. Simonis, H. (2005). "Sudoku as a Constraint Problem". In CP Workshop on Modeling and Reformulating Constraint Satisfaction Problems.
4. Lynce, I., & Ouaknine, J. (2006). "Sudoku as a SAT Problem". In Proceedings of the 9th International Symposium on Artificial Intelligence and Mathematics.
