# Game Theory DP Pattern

## Introduction

The Game Theory Dynamic Programming pattern deals with problems where two players take turns making optimal moves, and the goal is to determine the outcome of the game with perfect play from both sides. This pattern is particularly useful in competitive scenarios and decision-making processes.

=== "Overview"
    **Core Idea**: Determine the optimal strategy in two-player games where players take turns, and each makes the best possible move at their turn.
    
    **When to Use**:
    
    - When analyzing games with alternating turns
    - When each player must make an optimal decision
    - When the outcome depends on a sequence of choices by both players
    - When working with competitive optimization problems
    
    **Recurrence Relation**: 
    - For maximizing player: `dp[i][j] = max(choice1, choice2, ...)`
    - For minimizing player: `dp[i][j] = min(choice1, choice2, ...)`
    
    **Real-World Applications**:
    
    - AI for board games like chess, checkers, or Go
    - Economic models of competition
    - Auction strategy optimization
    - Resource allocation in competitive environments
    - Negotiation strategy modeling

=== "Example Problems"
    - **Nim Game**: Players take turns removing 1-3 stones from a pile
      - Problem: Determine if the first player can win with perfect play
      - Strategy: Win by leaving multiples of 4 stones to the opponent
    
    - **Stone Game**: Two players take turns picking stones from either end of an array
      - Problem: Can the first player win if each player picks optimally?
      - Insight: Consider the parity of positions and sum differences
    
    - **Predict the Winner**: Players take turns picking numbers from either end of an array
      - Problem: Will the first player win given the array of numbers?
      - Approach: Compare the final scores of both players
    
    - **Coin Game**: Players take coins from either end, trying to maximize their sum
      - Variation: Optimize for maximum value collected
      - Application: Shows how to handle value optimization in turn-based games
    
    - **Divisor Game**: Players take turns selecting factors of a number
      - Problem: The first player chooses x (a divisor of n), then n becomes n-x
      - Analysis: Even/odd pattern determines the winner

=== "Visualization"
    For the Stone Game with piles [5, 3, 4, 5]:
    
    ```text
    dp[i][j] = maximum score difference (my score - opponent score) 
               for the subgame from index i to j
    
    dp table:
    
         | 5 | 3 | 4 | 5 |
    -----|---|---|---|---|
      5  | 5 | 2 | 4 | 1 |
    -----|---|---|---|---|
      3  | - | 3 | 1 | 4 |
    -----|---|---|---|---|
      4  | - | - | 4 | 1 |
    -----|---|---|---|---|
      5  | - | - | - | 5 |
    
    The first player can win by 1 point (dp[0][3] = 1 > 0)
    ```
    
    Optimal play:
    - Player 1 takes 5 from the right
    - Player 2 takes 4
    - Player 1 takes 5 from the left
    - Player 2 takes 3
    - Final score: Player 1 = 10, Player 2 = 7
    
    ![Game Theory DP Visualization](https://i.imgur.com/tHJPDm4.png)

=== "Implementation"
    **Stone Game Solution**:
    
    ```python
    def stone_game(piles):
        n = len(piles)
        # dp[i][j] = maximum score difference for piles[i...j]
        dp = [[0] * n for _ in range(n)]
        
        # Base case: single pile
        for i in range(n):
            dp[i][i] = piles[i]
        
        # Fill the dp table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = max(
                    piles[i] - dp[i+1][j],  # Take from left
                    piles[j] - dp[i][j-1]   # Take from right
                )
        
        # Positive score difference means first player wins
        return dp[0][n-1] > 0
    ```
    
    **Nim Game Solution**:
    
    ```python
    def can_win_nim(n):
        # First player loses if and only if n is divisible by 4
        return n % 4 != 0
    ```
    
    **Predict the Winner**:
    
    ```python
    def predict_winner(nums):
        n = len(nums)
        # For small arrays, can use the same approach as stone game
        if n <= 2:
            return True
        
        # dp[i][j] = max score difference for subarray nums[i...j]
        dp = [[0] * n for _ in range(n)]
        
        # Base case: single number
        for i in range(n):
            dp[i][i] = nums[i]
        
        # Fill dp table diagonally
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = max(
                    nums[i] - dp[i+1][j],
                    nums[j] - dp[i][j-1]
                )
        
        # First player wins if final score difference is non-negative
        return dp[0][n-1] >= 0
    ```
    
    **Coin Game Variation (Maximize Total Value)**:
    
    ```python
    def max_coin_game(coins):
        n = len(coins)
        # dp[i][j][0] = max value for player 1 in range [i,j]
        # dp[i][j][1] = max value for player 2 in range [i,j]
        dp = [[[0, 0] for _ in range(n)] for _ in range(n)]
        
        # Base case: single coin
        for i in range(n):
            dp[i][i][0] = coins[i]  # First player gets the coin
            dp[i][i][1] = 0         # Second player gets nothing
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Player 1 takes left coin
                take_left = coins[i] + dp[i+1][j][1]
                # Player 1 takes right coin
                take_right = coins[j] + dp[i][j-1][1]
                
                if take_left > take_right:
                    dp[i][j][0] = take_left
                    dp[i][j][1] = dp[i+1][j][0]  # Player 2's best response
                else:
                    dp[i][j][0] = take_right
                    dp[i][j][1] = dp[i][j-1][0]  # Player 2's best response
        
        return dp[0][n-1][0]  # Maximum value player 1 can get
    ```

=== "Tips and Insights"
    - **Zero-Sum vs. Non-Zero-Sum Games**:
      - Zero-sum: One player's gain is the other's loss
      - Non-zero-sum: Total payoffs can vary
    - **State Representation**:
      - Single value: Maximum score difference (player1 - player2)
      - Pair of values: (player1 score, player2 score)
    - **Minimax Principle**:
      - Current player maximizes their score
      - Opponent will minimize current player's score
    - **Pattern Recognition**:
      - Even/odd patterns often emerge in simple games
      - For many games, the first player has a winning strategy
    - **Optimization Techniques**:
      - Alpha-beta pruning can speed up game tree search
      - Memoization is crucial for recursive implementations
    - **Common Game Types**:
      - Take-away games (Nim, Chomp)
      - Coin/stone picking games
      - Divisor/factor games
    - **Analyzing Game Outcomes**:
      - Look for invariants that indicate winning positions
      - Sometimes parity analysis reveals the pattern
      - Symmetry can be exploited in certain games
    - **Implementation Approaches**:
      - Top-down with memoization
      - Bottom-up by building from smaller subgames
      - Sometimes greedy strategies work
    - **Complexity Considerations**:
      - Time: Typically O(n²) for linear games, higher for more complex games
      - Space: Usually O(n²) for the DP table
    - **Mathematical Connections**:
      - Sprague-Grundy theory for impartial games
      - Combinatorial Game Theory for deeper analysis

## Pattern Recognition

The Game Theory DP pattern appears when:

1. **Turn-based decision making** with optimal play from both sides
2. **Zero-sum or competitive scenarios** where players have opposing goals
3. **Problems involving maximizing or minimizing** an outcome against an opponent
4. **Recursive game structures** where the state changes after each move
