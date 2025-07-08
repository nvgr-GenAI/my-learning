# Buy/Sell Stock Pattern

## Introduction

The Buy/Sell Stock pattern is a dynamic programming approach for optimization problems involving sequential decision-making, particularly in financial contexts where we need to determine when to buy and sell assets.

=== "Overview"
    **Core Idea**: Make sequential decisions (buy/sell/hold) to maximize profit or minimize loss.
    
    **When to Use**:
    
    - When modeling financial transactions with buying and selling
    - When each state depends on previous decisions
    - When working with problems requiring decisions with cooldown periods
    - When handling problems with transaction limits
    
    **Recurrence Relations**: Vary by problem variant, typically:
    
    - Buy state: `buy[i] = max(buy[i-1], sell[i-2] - price[i])` (hold previous buy state or buy after cooldown)
    - Sell state: `sell[i] = max(sell[i-1], buy[i-1] + price[i])` (hold previous sell state or sell current stock)
    
    **Real-World Applications**:
    
    - Stock trading algorithms
    - Asset management strategies
    - Resource allocation over time
    - Inventory management systems
    - Option pricing models

=== "Example Problems"
    - **Best Time to Buy and Sell Stock I**: One transaction allowed
      - Problem: Find the maximum profit by buying and selling a stock once
      - Approach: Track the minimum price seen so far and maximize profit
    
    - **Best Time to Buy and Sell Stock II**: Unlimited transactions
      - Problem: Find maximum profit with unlimited transactions
      - Approach: Capture all profitable price differences (greedy works here)
    
    - **Best Time to Buy and Sell Stock III**: Limited to 2 transactions
      - Problem: Find maximum profit with at most two transactions
      - Approach: Track the state after 0, 1, or 2 transactions
    
    - **Best Time to Buy and Sell Stock IV**: Limited to k transactions
      - Problem: Find maximum profit with at most k transactions
      - Approach: Generalize the state to handle k transactions
    
    - **Best Time to Buy and Sell Stock with Cooldown**: Cooldown period after selling
      - Problem: Find maximum profit with cooldown of 1 day after selling
      - Approach: Add a cooldown state in the transitions
    
    - **Best Time to Buy and Sell Stock with Transaction Fee**: Fee for each transaction
      - Problem: Find maximum profit with a transaction fee
      - Approach: Subtract the fee when calculating profit from selling

=== "Visualization"
    For the basic Buy and Sell Stock problem with prices [7, 1, 5, 3, 6, 4]:
    
    ```text
    Prices:  [7, 1, 5, 3, 6, 4]
    Min:     [7, 1, 1, 1, 1, 1]  (minimum price seen so far)
    Profit:  [0, 0, 4, 2, 5, 3]  (max profit possible at each day)
    ```
    
    Maximum profit: 5 (buy at 1, sell at 6)
    
    For Buy and Sell Stock with multiple transactions:
    
    ```text
    Prices: [7, 1, 5, 3, 6, 4]
    
    Day 0:
      Hold: 0
      Buy: -7
    
    Day 1:
      Hold: 0
      Buy: -1 (update min cost basis)
    
    Day 2:
      Hold: 4 (sell at 5 after buying at 1)
      Buy: -1 (keep holding)
    
    Day 3:
      Hold: 4 (keep holding)
      Buy: -1 (keep holding)
    
    Day 4:
      Hold: 5 (sell at 6 after buying at 1)
      Buy: -1 (keep holding)
    
    Day 5:
      Hold: 5 (keep holding)
      Buy: -1 (keep holding)
    ```
    
    Maximum profit: 5 (buy at 1, sell at 6)
    
    ![Buy Sell Stock Visualization](https://i.imgur.com/HPmN8YA.png)

=== "Implementation"
    **Buy and Sell Stock I** (one transaction):
    
    ```python
    def maxProfit(prices):
        if not prices:
            return 0
        
        max_profit = 0
        min_price = prices[0]
        
        for price in prices:
            # Update the minimum price seen so far
            min_price = min(min_price, price)
            # Update the maximum profit if selling at current price
            max_profit = max(max_profit, price - min_price)
        
        return max_profit
    
    # Time Complexity: O(n)
    # Space Complexity: O(1)
    ```
    
    **Buy and Sell Stock II** (unlimited transactions):
    
    ```python
    def maxProfitII(prices):
        profit = 0
        
        for i in range(1, len(prices)):
            # If price increases, we can make profit
            if prices[i] > prices[i-1]:
                profit += prices[i] - prices[i-1]
        
        return profit
    
    # Time Complexity: O(n)
    # Space Complexity: O(1)
    ```
    
    **Buy and Sell Stock with Cooldown**:
    
    ```python
    def maxProfitWithCooldown(prices):
        if not prices or len(prices) < 2:
            return 0
        
        n = len(prices)
        # States: buy[i] = max profit ending with a buy or wait
        #         sell[i] = max profit ending with a sell or wait
        #         cool[i] = max profit ending with a cooldown
        buy = [0] * n
        sell = [0] * n
        cool = [0] * n
        
        # Initialize
        buy[0] = -prices[0]  # Spend money to buy
        sell[0] = 0  # Cannot sell without buying
        cool[0] = 0  # Start with 0
        
        for i in range(1, n):
            # Buy after cooldown or continue to hold
            buy[i] = max(cool[i-1] - prices[i], buy[i-1])
            # Sell what we've bought or continue to wait
            sell[i] = max(buy[i-1] + prices[i], sell[i-1])
            # Cooldown after selling or continue cooldown
            cool[i] = max(sell[i-1], cool[i-1])
        
        # The final state is max of selling or cooldown
        return max(sell[n-1], cool[n-1])
    
    # Time Complexity: O(n)
    # Space Complexity: O(n)
    ```
    
    **Space-Optimized Version**:
    
    ```python
    def maxProfitWithCooldown_optimized(prices):
        if not prices or len(prices) < 2:
            return 0
        
        # Use variables instead of arrays
        buy, sell, cool = -prices[0], 0, 0
        
        for i in range(1, len(prices)):
            prev_buy = buy
            prev_sell = sell
            prev_cool = cool
            
            buy = max(prev_cool - prices[i], prev_buy)
            sell = max(prev_buy + prices[i], prev_sell)
            cool = max(prev_sell, prev_cool)
        
        return max(sell, cool)
    
    # Time Complexity: O(n)
    # Space Complexity: O(1)
    ```

=== "Tips and Insights"
    - **State Definition**: Define clear states for different phases (holding stock, not holding stock, etc.)
    - **Transaction Limits**: When k is large (k >= n/2), the problem reduces to the unlimited transactions case
    - **State Transition Visualization**: Drawing a state transition diagram helps clarify the logic
    - **Initialization**: Pay attention to initial state values, especially for buying
    - **Space Optimization**: Most variants can be optimized to use O(1) space by keeping only previous state variables
    - **Greedy vs DP**: For unlimited transactions without cooldown or fees, a greedy approach works
    - **Common Patterns**:
      - Tracking minimum purchase price
      - Maintaining separate states for holding and not holding stock
      - Using offset variables for cooldown periods
    - **Generalizing**: The pattern can be extended to other resource allocation problems with temporal constraints
    - **Edge Cases**: Handle empty arrays and single-element arrays carefully
